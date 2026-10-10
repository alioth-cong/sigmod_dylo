#!/usr/bin/env python3
"""Segment-parallel version of leiden_static.py, for graphs of LiveJournal's size.

Motivation: leiden_static is the rebuild baseline -- every batch reruns Leiden from scratch on the accumulated graph. On LiveJournal
one batch measured 1,213 s (38.56M edges), so 200 batches take 67 hours serially.

Key observation: from-scratch rebuilds of different batches are **independent**; batch k depends only on initial + inc_1..inc_k.
So the batches can be split into segments by k and run in parallel. Since la.find_partition(..., seed=42) is deterministic,
the parallel results are bit-identical to the serial ones.

Segments rather than per-batch dispatch: each worker reads initial.txt once (686MB, expensive to parse in pure Python)
and then adds edges incrementally over its contiguous range, reusing leiden_static.py's logic.
Worker w handles [a_w, b_w]: it first reads initial + inc_1..inc_{a_w}, then runs each k in turn.

Usage: leiden_static_par.py <initial> <incBase> <numBatches> <outDir> [--workers N]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igraph as ig
import leidenalg as la

sys.path.insert(0, str(Path(__file__).parent))
from leiden_static import (build_graph, extend_filename,  # noqa: E402
                           read_edges_uvw, write_communities)


def segment(args) -> list:
    """Run the segment [lo, hi] and return (k, latency_ms, Q, n_comms) per batch."""
    initial, inc_base, lo, hi, out_dir = args
    out_dir = Path(out_dir)

    cumulative = read_edges_uvw(Path(initial))
    for k in range(1, lo + 1):                     # catch up to the start of this segment
        cumulative.extend(read_edges_uvw(Path(extend_filename(inc_base, f"_{k}"))))

    rows = []
    for k in range(lo, hi + 1):
        if k > lo:                                  # the edges of batch lo have already been added
            cumulative.extend(read_edges_uvw(Path(extend_filename(inc_base, f"_{k}"))))
        t1 = time.time()
        g, node_list = build_graph(cumulative)
        part = la.find_partition(g, la.ModularityVertexPartition, seed=42)
        t2 = time.time()
        write_communities(part, node_list, out_dir / f"com_{k}.txt")
        rows.append((k, int((t2 - t1) * 1000), part.modularity,
                     len(set(part.membership))))
        del g, part, node_list
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("initial")
    ap.add_argument("inc_base")
    ap.add_argument("n_batches", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=25)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nb = a.n_batches
    W = min(a.workers, nb + 1)

    # split 0..nb evenly into W contiguous segments
    bounds, per, rem = [], (nb + 1) // W, (nb + 1) % W
    s = 0
    for i in range(W):
        n = per + (1 if i < rem else 0)
        bounds.append((s, s + n - 1))
        s += n
    print(f"{nb+1} batches split into {W} parallel segments of {per}~{per+1} batches", flush=True)

    t0 = time.time()
    allrows = []
    with ProcessPoolExecutor(W) as ex:
        futs = {ex.submit(segment, (a.initial, a.inc_base, lo, hi, str(out_dir))): (lo, hi)
                for lo, hi in bounds}
        for n, f in enumerate(as_completed(futs), 1):
            lo, hi = futs[f]
            rows = f.result()
            allrows.extend(rows)
            print(f"  [{n}/{W}] segment {lo}-{hi} done, "
                  f"mean per batch {sum(r[1] for r in rows)/len(rows)/1000:.0f}s "
                  f"({time.time()-t0:.0f}s)", flush=True)

    allrows.sort()
    with open(out_dir / "metrics.tsv", "w") as g:
        g.write("batch\tlatency_ms\tmodularity\tnum_communities\n")
        for k, lat, q, nc in allrows:
            g.write(f"{k}\t{lat}\t{q}\t{nc}\n")
    print(f"done: {len(allrows)} batches in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
