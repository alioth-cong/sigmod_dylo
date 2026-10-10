#!/usr/bin/env python3
"""
Build prefixes of LiveJournal for the scalability sweep.

Procedure:
  - Read all LJ edges from data/edges/livejournal.tsv
  - Shuffle with the same seed (42) the main prep uses
  - For each requested prefix size N: take first N edges, do 90/10 split into
    initial + 10 batches of 1000 edges each (10K stream window — small enough
    to bound timing on huge graphs while still measuring per-batch behaviour).
  - Output under data/runs/lj_<N_in_M>M/{initial.txt, inc_<k>.txt, meta.json}.

Usage:
  prep_lj_prefix.py [--prefixes 8,17,34,42] [--n-stream-batches 10]

Note: SNAP's soc-LiveJournal1 has 68.99M directed edges; symmetrized/deduped
undirected has ~42.85M edges. The spec lists prefixes ~{8M, 17M, 34M, 51M, 68M};
we approximate with {8M, 17M, 34M, 42M} (42M is our undirected total).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import json
import random
import sys
from pathlib import Path

SEED = 42
BATCH_SIZE = 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="8,17,34,42")
    ap.add_argument("--n-stream-batches", type=int, default=10)
    ap.add_argument("--in-edges", default=None)
    args = ap.parse_args()
    prefixes = [int(p) for p in args.prefixes.split(",")]
    root = Path(DYLO_ROOT + "")
    in_path = Path(args.in_edges or root/"data"/"edges"/"livejournal.tsv")

    print(f"[lj_prefix] loading {in_path} ...", file=sys.stderr)
    edges = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u, v = line.split("\t")[:2]
            edges.append((int(u), int(v)))
    print(f"[lj_prefix] total edges: {len(edges)}", file=sys.stderr)

    rng = random.Random(SEED)
    rng.shuffle(edges)

    for pN_m in prefixes:
        pN = pN_m * 1_000_000
        if pN > len(edges):
            pN_m_eff = len(edges) // 1_000_000
            print(f"[lj_prefix] {pN_m}M > available {len(edges)/1e6:.1f}M; capping at {pN_m_eff}M", file=sys.stderr)
            pN = len(edges)
            pN_m_eff = pN_m
        else:
            pN_m_eff = pN_m
        sub = edges[:pN]
        # 90/10 split, but cap stream at n_stream_batches * BATCH_SIZE
        cut = int(len(sub) * 0.9)
        initial = sub[:cut]
        stream  = sub[cut: cut + args.n_stream_batches * BATCH_SIZE]
        n_batches = len(stream) // BATCH_SIZE
        # Remap ONLY the edges we actually keep (initial + capped stream) to dense
        # 0..N-1 so downstream int32 tools (Java/C++) are not sized to the full
        # LiveJournal id space (~4.85M). Remapping the whole prefix would leave
        # gaps from the dropped tail edges; remap the kept set so it is truly dense
        # and the scalability curve is not distorted by uneven array oversizing.
        kept = initial + stream
        knodes = sorted({n for u, v in kept for n in (u, v)})
        fwd = {n: i for i, n in enumerate(knodes)}
        initial = [(fwd[u], fwd[v]) for u, v in initial]
        stream  = [(fwd[u], fwd[v]) for u, v in stream]

        out_dir = root/"data"/"runs"/f"lj_{pN_m_eff}M"
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir/"initial.txt", "w") as g:
            for u,v in initial:
                g.write(f"{u}\t{v}\t1.0\n")
        with open(out_dir/"all_edges.txt", "w") as g:
            for u,v in initial + stream:
                g.write(f"{u}\t{v}\t1.0\n")
        for k in range(n_batches):
            chunk = stream[k*BATCH_SIZE:(k+1)*BATCH_SIZE]
            with open(out_dir/f"inc_{k+1}.txt", "w") as g:
                for u,v in chunk:
                    g.write(f"{u}\t{v}\t{k+1}\n")

        nodes = set()
        for u,v in initial + stream:
            nodes.add(u); nodes.add(v)
        meta = dict(
            name=f"lj_{pN_m_eff}M",
            seed=SEED,
            prefix_edges=len(sub),
            nodes=len(nodes),
            edges=len(initial)+len(stream),
            initial_edges=len(initial),
            stream_edges=len(stream),
            batch_size=BATCH_SIZE,
            n_batches=n_batches,
            avg_degree=(2*(len(initial)+len(stream)))/max(1,len(nodes)),
            note=f"LiveJournal scalability prefix at {pN_m_eff}M edges (truncated stream of {n_batches} batches)",
        )
        with open(out_dir/"meta.json", "w") as g:
            json.dump(meta, g, indent=2)
        print(f"[lj_prefix:{pN_m_eff}M] initial={len(initial)} stream={len(stream)} n_batches={n_batches} -> {out_dir}", file=sys.stderr)

if __name__ == "__main__":
    main()
