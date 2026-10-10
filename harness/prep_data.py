#!/usr/bin/env python3
"""
Prepare a dataset for the uniform batched-stream harness.

Input:  data/edges/<name>.tsv   (one undirected edge per line, u<TAB>v, u<v)
Output: data/runs/<name>/
          meta.json
          initial.txt           90% of edges as u<TAB>v<TAB>1.0
          all_edges.txt         100% of edges (in stream order, useful for static refs)
          inc_1.txt..inc_K.txt  10% of edges in 1000-edge batches, format u<TAB>v<TAB><batch_idx>
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import json
import os
import random
import sys
from pathlib import Path

SEED = 42
BATCH_SIZE = 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--edges-file", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--initial-frac", type=float, default=0.90)
    args = ap.parse_args()

    root = Path(DYLO_ROOT + "")
    edges_file = Path(args.edges_file or root / "data" / "edges" / f"{args.name}.tsv")
    out_dir = Path(args.out_root or root / "data" / "runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_raw = []
    with open(edges_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            u, v = line.split("\t")[:2]
            edges_raw.append((int(u), int(v)))
    nodes_raw = set()
    for u,v in edges_raw:
        nodes_raw.add(u); nodes_raw.add(v)
    # Remap to dense 0..N-1 so downstream tools that assume int32 (Java/C++) work
    # on graphs like GPlus whose node IDs are 21-digit Google+ user IDs.
    node_list = sorted(nodes_raw)
    fwd = {n:i for i,n in enumerate(node_list)}
    edges = [(fwd[u], fwd[v]) for u,v in edges_raw]
    nodes = set(range(len(node_list)))
    print(f"[prep:{args.name}] loaded n={len(nodes_raw)} m={len(edges)} (remapped to dense 0..{len(node_list)-1})")

    rng = random.Random(SEED)
    rng.shuffle(edges)

    cut = int(len(edges) * args.initial_frac)
    initial = edges[:cut]
    stream  = edges[cut:]
    n_batches = (len(stream) + args.batch_size - 1) // args.batch_size
    print(f"[prep:{args.name}] initial={len(initial)} stream={len(stream)} batches={n_batches} (size={args.batch_size})")

    # initial.txt — u<TAB>v<TAB>1.0
    with open(out_dir/"initial.txt", "w") as g:
        for u,v in initial:
            g.write(f"{u}\t{v}\t1.0\n")
    # all_edges.txt — same edges in stream order (initial + stream)
    with open(out_dir/"all_edges.txt", "w") as g:
        for u,v in initial + stream:
            g.write(f"{u}\t{v}\t1.0\n")
    # inc_<K>.txt — u<TAB>v<TAB><K>
    for k in range(n_batches):
        chunk = stream[k*args.batch_size:(k+1)*args.batch_size]
        with open(out_dir/f"inc_{k+1}.txt", "w") as g:
            for u,v in chunk:
                g.write(f"{u}\t{v}\t{k+1}\n")

    meta = dict(
        name=args.name,
        seed=SEED,
        nodes=len(nodes),
        edges=len(edges),
        initial_edges=len(initial),
        stream_edges=len(stream),
        batch_size=args.batch_size,
        n_batches=n_batches,
        avg_degree=(2*len(edges))/max(1,len(nodes)),
    )
    with open(out_dir/"meta.json", "w") as g:
        json.dump(meta, g, indent=2)
    print(f"[prep:{args.name}] OK n_batches={n_batches} -> {out_dir}")

if __name__ == "__main__":
    main()
