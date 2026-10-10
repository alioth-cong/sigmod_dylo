#!/usr/bin/env python3
"""
Leiden-static reference (from-scratch each batch) via leidenalg + python-igraph.
Outputs per-batch community files in dynamo's native format:
  one community per line, tab-separated node IDs.

Usage:
  leiden_static.py <initial.txt> <inc-base/inc.txt> <numBatches> <outDir>

Also writes:
  metrics.tsv  batch<TAB>latency_ms<TAB>modularity<TAB>num_communities
"""
import sys
import time
from pathlib import Path

import igraph as ig
import leidenalg as la


def read_edges_uvw(path: Path):
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))
    return edges


def extend_filename(name: str, extend: str) -> str:
    i = name.rfind('.')
    return name[:i] + extend + '.' + name[i+1:]


def build_graph(edges):
    nodes = set()
    for u,v in edges:
        nodes.add(u); nodes.add(v)
    node_list = sorted(nodes)
    idx = {n:i for i,n in enumerate(node_list)}
    g = ig.Graph(n=len(node_list), edges=[(idx[u], idx[v]) for u,v in edges], directed=False)
    g.simplify(multiple=True, loops=True)
    return g, node_list


def write_communities(part, node_list, out_path: Path):
    """`part` is a leidenalg VertexPartition. node_list maps internal idx -> original id."""
    bag = {}
    for vid, cid in enumerate(part.membership):
        bag.setdefault(cid, []).append(node_list[vid])
    with open(out_path, "w") as g:
        for cid, members in bag.items():
            g.write("\t".join(str(x) for x in members) + "\n")


def main():
    if len(sys.argv) < 5:
        print("Usage: leiden_static.py <initial> <incBase> <numBatches> <outDir>", file=sys.stderr)
        sys.exit(2)
    initial = Path(sys.argv[1])
    inc_base = sys.argv[2]
    n_batches = int(sys.argv[3])
    out_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)

    cumulative = read_edges_uvw(initial)

    metrics = open(out_dir / "metrics.tsv", "w")
    metrics.write("batch\tlatency_ms\tmodularity\tnum_communities\n")
    metrics.flush()

    # batch 0
    t1 = time.time()
    g, node_list = build_graph(cumulative)
    part = la.find_partition(g, la.ModularityVertexPartition, seed=42)
    t2 = time.time()
    write_communities(part, node_list, out_dir / "com_0.txt")
    metrics.write(f"0\t{int((t2-t1)*1000)}\t{part.modularity}\t{len(set(part.membership))}\n")
    metrics.flush()

    for k in range(1, n_batches + 1):
        inc_path = extend_filename(inc_base, f"_{k}")
        cumulative.extend(read_edges_uvw(Path(inc_path)))
        t1 = time.time()
        g, node_list = build_graph(cumulative)
        part = la.find_partition(g, la.ModularityVertexPartition, seed=42)
        t2 = time.time()
        write_communities(part, node_list, out_dir / f"com_{k}.txt")
        metrics.write(f"{k}\t{int((t2-t1)*1000)}\t{part.modularity}\t{len(set(part.membership))}\n")
        metrics.flush()

    metrics.close()


if __name__ == "__main__":
    main()
