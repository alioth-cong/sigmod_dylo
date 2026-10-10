#!/usr/bin/env python3
"""
Louvain from-scratch reference that ALSO emits its full multi-level dendrogram,
for the hierarchical re-summarization cost (P1). Uses python-igraph
community_multilevel(return_levels=True).

Usage:
  louvain_hier.py <initial.txt> <inc-base/inc.txt> <numBatches> <outDir>

Per batch k it writes:
  com_<k>.txt        the FINAL (coarsest) partition  -> flat cost, like other methods
  com_<k>_L0.txt     finest level (leaves of the summary tree)
  com_<k>_L1.txt ... coarser levels (L<top> == final partition)
  metrics.tsv        batch<TAB>latency_ms<TAB>modularity<TAB>num_communities
All community files: one community per line, tab-separated ORIGINAL vertex ids.
"""
import sys
import time
from pathlib import Path

import igraph as ig


def read_edges_uvw(path: Path):
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            edges.append((int(p[0]), int(p[1])))
    return edges


def extend_filename(name: str, extend: str) -> str:
    i = name.rfind('.')
    return name[:i] + extend + '.' + name[i+1:]


def build_graph(edges):
    nodes = set()
    for u, v in edges:
        nodes.add(u); nodes.add(v)
    node_list = sorted(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    g = ig.Graph(n=len(node_list), edges=[(idx[u], idx[v]) for u, v in edges], directed=False)
    g.simplify(multiple=True, loops=True)
    return g, node_list


def write_membership(membership, node_list, out_path: Path):
    bag = {}
    for vid, cid in enumerate(membership):
        bag.setdefault(cid, []).append(node_list[vid])
    with open(out_path, "w") as g:
        for cid, members in bag.items():
            g.write("\t".join(str(x) for x in members) + "\n")


def run_batch(cumulative, out_dir: Path, k: int, metrics):
    t1 = time.time()
    g, node_list = build_graph(cumulative)
    levels = g.community_multilevel(return_levels=True)   # [finest ... coarsest]
    t2 = time.time()
    # hierarchy level files: L0 = finest
    for i, cl in enumerate(levels):
        write_membership(cl.membership, node_list, out_dir / f"com_{k}_L{i}.txt")
    final = levels[-1]                                     # coarsest = final communities
    write_membership(final.membership, node_list, out_dir / f"com_{k}.txt")
    mod = g.modularity(final.membership)
    metrics.write(f"{k}\t{(t2-t1)*1000.0:.6f}\t{mod}\t{len(set(final.membership))}\n")
    metrics.flush()


def main():
    if len(sys.argv) < 5:
        print("Usage: louvain_hier.py <initial> <incBase> <numBatches> <outDir>", file=sys.stderr)
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

    run_batch(cumulative, out_dir, 0, metrics)
    for k in range(1, n_batches + 1):
        cumulative.extend(read_edges_uvw(Path(extend_filename(inc_base, f"_{k}"))))
        run_batch(cumulative, out_dir, k, metrics)
    metrics.close()


if __name__ == "__main__":
    main()
