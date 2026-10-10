#!/usr/bin/env python3
"""
slm-static: GENUINE Smart Local Moving (Waltman & van Eck 2013), from-scratch each
batch, via the SLM4J "Modularity Optimizer" (org.slm4j.ModularityOptimizer) with
optimization_algorithm=3 (SLM). Replaces the previous networkanalysis Louvain-mode
stand-in (P4).

Usage:
  slm_static.py <initial.txt> <inc-base/inc.txt> <numBatches> <outDir>

Per batch: build the cumulative graph (dense-remapped), run SLM4J SLM (modularity
function=standard, resolution=1.0, 1 random start, 10 iterations, seed=42), write the
flat partition (original ids) to com_<k>.txt, and record latency (subprocess wall
clock, like the prior jar-based slm-static), modularity (igraph), and #communities.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import igraph as ig

ROOT = Path(__file__).resolve().parent.parent
SLM4J_BUILD = ROOT / "external" / "slm4j" / "build"
JAVA = Path(os.environ.get("JAVA_HOME", os.path.expanduser("~/.local/opt/jdk-21.0.5+11"))) / "bin" / "java"


def read_edges(path: Path):
    e = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split("\t")
            e.append((int(p[0]), int(p[1])))
    return e


def extend_filename(name: str, ext: str) -> str:
    i = name.rfind('.')
    return name[:i] + ext + '.' + name[i + 1:]


def run_batch(cum, out_dir: Path, k: int, metrics):
    nodes = sorted({n for e in cum for n in e})
    idx = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in cum], directed=False)
    g.simplify(multiple=True, loops=True)
    # SLM4J input: tab-delimited, 0-indexed, lower index first, sorted
    el = sorted((min(u, v), max(u, v)) for u, v in g.get_edgelist())
    ef = out_dir / f"_slm_in_{k}.txt"
    of = out_dir / f"_slm_out_{k}.txt"
    with open(ef, "w") as f:
        for a, b in el:
            f.write(f"{a}\t{b}\n")
    t = time.time()
    # input output modfunc resolution algo(3=SLM) n_starts n_iters seed print
    subprocess.run([str(JAVA), "-Xmx16g", "-cp", str(SLM4J_BUILD),
                    "org.slm4j.ModularityOptimizer", str(ef), str(of),
                    "1", "1.0", "3", "1", "10", "42", "0"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = (time.time() - t) * 1000.0
    memb = [int(x) for x in open(of) if x.strip()]
    # SLM4J emits one community id per node in index order; pad isolated tail nodes
    while len(memb) < len(nodes):
        memb.append(max(memb) + 1 if memb else 0)
    bag = {}
    for vid, cid in enumerate(memb):
        bag.setdefault(cid, []).append(nodes[vid])
    with open(out_dir / f"com_{k}.txt", "w") as gf:
        for cid, mem in bag.items():
            gf.write("\t".join(str(x) for x in mem) + "\n")
    mod = g.modularity(memb)
    metrics.write(f"{k}\t{dt:.6f}\t{mod}\t{len(set(memb))}\n")
    metrics.flush()
    ef.unlink(missing_ok=True)
    of.unlink(missing_ok=True)


def main():
    if len(sys.argv) < 5:
        print("Usage: slm_static.py <initial> <incBase> <numBatches> <outDir>", file=sys.stderr)
        sys.exit(2)
    initial = Path(sys.argv[1]); inc_base = sys.argv[2]
    n_batches = int(sys.argv[3]); out_dir = Path(sys.argv[4])
    out_dir.mkdir(parents=True, exist_ok=True)
    cum = read_edges(initial)
    metrics = open(out_dir / "metrics.tsv", "w")
    metrics.write("batch\tlatency_ms\tmodularity\tnum_communities\n")
    metrics.flush()
    run_batch(cum, out_dir, 0, metrics)
    for k in range(1, n_batches + 1):
        cum.extend(read_edges(Path(extend_filename(inc_base, f"_{k}"))))
        run_batch(cum, out_dir, k, metrics)
    metrics.close()


if __name__ == "__main__":
    main()
