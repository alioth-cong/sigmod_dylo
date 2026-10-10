#!/usr/bin/env python3
"""
Static reference using CWTSLeiden/networkanalysis Java jar.

Modes:
  --algo louvain  : "SLM" reference (the older SLM is folded into networkanalysis as Louvain)
  --algo leiden   : Leiden reference

Per batch:
  - Rebuild cumulative edge list (initial + inc_1..inc_k)
  - Remap node IDs to dense 0..N-1
  - Call: java -cp networkanalysis.jar nl.cwts.networkanalysis.run.RunNetworkClustering \
            -a {Louvain|Leiden} -q Modularity --seed 42 -o cl.txt edges.tsv
  - Read cl.txt (node<TAB>cluster), invert ID map, write com_<k>.txt in dynamo native format.

Also writes:
  metrics.tsv  batch<TAB>latency_ms<TAB>modularity<TAB>num_communities

Usage:
  netanalysis_static.py --algo {louvain|leiden} <initial> <incBase> <numBatches> <outDir>
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(DYLO_ROOT + "")
JAR_GLOB = list((ROOT / "external" / "networkanalysis" / "build" / "libs").glob("networkanalysis-*.jar"))
if not JAR_GLOB:
    sys.exit("networkanalysis jar not found")
JAR = JAR_GLOB[0]
RUNCLASS = "nl.cwts.networkanalysis.run.RunNetworkClustering"

def read_edges(path: Path):
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = line.split("\t")
            edges.append((int(p[0]), int(p[1])))
    return edges

def extend_filename(name: str, extend: str) -> str:
    i = name.rfind('.')
    return name[:i] + extend + '.' + name[i+1:]

def remap_and_write(edges, tmp_dir: Path):
    nodes = set()
    for u,v in edges:
        nodes.add(u); nodes.add(v)
    node_list = sorted(nodes)
    fwd = {n:i for i,n in enumerate(node_list)}
    edges_dense_path = tmp_dir / "edges.tsv"
    seen = set()
    with open(edges_dense_path, "w") as g:
        for u,v in edges:
            if u == v: continue
            a, b = fwd[u], fwd[v]
            if a > b: a, b = b, a
            if (a,b) in seen: continue
            seen.add((a,b))
            g.write(f"{a}\t{b}\n")
    return node_list, edges_dense_path

def run_clustering(algo: str, edges_dense_path: Path, out_clust: Path, log: Path):
    java = os.environ.get("JAVA_HOME", os.path.expanduser("~/.local/opt/jdk-21.0.5+11")) + "/bin/java"
    algo_name = "Louvain" if algo == "louvain" else "Leiden"
    fastutil = os.path.expanduser("~/.m2/repository/it/unimi/dsi/fastutil/8.2.2/fastutil-8.2.2.jar")
    cp = f"{JAR}:{fastutil}"
    cmd = [java, "-Xmx128g", "-cp", cp, RUNCLASS,
           "-a", algo_name, "-q", "Modularity", "--seed", "42",
           "-o", str(out_clust), str(edges_dense_path)]
    with open(log, "a") as lg:
        lg.write(" ".join(cmd) + "\n")
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, check=True)

def read_clustering(out_clust: Path, node_list):
    """Map node<TAB>cluster lines back to original IDs grouped by cluster."""
    bag = {}
    with open(out_clust) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            p = line.split("\t")
            v = int(p[0]); c = int(p[1])
            bag.setdefault(c, []).append(node_list[v])
    return bag

def write_dynamo_community(bag, out_path: Path):
    with open(out_path, "w") as g:
        for c, members in bag.items():
            g.write("\t".join(str(x) for x in members) + "\n")

def modularity_of(edges, membership):
    """Compute unweighted modularity given edge list and {node: cluster} membership."""
    from collections import defaultdict
    deg = defaultdict(int)
    for u,v in edges:
        deg[u] += 1; deg[v] += 1
    m = len(edges)
    if m == 0: return 0.0
    # sum_in
    in_links = defaultdict(int)
    tot = defaultdict(int)
    for u,v in edges:
        cu = membership[u]; cv = membership[v]
        if cu == cv:
            in_links[cu] += 1
        tot[cu] += 1
        tot[cv] += 1
    q = 0.0
    twoM = 2.0 * m
    for c, ll in in_links.items():
        q += (2.0*ll)/twoM - (tot.get(c,0)/twoM)**2
    return q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["louvain","leiden"], required=True)
    ap.add_argument("initial")
    ap.add_argument("incBase")
    ap.add_argument("numBatches", type=int)
    ap.add_argument("outDir")
    args = ap.parse_args()

    out_dir = Path(args.outDir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"; tmp_dir.mkdir(exist_ok=True)
    log = out_dir / "netanalysis.log"

    cumulative = read_edges(Path(args.initial))
    metrics = open(out_dir/"metrics.tsv","w")
    metrics.write("batch\tlatency_ms\tmodularity\tnum_communities\n")

    def do_batch(k):
        node_list, edges_dense = remap_and_write(cumulative, tmp_dir)
        cl_path = tmp_dir / f"clust_{k}.tsv"
        t1 = time.time()
        run_clustering(args.algo, edges_dense, cl_path, log)
        t2 = time.time()
        bag = read_clustering(cl_path, node_list)
        write_dynamo_community(bag, out_dir/f"com_{k}.txt")
        # Build membership for Q
        membership = {}
        for c, members in bag.items():
            for x in members: membership[x] = c
        Q = modularity_of(cumulative, membership)
        metrics.write(f"{k}\t{int((t2-t1)*1000)}\t{Q}\t{len(bag)}\n")
        metrics.flush()

    do_batch(0)
    for k in range(1, args.numBatches+1):
        cumulative.extend(read_edges(Path(extend_filename(args.incBase, f"_{k}"))))
        do_batch(k)
    metrics.close()
    # cleanup tmp
    for p in tmp_dir.glob("*"):
        try: p.unlink()
        except: pass

if __name__ == "__main__":
    main()
