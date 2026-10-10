#!/usr/bin/env python3
"""
Exp-7 (tab:batch): effect of batch size |dE| on DyLo-Leiden. Larger batches amortize
re-optimization and re-summarization over more updates. We prep the same stream at each
|dE| and run DyLo-Leiden's incremental path, reporting per-batch latency, amortized per-edge
latency (us), re-summarization cost per edge, and modularity Q.

Feasible by capping the stream (full initial graph + a fixed STREAM_EDGES suffix) so |dE|=1
is tractable. Dataset: dblp (LiveJournal hierarchical maintenance is out of budget; the
per-edge amortization trend is dataset-independent). Output: results/batchsize.csv.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv, os, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT/"harness/cpp/dylo/dylo_driver"
PREP = ROOT/"harness/prep_data.py"
RUNS = ROOT/"data"/"runs"
RESULTS = ROOT/"results"
sys.path.insert(0, str(ROOT/"harness"))
from dirty_cost import stream_cost  # noqa

STREAM_EDGES = 10000
SIZES = [1, 10, 100, 1000, 10000]


def make_capped(base="dblp"):
    """edges file = full initial + STREAM_EDGES stream edges, from <base>/all_edges.txt."""
    ae = (RUNS/base/"all_edges.txt").read_text().splitlines()
    meta = __import__("json").loads((RUNS/base/"meta.json").read_text())
    n_init = meta["initial_edges"]
    cap = ae[:n_init + STREAM_EDGES]
    out = Path("/tmp/dblp_cap_edges.tsv")
    out.write_text("\n".join("\t".join(l.split("\t")[:2]) for l in cap) + "\n")
    return out, n_init


def main():
    edges, n_init = make_capped("dblp")
    frac = n_init/(n_init+STREAM_EDGES)
    rows = []
    for dE in SIZES:
        name = f"dblp_bs{dE}"
        subprocess.run([sys.executable, str(PREP), "--name", name, "--edges-file", str(edges),
                        "--batch-size", str(dE), "--initial-frac", f"{frac:.6f}",
                        "--out-root", "/tmp/runs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        rd = Path("/tmp/runs")/name
        nB = min(len(list(rd.glob("inc_*.txt"))), 600)   # cap batches; per-edge metrics stay representative
        od = f"/tmp/ex7_{dE}"
        subprocess.run([str(DRIVER), "leiden", str(rd/"initial.txt"), str(rd/"inc.txt"),
                        str(nB), od], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        m = [l.split("\t") for l in open(Path(od)/"metrics.tsv")][1:]
        lat = statistics.mean(float(r[1]) for r in m[1:]) if len(m) > 1 else 0.0
        Q = statistics.mean(float(r[2]) for r in m)
        fc = stream_cost(Path(od), nB)
        tot_cost = sum(r[1] for r in fc)          # cost_abs summed over batches
        edges_done = dE*nB
        rows.append(dict(dE=dE, n_batches=nB, latency_ms=lat,
                         us_per_edge=lat*1000.0/dE, cost_per_edge=tot_cost/edges_done, Q=Q))
        print(f"[exp7] |dE|={dE:6d} nB={nB:5d} lat={lat:.3f}ms us/edge={lat*1000/dE:.2f} "
              f"cost/edge={tot_cost/edges_done:.3f} Q={Q:.4f}", flush=True)

    with open(RESULTS/"batchsize.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "n_batches", "latency_ms_per_batch", "us_per_edge",
                    "resumm_cost_per_edge", "modularity_Q", "dataset", "note"])
        for r in rows:
            w.writerow([r["dE"], r["n_batches"], f"{r['latency_ms']:.3f}", f"{r['us_per_edge']:.3f}",
                        f"{r['cost_per_edge']:.4f}", f"{r['Q']:.5f}", "dblp(capped)",
                        "DyLo-Leiden incremental; cost = flat re-summ weight (lambda=0 structural)"])
    print(f"[exp7] wrote results/batchsize.csv")


if __name__ == "__main__":
    main()
