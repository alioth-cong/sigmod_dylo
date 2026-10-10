#!/usr/bin/env python3
"""
Generalizes exp7_batchsize.py (which only ever ran on DBLP, per its own docstring's
un-tested assertion that "the per-edge amortization trend is dataset-independent") to
several more datasets spanning scale, to actually check that claim rather than assert it.
Same methodology as the original: capped stream (initial + STREAM_EDGES suffix) so |dE|=1
is tractable, same SIZES sweep, same DyLo-Leiden incremental path.

Output: results/batchsize_multi.csv (does NOT touch results/batchsize.csv).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv, statistics, subprocess, sys, json
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
DATASETS = ["facebook", "cit-hepth", "epinions", "twitter"]


def make_capped(base):
    ae = (RUNS/base/"all_edges.txt").read_text().splitlines()
    meta = json.loads((RUNS/base/"meta.json").read_text())
    n_init = meta["initial_edges"]
    cap = ae[:n_init + STREAM_EDGES]
    out = Path(f"/tmp/{base}_cap_edges.tsv")
    out.write_text("\n".join("\t".join(l.split("\t")[:2]) for l in cap) + "\n")
    return out, n_init


def run_dataset(base):
    edges, n_init = make_capped(base)
    frac = n_init/(n_init+STREAM_EDGES)
    rows = []
    for dE in SIZES:
        name = f"{base}_bs{dE}"
        subprocess.run([sys.executable, str(PREP), "--name", name, "--edges-file", str(edges),
                        "--batch-size", str(dE), "--initial-frac", f"{frac:.6f}",
                        "--out-root", "/tmp/runs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        rd = Path("/tmp/runs")/name
        nB = min(len(list(rd.glob("inc_*.txt"))), 600)
        od = f"/tmp/ex7_{base}_{dE}"
        subprocess.run([str(DRIVER), "leiden", str(rd/"initial.txt"), str(rd/"inc.txt"),
                        str(nB), od], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        m = [l.split("\t") for l in open(Path(od)/"metrics.tsv")][1:]
        lat = statistics.mean(float(r[1]) for r in m[1:]) if len(m) > 1 else 0.0
        Q = statistics.mean(float(r[2]) for r in m)
        fc = stream_cost(Path(od), nB)
        tot_cost = sum(r[1] for r in fc)
        edges_done = dE*nB
        rows.append(dict(dataset=base, dE=dE, n_batches=nB, latency_ms=lat,
                         us_per_edge=lat*1000.0/dE, cost_per_edge=tot_cost/edges_done, Q=Q))
        print(f"[exp7-multi] {base:10s} |dE|={dE:6d} nB={nB:5d} lat={lat:.3f}ms "
              f"us/edge={lat*1000/dE:.2f} cost/edge={tot_cost/edges_done:.3f} Q={Q:.4f}", flush=True)
    return rows


def main():
    all_rows = []
    for base in DATASETS:
        all_rows.extend(run_dataset(base))

    with open(RESULTS/"batchsize_multi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "batch_size", "n_batches", "latency_ms_per_batch", "us_per_edge",
                    "resumm_cost_per_edge", "modularity_Q", "note"])
        for r in all_rows:
            w.writerow([r["dataset"], r["dE"], r["n_batches"], f"{r['latency_ms']:.3f}",
                        f"{r['us_per_edge']:.3f}", f"{r['cost_per_edge']:.4f}", f"{r['Q']:.5f}",
                        "DyLo-Leiden incremental; cost = flat re-summ weight (lambda=0 structural)"])
    print(f"[exp7-multi] wrote results/batchsize_multi.csv ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
