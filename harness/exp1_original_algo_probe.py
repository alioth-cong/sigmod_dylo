#!/usr/bin/env python3
"""
Diagnostic (not in original package): does the lambda=0 -> lambda=0.1 latency INCREASE
(instead of the decrease the paper's Corollary predicts) come from the new dual-anchor /
shared-ancestor-billing / closure-based selection code, or was it already present in the
original single-anchor algorithm? Runs the SAME controlled hier=1 probe (lambda=0 and
lambda=0.1) against the backed-up original single-anchor binary
(cpp/dylo/backup_single_anchor/dylo_driver), same 4 datasets as the production probe.

Output: results/latency_original_algo_probe.csv (does not touch production results).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv, json, os, statistics, subprocess
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/backup_single_anchor/dylo_driver"
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "results" / "latency_original_algo_probe.csv"
HDR = ["instance", "dataset", "lambda", "mean_latency_ms", "median_latency_ms",
       "mean_latency_ms_excl_rebuild", "n_rebuild_batches", "Q_mean", "batches", "notes"]
DATASETS = ["facebook", "cit-hepth", "cit-hepph", "epinions"]
INSTANCES = ["louvain", "slm", "leiden"]


def run_cell(inst, ds, lam):
    nB = json.loads((RUNS / ds / "meta.json").read_text())["n_batches"]
    od = f"/tmp/ex1_orig_{inst}_{ds}_{lam}"
    subprocess.run([str(DRIVER), inst, str(RUNS / ds / "initial.txt"),
                    str(RUNS / ds / "inc.txt"), str(nB), od],
                   env=dict(os.environ, DYLO_EMIT_HIER="1", DYLO_LAMBDA=str(lam),
                            DYLO_NO_COM="1"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    rows = [l.split("\t") for l in open(Path(od) / "metrics.tsv").read().splitlines()[1:]]
    b = [(int(r[0]), float(r[1]), float(r[2])) for r in rows if int(r[0]) >= 1]
    lat = [x[1] for x in b]
    qs = [x[2] for x in b]
    reb = [x[1] for i, x in enumerate(b) if (i + 1) % 8 == 0]
    non = [x[1] for i, x in enumerate(b) if (i + 1) % 8 != 0]
    return dict(instance=inst, dataset=ds, **{"lambda": str(lam)},
                mean_latency_ms=f"{statistics.mean(lat):.3f}",
                median_latency_ms=f"{statistics.median(lat):.3f}",
                mean_latency_ms_excl_rebuild=f"{statistics.mean(non):.3f}" if non else "",
                n_rebuild_batches=len(reb),
                Q_mean=f"{statistics.mean(qs):.6f}",
                batches=len(b),
                notes="ORIGINAL single-anchor binary (backup_single_anchor/dylo_driver), "
                      "DYLO_EMIT_HIER=1, K=8 default, DYLO_NO_COM=1")


def main():
    rows = []
    for ds in DATASETS:
        for inst in INSTANCES:
            for lam in ("0", "0.1"):
                r = run_cell(inst, ds, lam)
                rows.append(r)
                print(f"[orig-algo] {inst:8s} {ds:12s} lambda={lam:4s} "
                      f"mean={r['mean_latency_ms']}ms excl-reb={r['mean_latency_ms_excl_rebuild']}ms "
                      f"Q={r['Q_mean']}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HDR)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
