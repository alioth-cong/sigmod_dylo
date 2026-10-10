#!/usr/bin/env python3
"""
Writing-agent audit item 3: DyLo per-batch latency at the DEFAULT lambda=0.1 in the
Exp-3/4/5 configuration (DYLO_EMIT_HIER=1, cost-aware hierarchical path, audit
interval K=8 with in-loop lazy rebuild -> the K=8 rebuild spikes are INCLUDED in the
per-batch mean, i.e. amortized audit). DYLO_NO_COM=1 suppresses com/level file
emission so the measurement is compute-bound (same convention as the full-LJ retime).

Strictly SERIAL (timing isolation). Appends one row per (instance,dataset) to
results/latency_lambda01.csv as each cell finishes, so partial progress is durable.

Usage: exp1_lambda01_latency.py [--datasets d1,d2,...] [--instances louvain,slm,leiden]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse, csv, json, os, statistics, subprocess
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver"
RUNS = ROOT / "data" / "runs"
OUT = ROOT / "results" / "latency_lambda01.csv"
HDR = ["instance", "dataset", "lambda", "audit_K", "mean_latency_ms", "median_latency_ms",
       "p95_latency_ms", "mean_latency_ms_excl_rebuild", "n_rebuild_batches", "Q_mean",
       "Q_last", "batches", "notes"]
NOTE = ("Exp-3/4/5 config: DYLO_EMIT_HIER=1 DYLO_LAMBDA=0.1, in-loop lazy rebuild every "
        "K=8 batches (amortized audit INCLUDED in mean); DYLO_NO_COM=1 (compute-bound, "
        "no com-file I/O); single-thread; serial isolated run")


def run_cell(inst, ds):
    nB = json.loads((RUNS / ds / "meta.json").read_text())["n_batches"]
    od = f"/tmp/ex1_lam01_{inst}_{ds}"
    subprocess.run([str(DRIVER), inst, str(RUNS / ds / "initial.txt"),
                    str(RUNS / ds / "inc.txt"), str(nB), od],
                   env=dict(os.environ, DYLO_EMIT_HIER="1", DYLO_LAMBDA="0.1",
                            DYLO_NO_COM="1"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    rows = [l.split("\t") for l in open(Path(od) / "metrics.tsv").read().splitlines()[1:]]
    b = [(int(r[0]), float(r[1]), float(r[2])) for r in rows if int(r[0]) >= 1]
    lat = [x[1] for x in b]
    qs = [x[2] for x in b]
    lat_s = sorted(lat)
    # rebuild batches are every K=8th processed batch (in-loop lazy rebuild)
    reb = [x[1] for i, x in enumerate(b) if (i + 1) % 8 == 0]
    non = [x[1] for i, x in enumerate(b) if (i + 1) % 8 != 0]
    return dict(instance=inst, dataset=ds, **{"lambda": "0.1"}, audit_K=8,
                mean_latency_ms=f"{statistics.mean(lat):.3f}",
                median_latency_ms=f"{statistics.median(lat):.3f}",
                p95_latency_ms=f"{lat_s[int(0.95 * (len(lat_s) - 1))]:.3f}",
                mean_latency_ms_excl_rebuild=f"{statistics.mean(non):.3f}" if non else "",
                n_rebuild_batches=len(reb),
                Q_mean=f"{statistics.mean(qs):.6f}", Q_last=f"{qs[-1]:.6f}",
                batches=len(b), notes=NOTE)


def append_row(row):
    exists = OUT.exists()
    have = []
    if exists:
        have = list(csv.DictReader(open(OUT)))
        have = [r for r in have if not (r["instance"] == row["instance"] and r["dataset"] == row["dataset"])]
    have.append(row)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HDR)
        w.writeheader()
        for r in have:
            w.writerow({k: r.get(k, "") for k in HDR})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="facebook,cit-hepth,cit-hepph,epinions,dblp,twitter,youtube,flickr,gplus")
    ap.add_argument("--instances", default="louvain,slm,leiden")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        for inst in args.instances.split(","):
            row = run_cell(inst, ds)
            append_row(row)
            print(f"[lam01] {inst:8s} {ds:12s} mean={row['mean_latency_ms']}ms "
                  f"(excl-rebuild {row['mean_latency_ms_excl_rebuild']}ms) "
                  f"Q={row['Q_mean']} nB={row['batches']}", flush=True)


if __name__ == "__main__":
    main()
