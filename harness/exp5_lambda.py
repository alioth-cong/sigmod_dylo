#!/usr/bin/env python3
"""
Exp-5 lambda-sweep runner. For each (instance, dataset, lambda) run the cost-aware
hierarchical DyLo driver (DYLO_EMIT_HIER=1, DYLO_LAMBDA=lambda) and record:
  Q, cost_hier, cost_flat, dirty_frac, latency_ms, num_comms
averaged over the streamed batches. Appends rows to results/lambda.csv.

This traces DyLo's quality-cost frontier. lambda=0 is the cost-oblivious anchor
(== HIT-Leiden: hierarchical incremental Leiden, full affected-region re-optimisation);
lambda=inf freezes (never re-summarise). The GreMod reference point (flat, add-only) is
appended separately so the frontier figure can overlay it as a dominated dot.

Usage:
  exp5_lambda.py --instances leiden --datasets dblp --lambdas 0,1e-3,1e-2,1e-1,0.3,1,3,10,inf --batches 20
NOTE: this is the cost/Q experiment; its latency_ms column is the hier-mode per-batch time
(DP + dendrogram rebuild), NOT DyLo's production latency (see results/latency.csv, run solo).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse, csv, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver"
RUNS = ROOT / "data" / "runs"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "harness"))
from dirty_cost import stream_cost          # noqa
from hier_cost import stream_hier_cost       # noqa

HEADER = ["instance", "dataset", "lambda", "Q", "cost_hier", "cost_flat",
          "dirty_frac", "latency_ms", "num_comms", "batches"]


def run_one(inst, dataset, lam, batches, outdir):
    run_dir = RUNS / dataset
    env = dict(os.environ, DYLO_EMIT_HIER="1", DYLO_LAMBDA=str(lam))
    cmd = [str(DRIVER), inst, str(run_dir / "initial.txt"), str(run_dir / "inc.txt"),
           str(batches), str(outdir)]
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    od = Path(outdir)
    mrows = [l.rstrip("\n").split("\t") for l in open(od / "metrics.tsv")][1:]
    # skip batch 0 (no update) for latency mean
    Q = statistics.mean(float(r[2]) for r in mrows)
    lat = statistics.mean(float(r[1]) for r in mrows[1:]) if len(mrows) > 1 else 0.0
    ncomm = float(mrows[-1][3])
    fc = stream_cost(od, batches)
    cost_flat = statistics.mean(r[2] for r in fc) if fc else 0.0
    try:
        hc = stream_hier_cost(od, batches)
        cost_hier = statistics.mean(r[2] for r in hc) if hc else cost_flat
    except Exception:
        cost_hier = cost_flat
    dp = od / "dp.tsv"
    if dp.exists():
        drows = [l.rstrip("\n").split("\t") for l in open(dp)][1:]
        dirty_frac = statistics.mean(float(r[3]) for r in drows) if drows else 0.0
    else:
        dirty_frac = float("nan")
    return dict(instance=inst, dataset=dataset, lam=lam, Q=Q, cost_hier=cost_hier,
                cost_flat=cost_flat, dirty_frac=dirty_frac, latency_ms=lat,
                num_comms=ncomm, batches=len(mrows) - 1)


def append_rows(rows, out_csv):
    out_csv = Path(out_csv)
    existing = []
    if out_csv.exists():
        existing = list(csv.DictReader(open(out_csv)))
    # replace any (instance,dataset,lambda) we just recomputed
    key = lambda r: (r["instance"], r["dataset"], str(r["lambda"]))
    newkeys = {(r["instance"], r["dataset"], str(r["lam"])) for r in rows}
    existing = [e for e in existing if key(e) not in newkeys]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for e in existing:
            w.writerow({k: e.get(k, "") for k in HEADER})
        for r in rows:
            w.writerow({"instance": r["instance"], "dataset": r["dataset"], "lambda": r["lam"],
                        "Q": f"{r['Q']:.5f}", "cost_hier": f"{r['cost_hier']:.5f}",
                        "cost_flat": f"{r['cost_flat']:.5f}", "dirty_frac": f"{r['dirty_frac']:.5f}",
                        "latency_ms": f"{r['latency_ms']:.3f}", "num_comms": int(r["num_comms"]),
                        "batches": r["batches"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="leiden")
    ap.add_argument("--datasets", default="dblp")
    ap.add_argument("--lambdas", default="0,1e-3,1e-2,1e-1,0.3,1,3,10,inf")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--out", default=str(RESULTS / "lambda.csv"))
    args = ap.parse_args()
    insts = args.instances.split(",")
    dsets = args.datasets.split(",")
    lams = args.lambdas.split(",")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for inst in insts:
            for ds in dsets:
                for lam in lams:
                    t0 = time.time()
                    od = Path(td) / f"{inst}_{ds}_{lam}"
                    od.mkdir(parents=True, exist_ok=True)
                    try:
                        r = run_one(inst, ds, lam, args.batches, od)
                        rows.append(r)
                        print(f"[exp5] {inst:7s} {ds:10s} lam={lam:5s} "
                              f"Q={r['Q']:.4f} cost_hier={r['cost_hier']:.4f} "
                              f"cost_flat={r['cost_flat']:.4f} dirty={r['dirty_frac']:.3f} "
                              f"lat={r['latency_ms']:.1f}ms ({time.time()-t0:.0f}s)", flush=True)
                    except Exception as e:
                        print(f"[exp5] FAILED {inst} {ds} lam={lam}: {e}", flush=True)
    append_rows(rows, args.out)
    print(f"[exp5] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
