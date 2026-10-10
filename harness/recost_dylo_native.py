#!/usr/bin/env python3
"""Re-measure the re-summarization cost of DyLo's three instances with the **native number of batches**.

The problem: in tab:main, DyLo's Cost came from the exp5 lambda sweep, which ran a fixed
20 batches, while every baseline's Cost came from the dataset's native batch split (Facebook 9 batches,
GPlus 1224, Flickr 320, YouTube 299, ...). The normalized cost is "the share of the summary tree dirtied per batch";
larger batches dirty more per batch, so the two sets of numbers were not on the same scale, and the bias direction
even varied by dataset (on Facebook DyLo's batches were smaller and biased low; on GPlus larger and biased high).

This script uses exactly the driver and environment of exp5_lambda.py (DYLO_EMIT_HIER=1,
DYLO_LAMBDA=0.1), only replacing the batch count with n_batches from meta.json, so that DyLo and the baselines
are on the same scale.

Disk: hierarchical output is large (YouTube 1.13M vertices x 3 levels x 299 batches = 7 GB per instance),
so each (dataset, instance) is scored right after it runs and its directory deleted; the peak keeps only as many copies as running jobs.

Usage: recost_dylo_native.py [--workers N] [--datasets a,b,c] [--min-free-gb 80]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver"
RUNS = ROOT / "data" / "runs"
SCRATCH = Path(DYLO_ROOT + "/tmpruns/recost")
sys.path.insert(0, str(ROOT / "harness"))

LAMBDA = "1e-1"          # the paper's default operating point
INSTANCES = ["louvain", "slm", "leiden"]


def free_gb(p=DYLO_ROOT) -> float:
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9


def one(dataset: str, inst: str, min_free: float) -> dict:
    from dirty_cost import stream_cost
    from hier_cost import stream_hier_cost
    import json

    rec = dict(dataset=dataset, instance=inst, lam=LAMBDA)
    run_dir = RUNS / dataset
    meta = json.loads((run_dir / "meta.json").read_text())
    nb = meta["n_batches"]
    rec["batches"] = nb

    if free_gb() < min_free:
        return dict(rec, status=f"skip: only {free_gb():.0f}G disk left")

    out = SCRATCH / f"{dataset}__{inst}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DYLO_EMIT_HIER="1", DYLO_LAMBDA=LAMBDA)
    cmd = [str(DRIVER), inst, str(run_dir / "initial.txt"),
           str(run_dir / "inc.txt"), str(nb), str(out)]
    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=14400)
    except subprocess.TimeoutExpired:
        shutil.rmtree(out, ignore_errors=True)
        return dict(rec, status="TLE(4h)")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(out, ignore_errors=True)
        return dict(rec, status=f"exit={e.returncode}")
    rec["run_sec"] = round(time.time() - t0, 1)

    try:
        mrows = [l.rstrip("\n").split("\t") for l in open(out / "metrics.tsv")][1:]
        rec["Q"] = statistics.mean(float(r[2]) for r in mrows)
        rec["latency_ms"] = (statistics.mean(float(r[1]) for r in mrows[1:])
                             if len(mrows) > 1 else 0.0)
        fc = stream_cost(out, nb)
        rec["cost_flat"] = statistics.mean(r[2] for r in fc) if fc else 0.0
        hc = stream_hier_cost(out, nb)
        rec["cost_hier"] = statistics.mean(r[2] for r in hc) if hc else rec["cost_flat"]
        rec["n_used"] = len(hc) if hc else len(fc)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"cost-error: {type(e).__name__}: {str(e)[:70]}"
    finally:
        shutil.rmtree(out, ignore_errors=True)     # delete after scoring to bound peak disk usage
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default="")
    ap.add_argument("--min-free-gb", type=float, default=80.0)
    ap.add_argument("--out", default=str(ROOT / "results/dylo_native_cost.csv"))
    args = ap.parse_args()

    if args.datasets:
        dss = args.datasets.split(",")
    else:
        dss = [d.name for d in sorted(RUNS.iterdir())
               if (d / "meta.json").exists() and (d / "initial.txt").exists()]
    jobs = [(d, i) for d in dss for i in INSTANCES]
    print(f"{len(jobs)} jobs ({len(dss)} datasets x {len(INSTANCES)} instances), "
          f"{args.workers} in parallel, {free_gb():.0f}G disk free", flush=True)

    res = []
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, d, i, args.min_free_gb): (d, i) for d, i in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res.append(r)
            if r.get("status") == "ok":
                print(f"  [{n}/{len(jobs)}] {r['dataset']:<11} dylo-{r['instance']:<8} "
                      f"batches={r['batches']:<5} cost_hier={r['cost_hier']:.6f} "
                      f"Q={r['Q']:.5f} ({r['run_sec']}s) {free_gb():.0f}G left", flush=True)
            else:
                print(f"  [{n}/{len(jobs)}] {r['dataset']:<11} dylo-{r['instance']:<8} "
                      f"{r.get('status')}", flush=True)

    res.sort(key=lambda r: (r["dataset"], r["instance"]))
    cols = ["dataset", "instance", "lam", "batches", "n_used", "Q",
            "latency_ms", "cost_hier", "cost_flat", "run_sec", "status"]
    with open(args.out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nsucceeded {sum(1 for r in res if r.get('status')=='ok')}/{len(res)} -> {args.out}")


if __name__ == "__main__":
    main()
