#!/usr/bin/env python3
"""Layer 4: under the same std4 main-table configuration, decide with old vs B1 weights and compare decisions, cost, and Q.

Variants (all in the main-table configuration: leiden/louvain/slm x recl, DEFER=0, lambda default 1e-1):
  old     dylo_driver_std4 (current main table)
  b1      dylo_driver_std4w, DYLO_WMODEL=b1 (S automatic: initial total tree weight equal to the old model's)
  b1park  same + DYLO_WB1_PARK=1 (parking a new vertex is priced as one single-member summary, consistently in selection/gate/peel)
Each run keeps its per-batch partitions (for decision comparison) and is scored with hier_cost_w under both the old and the b1 weights.

Usage: layer4_run.py --datasets a,b --instances leiden --variants old,b1,b1park
                    [--lambdas 1e-1] [--workers 24] --out results/layer4.csv
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
D = ROOT / "harness/cpp/dylo"
RUNS = ROOT / "data/runs"
SCR = ROOT / "tmpruns/layer4"
sys.path.insert(0, str(ROOT / "harness"))

COMMON = dict(
    DYLO_EMIT_HIER="1", DYLO_AUDIT_K="8", DYLO_REBUILD_R="1e18",
    DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2", DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0",
    DYLO_HIER_GAMMA="0.1", DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_DEFER="0",
    DYLO_LAZY_CERT="1", DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1",
    DYLO_RECLAIM="1", DYLO_FAST_RECLAIM="1",
)
VARIANTS = {
    "old": ("dylo_driver_std4", {}),
    "b1": ("dylo_driver_std4w", {"DYLO_WMODEL": "b1"}),
    "b1park": ("dylo_driver_std4w", {"DYLO_WMODEL": "b1", "DYLO_WB1_PARK": "1"}),
}


def run_dir(ds, inst, var, lam):
    return SCR / f"{ds}__{inst}__{var}__lam{lam}"


def one(ds, inst, var, lam):
    from hier_cost_w import stream, summarize
    nb = json.loads((RUNS / ds / "meta.json").read_text())["n_batches"]
    out = run_dir(ds, inst, var, lam)
    out.mkdir(parents=True, exist_ok=True)
    binary, extra = VARIANTS[var]
    env = dict(os.environ, **COMMON, **extra, DYLO_LAMBDA=lam)
    rec = dict(dataset=ds, instance=inst, variant=var, lam=lam, batches=nb)
    t0 = time.time()
    try:
        with open(out / "err", "w") as ef:
            subprocess.run([str(D / binary), inst, str(RUNS / ds / "initial.txt"),
                            str(RUNS / ds / "inc.txt"), str(nb), str(out)],
                           env=env, stdout=subprocess.DEVNULL, stderr=ef, check=True,
                           timeout=6 * 3600)
    except Exception as e:  # noqa: BLE001
        return dict(rec, status=f"run-fail: {type(e).__name__}")
    rec["run_sec"] = round(time.time() - t0, 1)
    err = (out / "err").read_text()
    m = re.search(r"S=([0-9.eE+-]+)", err)
    rec["S"] = float(m.group(1)) if m else ""
    for key in ("reclaims", "pruned", "vertices_reclaimed"):
        m = re.search(rf"\b{key}=(\d+)", err)
        rec[key] = int(m.group(1)) if m else ""
    mt = [l.rstrip("\n").split("\t") for l in open(out / "metrics.tsv")][1:]
    rec["Q"] = statistics.mean(float(r[2]) for r in mt)
    rec["Q_end"] = float(mt[-1][2])
    rec["comms_end"] = int(mt[-1][3])
    rec["latency_ms"] = statistics.mean(float(r[1]) for r in mt[1:]) if len(mt) > 1 else 0.0
    # number of singleton leaf communities in the last batch
    f = out / f"com_{nb}_L0.txt"
    if f.exists():
        sizes = [len(l.split()) for l in open(f) if l.strip()]
        rec["leaves_end"] = len(sizes)
        rec["singletons_end"] = sum(1 for s in sizes if s == 1)
    rs = [l.rstrip("\n").split("\t") for l in open(out / "resum.tsv")][1:]
    rs = [x for x in rs if len(x) >= 6]
    rec["cost_resum_own"] = statistics.mean(float(x[5]) for x in rs) if rs else 0.0
    rec.update({f"sc_{k}": v for k, v in summarize(stream(out, nb, True)).items()})
    rec["status"] = "ok"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="facebook,cit-hepth,cit-hepph,epinions,twitter,dblp")
    ap.add_argument("--instances", default="leiden")
    ap.add_argument("--variants", default="old,b1,b1park")
    ap.add_argument("--lambdas", default="1e-1")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(d, i, v, l) for d in a.datasets.split(",") for i in a.instances.split(",")
            for v in a.variants.split(",") for l in a.lambdas.split(",")]
    print(f"{len(jobs)} jobs, {a.workers} in parallel", flush=True)
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, *j): j for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res.append(r)
            if r.get("status") == "ok":
                print(f"  [{n}/{len(jobs)}] {r['dataset']:<10} {r['instance']:<7} {r['variant']:<7} "
                      f"λ={r['lam']:<5} Q={r['Q']:.5f} old={r['sc_cost_old']:.4f} "
                      f"b1={r['sc_cost_b1']:.4f} calls={r['sc_calls']:.1f} "
                      f"singletons={r.get('singletons_end')} ({r['run_sec']}s)", flush=True)
            else:
                print(f"  [{n}/{len(jobs)}] {futs[f]} {r.get('status')}", flush=True)
    cols = sorted({k for r in res for k in r}, key=lambda k: (k not in ("dataset", "instance", "variant", "lam"), k))
    with open(a.out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for r in sorted(res, key=lambda r: (r["dataset"], r["instance"], r["variant"], float(r["lam"]))):
            w.writerow(r)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
