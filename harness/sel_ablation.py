#!/usr/bin/env python3
"""Selector ablation: exact tree DP vs per-destination greedy vs per-move independent pricing (the naive scheme of Challenge 2).

Configuration = the paper's uniform setting: std4 + singleton reclamation, DEFER=1, eps=0.05 (single anchor -> tree-DP path), leiden.
Driver dylo_driver_selector (DYLO_SEL_MODE=dp|greedy|indep; dp is bit-identical to std4).
lambda=0 is run once (the three selectors are equivalent at lambda=0) as the modularity-only upper reference.
DYLO_NO_COM=1: per-batch partitions are not written (cost comes from the driver's own refresh plan resum.tsv, the correct accounting with deferral).

Each run records:
  Q, Q_end                 metrics.tsv
  cost                     per-batch mean of resum.tsv cost_norm (all refreshes, including reclamation and settlement)
  sel_cost                 per-batch mean of refreshes committed by selection (sum of accepted rounds' trueCost / that batch's tree weight)
  rounds, moves            accepted rounds / committed moves
  sumDF, sumGain           sum of realized dF (= measured gain - lambda * true cost, in driver units) / sum of measured gains
Results are appended to --out one by one (finished runs can be inspected while long ones such as GPlus continue).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
BIN = ROOT / "harness/cpp/dylo/dylo_driver_selector"
RUNS = Path(os.environ.get("SEL_RUNS", str(ROOT / "data/runs")))
SCR = ROOT / "tmpruns/selabl"
COMMON = dict(
    DYLO_EMIT_HIER="1", DYLO_AUDIT_K="8", DYLO_REBUILD_R="1e18", DYLO_SIGMA_ONLY="1",
    DYLO_SIGMA="0.2", DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0", DYLO_HIER_GAMMA="0.1",
    DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_DEFER="1", DYLO_DEFER_EPS="0.05",
    DYLO_LAZY_CERT="1", DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1",
    DYLO_RECLAIM="1", DYLO_FAST_RECLAIM="1", DYLO_NO_COM="1",
)
COLS = ["dataset", "mode", "lam", "status", "batches", "Q", "Q_end", "cost", "sel_cost",
        "rounds", "moves", "peels", "sumDF", "sumGain", "run_sec"]


def one(ds, mode, lam, inst):
    nb = json.loads((RUNS / ds / "meta.json").read_text())["n_batches"]
    out = SCR / f"{ds}__{inst}__{mode}__lam{lam}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **COMMON, DYLO_SEL_MODE=mode, DYLO_LAMBDA=lam)
    rec = dict(dataset=ds, mode=mode, lam=lam, batches=nb)
    t0 = time.time()
    try:
        with open(out / "err", "w") as ef:
            subprocess.run([str(BIN), inst, str(RUNS / ds / "initial.txt"),
                            str(RUNS / ds / "inc.txt"), str(nb), str(out)],
                           env=env, stdout=subprocess.DEVNULL, stderr=ef, check=True,
                           timeout=24 * 3600)
    except Exception as e:  # noqa: BLE001
        return dict(rec, status=f"fail:{type(e).__name__}", run_sec=round(time.time() - t0))
    rec["run_sec"] = round(time.time() - t0, 1)
    mt = [l.rstrip("\n").split("\t") for l in open(out / "metrics.tsv")][1:]
    rec["Q"] = statistics.mean(float(r[2]) for r in mt)
    rec["Q_end"] = float(mt[-1][2])
    rs = [l.rstrip("\n").split("\t") for l in open(out / "resum.tsv")][1:]
    rs = [x for x in rs if len(x) >= 6]
    rec["cost"] = statistics.mean(float(x[5]) for x in rs) if rs else 0.0
    tw = {int(x[0]): float(x[4]) for x in rs}
    sel = [l.rstrip("\n").split("\t") for l in open(out / "selection.tsv")][1:]
    acc = [x for x in sel if len(x) > 11 and x[2] != "PEEL" and x[11] == "1"]
    peel = [x for x in sel if len(x) > 8 and x[2] == "PEEL"]
    rec["rounds"] = len(acc)
    rec["moves"] = sum(int(x[3]) for x in acc) + len(peel)
    rec["peels"] = len(peel)
    rec["sumDF"] = sum(float(x[10]) for x in acc) + sum(float(x[8]) for x in peel)
    rec["sumGain"] = sum(float(x[7]) for x in acc)
    per_batch = {}
    for x in acc:
        per_batch[int(x[0])] = per_batch.get(int(x[0]), 0.0) + float(x[9])
    rec["sel_cost"] = (sum(c / tw[k] for k, c in per_batch.items() if tw.get(k)) / len(rs)) if rs else 0.0
    rec["status"] = "ok"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="facebook,cit-hepth,epinions,dblp,youtube,gplus")
    ap.add_argument("--lambdas", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    ap.add_argument("--modes", default="dp,greedy,indep")
    ap.add_argument("--instance", default="leiden")
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(d, m, l) for d in a.datasets.split(",") for m in a.modes.split(",")
            for l in a.lambdas.split(",")]
    jobs += [(d, "dp", "0") for d in a.datasets.split(",")]
    # long jobs first, so they do not trail serially at the end
    order = {"gplus": 0, "youtube": 1}
    jobs.sort(key=lambda j: (order.get(j[0], 9), float(j[2])))
    new = not Path(a.out).exists()
    with open(a.out, "a", newline="") as g:
        w = csv.DictWriter(g, fieldnames=COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        print(f"{len(jobs)} jobs, {a.workers} in parallel", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(one, d, m, l, a.instance): (d, m, l) for d, m, l in jobs}
            for n, f in enumerate(as_completed(futs), 1):
                r = f.result()
                w.writerow(r)
                g.flush()
                print(f"[{n}/{len(jobs)}] {r['dataset']:<10} {r['mode']:<7} λ={r['lam']:<5} "
                      f"{r.get('status')} Q={r.get('Q', 0):.5f} cost={r.get('cost', 0):.5f} "
                      f"moves={r.get('moves', '')} ΣΔF={r.get('sumDF', 0):.1f} ({r.get('run_sec')}s)",
                      flush=True)


if __name__ == "__main__":
    main()
