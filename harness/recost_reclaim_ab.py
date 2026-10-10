#!/usr/bin/env python3
"""Singleton reclamation on/off A/B: full stream, native batch counts, only RECLAIM toggled.

Origin: the paper's Exp-1/Exp-4 used harness/cpp/dylo/dylo_driver, which has **no** singleton reclamation
at all (it only reads AUDIT_K / REBUILD_R / LAMBDA / SELECTOR / EMIT_HIER). Reclamation exists only in
dylo_driver_std4. A probe on Facebook shows the three differ widely:

  dylo_driver (paper)       cost=.1016   batch-8 peak .921
  std4 + reclamation        cost=.0685   batch-8 peak .609
  std4 without reclamation  cost=.0009   no peak

i.e. the paper's .1016 came almost entirely from one whole-tree re-partition by the old driver at an audit point; with std4 and reclamation off
that re-partition never happens, the cost drops by two orders of magnitude, and Q is unchanged (.8359 vs .8360).

This script runs the full stream with std4, the std configuration of run_dylo_driver.sh, and native batch counts,
toggling only DYLO_RECLAIM / DYLO_FAST_RECLAIM and recording cost, Q, last-batch community count, and reclamation count.

Note that DYLO_RECLAIM / DYLO_FAST_RECLAIM are tested as getenv(...)!=nullptr, so writing =0 actually
turns them on; for norecl both variables must be left unset.

Usage: recost_reclaim_ab.py [--workers N] [--datasets a,b,c] [--instances leiden,...]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver_std4"
RUNS = ROOT / "data" / "runs"
SCRATCH = ROOT / "tmpruns/reclaim_ab"
sys.path.insert(0, str(ROOT / "harness"))

# the std configuration of run_dylo_driver.sh, kept verbatim
COMMON = dict(
    DYLO_EMIT_HIER="1", DYLO_LAMBDA="1e-1", DYLO_AUDIT_K="8",
    DYLO_REBUILD_R="1e18", DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2",
    DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0", DYLO_HIER_GAMMA="0.1",
    DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_DEFER="0",
    DYLO_LAZY_CERT="1", DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1",
)


def free_gb(p=DYLO_ROOT) -> float:
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9


def one(dataset: str, inst: str, variant: str, min_free: float,
        max_batches: int = 0, gamma: str = "", defer: str = "") -> dict:
    from dirty_cost import stream_cost
    from hier_cost import stream_hier_cost

    rec = dict(dataset=dataset, instance=inst, variant=variant,
               gamma=gamma or COMMON["DYLO_HIER_GAMMA"],
               defer=defer or COMMON["DYLO_DEFER"])
    run_dir = RUNS / dataset
    nb = json.loads((run_dir / "meta.json").read_text())["n_batches"]
    # see the parameter of the same name in run.py: LiveJournal's native 4286 batches are infeasible, so it is cut to the first N batches.
    if max_batches and max_batches < nb:
        nb = max_batches
    rec["batches"] = nb
    if free_gb() < min_free:
        return dict(rec, status=f"skip: only {free_gb():.0f}G disk left")

    # directory names must include gamma/defer: when several configurations run in parallel, identically named directories overwrite each other,
    # showing up as random FileNotFoundError / IndexError (seen on flickr/leiden at the same time).
    out = SCRATCH / f"{dataset}__{inst}__{variant}__g{gamma or 'd'}__f{defer or 'd'}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **COMMON)
    if gamma:
        env["DYLO_HIER_GAMMA"] = gamma
    if defer:
        env["DYLO_DEFER"] = defer
    if variant == "recl":
        env["DYLO_RECLAIM"] = "1"
        env["DYLO_FAST_RECLAIM"] = "1"
    # norecl: leave both variables unset (presence turns them on, so writing =0 would backfire)

    cmd = [str(DRIVER), inst, str(run_dir / "initial.txt"),
           str(run_dir / "inc.txt"), str(nb), str(out)]
    t0 = time.time()
    try:
        with open(out / "err", "w") as ef:
            subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=ef,
                           check=True, timeout=14400)
    except subprocess.TimeoutExpired:
        shutil.rmtree(out, ignore_errors=True)
        return dict(rec, status="TLE(4h)")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(out, ignore_errors=True)
        return dict(rec, status=f"exit={e.returncode}")
    rec["run_sec"] = round(time.time() - t0, 1)

    try:
        err = (out / "err").read_text()
        rec["reclaims"] = next((int(t.split("=")[1]) for t in err.split()
                                if t.startswith("reclaims=")), "")
        m = [l.rstrip("\n").split("\t") for l in open(out / "metrics.tsv")][1:]
        rec["Q"] = statistics.mean(float(r[2]) for r in m)
        rec["Q_end"] = float(m[-1][2])
        rec["num_comms_end"] = m[-1][3]
        rec["latency_ms"] = (statistics.mean(float(r[1]) for r in m[1:])
                             if len(m) > 1 else 0.0)
        # the refresh plan the driver itself reports is the accounting in which deferral takes effect: hier_cost re-derives from partitions
        # and charges a source community as soon as its member set changes, so deferral is invisible to it (driver_std4.cxx:243 warns about this).
        rs = [l.rstrip("\n").split("\t") for l in open(out / "resum.tsv")][1:]
        rs = [x for x in rs if len(x) >= 6]
        rec["cost_resum"] = statistics.mean(float(x[5]) for x in rs) if rs else 0.0
        fc = stream_cost(out, nb)
        rec["cost_flat"] = statistics.mean(r[2] for r in fc) if fc else 0.0
        hc = stream_hier_cost(out, nb)
        rec["cost_hier"] = statistics.mean(r[2] for r in hc) if hc else rec["cost_flat"]
        rec["cost_max_batch"] = max((r[2] for r in hc), default=0.0)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"cost-error: {type(e).__name__}: {str(e)[:70]}"
    finally:
        shutil.rmtree(out, ignore_errors=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--datasets", default="facebook,cit-hepth,cit-hepph,epinions,dblp,twitter,flickr,youtube,gplus")
    ap.add_argument("--instances", default="louvain,slm,leiden")
    ap.add_argument("--min-free-gb", type=float, default=80.0)
    ap.add_argument("--out", default=str(ROOT / "results/reclaim_ab.csv"))
    ap.add_argument("--max-batches", type=int, default=0, help="0 = native batch count from meta")
    ap.add_argument("--gamma", default="", help="override DYLO_HIER_GAMMA; empty = COMMON's 0.1")
    ap.add_argument("--defer", default="", help="override DYLO_DEFER; empty = COMMON's 0 (eager)")
    ap.add_argument("--variants", default="norecl,recl")
    args = ap.parse_args()

    jobs = [(d, i, v) for d in args.datasets.split(",")
            for i in args.instances.split(",") for v in args.variants.split(",")]
    print(f"{len(jobs)} jobs, {args.workers} in parallel, {free_gb():.0f}G disk free", flush=True)

    res = []
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, d, i, v, args.min_free_gb, args.max_batches,
                          args.gamma, args.defer): (d, i, v) for d, i, v in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result()
            res.append(r)
            if r.get("status") == "ok":
                print(f"  [{n}/{len(jobs)}] {r['dataset']:<11} {r['instance']:<8} "
                      f"{r['variant']:<7} batches={r['batches']:<5} cost={r['cost_hier']:.6f} "
                      f"peak={r['cost_max_batch']:.3f} Q={r['Q']:.5f} "
                      f"communities={r['num_comms_end']} reclaims={r['reclaims']} "
                      f"({r['run_sec']}s)", flush=True)
            else:
                print(f"  [{n}/{len(jobs)}] {r['dataset']:<11} {r['instance']:<8} "
                      f"{r['variant']:<7} {r.get('status')}", flush=True)

    res.sort(key=lambda r: (r["dataset"], r["instance"], r["variant"]))
    cols = ["dataset", "instance", "variant", "gamma", "defer", "cost_resum", "batches", "reclaims", "Q", "Q_end",
            "num_comms_end", "latency_ms", "cost_hier", "cost_flat",
            "cost_max_batch", "run_sec", "status"]
    with open(args.out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=cols)
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nsucceeded {sum(1 for r in res if r.get('status')=='ok')}/{len(res)} -> {args.out}")


if __name__ == "__main__":
    main()
