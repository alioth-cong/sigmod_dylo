#!/usr/bin/env python3
"""
Exp-6 full-precision export (writing-agent audit item 4).

Re-runs the exact recompute-policy simulation from exp6_recompute.py (same drift
curve: DyLo never-rebuild vs the committed <inst>-cpp-static reference; same
policies) but writes UNROUNDED values (repr precision) so paper ratios can be
computed without rounding artifacts. recompute.csv itself is left untouched.

Output: results/recompute_precise.csv with rows
  never / fixedK4 / fixedK8 / fixedK16 / drift-monitor / optimal-offline / always / gremod
per dataset, columns: n_rebuilds, mean_gap, max_gap, total_cost, competitive_ratio
(all full precision), plus the headline ratio gremod_gap/drift_gap per dataset.

Usage: exp6_precise.py [--datasets dblp,epinions,cit-hepph,twitter] [--rho 0.002]
"""
import argparse, csv, importlib.util, statistics, sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
RESULTS = HARNESS.parent / "results"
PER_BATCH = RESULTS / "per_batch"

spec = importlib.util.spec_from_file_location("exp6", HARNESS / "exp6_recompute.py")
exp6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp6)


def gremod_gap(ds, inst, nB):
    """GreMod gap vs the same from-scratch reference (as in exp6_recompute.py)."""
    gp = PER_BATCH / f"gremod__{ds}.tsv"
    ref = PER_BATCH / f"{inst}-cpp-static__{ds}.tsv"
    if not gp.exists() or not ref.exists():
        return None
    Qg = {int(r.split("\t")[0]): float(r.split("\t")[2]) for r in gp.read_text().splitlines()[1:] if r.strip()}
    Qr = {int(r.split("\t")[0]): float(r.split("\t")[2]) for r in ref.read_text().splitlines()[1:] if r.strip()}
    bs = sorted(b for b in Qg if b in Qr and 1 <= b <= nB)
    return [max(0.0, Qr[b] - Qg[b]) for b in bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dblp,epinions,cit-hepph,twitter")
    ap.add_argument("--inst", default="leiden")
    ap.add_argument("--rho", type=float, default=0.002)
    args = ap.parse_args()
    NB = {"dblp": 105, "epinions": 41, "cit-hepph": 43, "twitter": 135}

    hdr = ["policy", "dataset", "inst", "rho", "n_rebuilds", "mean_gap", "max_gap",
           "total_cost", "competitive_ratio", "batches", "gremod_over_drift_gap"]
    out = []
    for ds in args.datasets.split(","):
        nB = NB[ds]
        bs, gap, Qi, Qr = exp6.incr_gap(ds, args.inst, nB)
        rho = args.rho

        def cost_of(rebuilds):
            return exp6.maintenance(bs, gap, rebuilds) + len(rebuilds) * rho

        opt_cost, opt_sched = exp6.optimal_offline(bs, gap, rho)

        def carried(rebuilds):
            reb = sorted(rebuilds); res = []; gl = 0.0; ri = 0
            for b in bs:
                while ri < len(reb) and reb[ri] <= b:
                    gl = gap.get(reb[ri], 0.0); ri += 1
                res.append(max(0.0, gap[b] - gl))
            return res

        rows = {}
        policies = {
            "never": [],
            "fixedK4": [b for i, b in enumerate(bs) if (i + 1) % 4 == 0],
            "fixedK8": [b for i, b in enumerate(bs) if (i + 1) % 8 == 0],
            "fixedK16": [b for i, b in enumerate(bs) if (i + 1) % 16 == 0],
            "drift-monitor": exp6.ski_rental(bs, gap, rho),
            "optimal-offline": opt_sched,
        }
        for pol, reb in policies.items():
            car = carried(reb)
            tot = cost_of(reb)
            rows[pol] = dict(policy=pol, dataset=ds, inst=args.inst, rho=repr(rho),
                             n_rebuilds=len(reb), mean_gap=repr(statistics.mean(car)),
                             max_gap=repr(max(car)), total_cost=repr(tot),
                             competitive_ratio=repr(tot / opt_cost) if opt_cost > 0 else "",
                             batches=len(bs))
        # always: rebuild every batch -> zero carried gap, cost = nB*rho
        tot = len(bs) * rho
        rows["always"] = dict(policy="always", dataset=ds, inst=args.inst, rho=repr(rho),
                              n_rebuilds=len(bs), mean_gap=repr(0.0), max_gap=repr(0.0),
                              total_cost=repr(tot),
                              competitive_ratio=repr(tot / opt_cost) if opt_cost > 0 else "",
                              batches=len(bs))
        gg = gremod_gap(ds, args.inst, nB)
        ratio = ""
        if gg:
            dm_gap = statistics.mean(carried(policies["drift-monitor"]))
            ratio = repr(statistics.mean(gg) / dm_gap) if dm_gap > 0 else "inf"
            rows["gremod"] = dict(policy="gremod", dataset=ds, inst="-", rho=repr(args.rho),
                                  n_rebuilds=0, mean_gap=repr(statistics.mean(gg)),
                                  max_gap=repr(max(gg)), total_cost="", competitive_ratio="",
                                  batches=len(gg))
        for pol in ["never", "fixedK4", "fixedK8", "fixedK16", "drift-monitor",
                    "optimal-offline", "always", "gremod"]:
            if pol in rows:
                rows[pol]["gremod_over_drift_gap"] = ratio if pol == "drift-monitor" else ""
                out.append(rows[pol])
        dm = rows["drift-monitor"]
        print(f"[exp6-precise] {ds}: drift-monitor mean_gap={dm['mean_gap']} "
              f"n_reb={dm['n_rebuilds']} CR={dm['competitive_ratio']} "
              f"gremod/drift={ratio}", flush=True)

    with open(RESULTS / "recompute_precise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in hdr})
    print(f"[exp6-precise] wrote {RESULTS/'recompute_precise.csv'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
