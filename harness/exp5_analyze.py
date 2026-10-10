#!/usr/bin/env python3
"""
Exp-5 analysis: from results/lambda.csv (DyLo frontier) + committed GreMod/HIT-Leiden
reference points, compute the Pareto "money cells":
  - per dataset, the lambda at which DyLo's (hierarchical) cost <= GreMod's cost, and
    DyLo's Q there vs GreMod's Q  (target: DyLo Q strictly higher at matched cost);
  - the Pareto-dominance verdict per dataset and overall (k/9);
  - the knee lambda (max drop in cost per unit Q lost) -> proposed default lambda.

GreMod is flat (add-only) so it earns NO shared-ancestor savings: its hierarchical
re-summarization cost is >= its flat cost (every dirty leaf forces re-summarising its
ancestors). We therefore use GreMod_hier := GreMod_flat as a *lower bound* on its
hierarchical cost — a conservative (pro-GreMod) choice that still shows dominance.

Usage: exp5_analyze.py [--lambda-csv results/lambda.csv] [--instance leiden]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse, csv
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
RESULTS = ROOT / "results"


def load_gremod():
    q, cost = {}, {}
    for r in csv.DictReader(open(RESULTS / "quality.csv")):
        if r["method"] == "gremod" and r["value"]:
            q[r["dataset"]] = float(r["value"])
    for r in csv.DictReader(open(RESULTS / "resumm.csv")):
        if r["method"] == "gremod" and r["cost_norm_weight"]:
            cost[r["dataset"]] = float(r["cost_norm_weight"])
    return q, cost


def best_baseline_Q():
    """Per dataset, the best Q among ALL baselines (for the 'DyLo >= best baseline' check)."""
    bestQ = {}
    for r in csv.DictReader(open(RESULTS / "quality.csv")):
        ds, m = r["dataset"], r["method"]
        try:
            v = float(r["value"])
        except ValueError:
            continue
        if m.startswith("dylo") or m == "hit-leiden":
            continue
        bestQ[ds] = max(bestQ.get(ds, -1), v)
    return bestQ


def fnum(x):
    return float("inf") if x in ("inf", "Inf") else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-csv", default=str(RESULTS / "lambda.csv"))
    ap.add_argument("--instance", default="leiden")
    ap.add_argument("--axis", default="cost_hier", choices=["cost_hier", "cost_flat"])
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.lambda_csv)))
    gq, gcost = load_gremod()
    bbq = best_baseline_Q()

    by_ds = {}
    for r in rows:
        if r["instance"] != args.instance:
            continue
        by_ds.setdefault(r["dataset"], []).append(r)

    print(f"=== Exp-5 Pareto money cells (instance={args.instance}, axis={args.axis}) ===")
    print(f"GreMod_hier := GreMod_flat (lower bound; flat method earns no ancestor savings)\n")
    hdr = f"{'dataset':10s} {'gremod_Q':>8s} {'gremod_C':>8s} | {'lam*':>6s} {'DyLo_Q@C':>9s} {'DyLo_C':>8s} {'dQ':>7s} | {'domin':>5s} {'>=best':>6s}"
    print(hdr)
    print("-" * len(hdr))
    ndom = 0
    nds = 0
    knee_votes = {}
    for ds, frr in sorted(by_ds.items()):
        if ds not in gcost:
            continue
        nds += 1
        fr = sorted(frr, key=lambda r: fnum(r["lambda"]))
        gQ, gC = gq.get(ds, float("nan")), gcost[ds]
        # money cell: smallest-lambda point whose cost <= GreMod cost (matched or cheaper)
        cell = None
        for r in fr:
            if float(r[args.axis]) <= gC:
                cell = r
                break
        # pareto dominance: exists a frontier point with cost <= gC and Q > gQ
        dom = any(float(r[args.axis]) <= gC and float(r["Q"]) > gQ for r in fr)
        ndom += dom
        # knee: max |dCost/dQ| ... pick lambda with largest cost drop while Q within 0.003 of max
        Qmax = max(float(r["Q"]) for r in fr)
        cand = [r for r in fr if float(r["Q"]) >= Qmax - 0.003]
        knee = min(cand, key=lambda r: float(r[args.axis])) if cand else fr[0]
        knee_votes[ds] = knee["lambda"]
        ge = ">=" if (ds in bbq and Qmax >= bbq[ds] - 1e-6) else "<"
        if cell:
            print(f"{ds:10s} {gQ:8.4f} {gC:8.4f} | {cell['lambda']:>6s} {float(cell['Q']):9.4f} "
                  f"{float(cell[args.axis]):8.4f} {float(cell['Q'])-gQ:+7.4f} | {'YES' if dom else 'no':>5s} {ge:>6s}")
        else:
            print(f"{ds:10s} {gQ:8.4f} {gC:8.4f} | {'none':>6s} {'':9s} {'':8s} {'':7s} | {'no':>5s} {ge:>6s}")
    print("-" * len(hdr))
    print(f"Pareto-dominates GreMod on {ndom}/{nds} datasets (axis={args.axis}).")
    print(f"Knee lambda per dataset: {knee_votes}")
    # modal knee as proposed default
    from collections import Counter
    if knee_votes:
        common = Counter(knee_votes.values()).most_common(1)[0]
        print(f"Proposed default lambda (modal knee): {common[0]}  ({common[1]}/{len(knee_votes)} datasets)")


if __name__ == "__main__":
    main()
