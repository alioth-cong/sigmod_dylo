#!/usr/bin/env python3
"""
Exp-6 recompute-policy / ski-rental simulation on the REAL per-batch drift curve.

Drift gap(b) = Q_fromscratch(b) - Q_incremental(b) over a stream (DyLo never-rebuild vs the
committed from-scratch reference). A rebuild at batch r re-anchors the incremental partition
to from-scratch, so the carried drift since the last rebuild r is
   carried(b | r) = max(0, gap(b) - gap(r)).
A policy picks rebuild batches S; its cost = maintenance (area under carried drift) + |S|*rho,
where rho is the rebuild cost (one full re-summarisation), expressed in drift-equivalent units.

Policies: never, fixedK (K in given list), drift-monitor (ski-rental: rebuild when carried
drift >= rho -> 2-competitive, thm:competitive), optimal-offline (exact O(n^2) DP). Reports
#rebuilds, mean carried gap, total cost, and competitive ratio = total / optimal_total.
GreMod (add-only) included as a reference whose gap grows and is never re-optimised.

Output: results/recompute.csv. Usage:
  exp6_recompute.py --dataset dblp --inst leiden --batches 105 --rho 0.0015
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse, csv, os, statistics, subprocess
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver"
RUNS = ROOT / "data" / "runs"
RESULTS = ROOT / "results"


def incr_gap(ds, inst, nB):
    """Run DyLo never-rebuild, return gap[b] vs committed from-scratch reference."""
    od = f"/tmp/ex6_never_{ds}"
    subprocess.run([str(DRIVER), inst, str(RUNS/ds/"initial.txt"), str(RUNS/ds/"inc.txt"),
                    str(nB), od], env=dict(os.environ, DYLO_REBUILD_R="1e18"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    Qi = {int(r.split("\t")[0]): float(r.split("\t")[2]) for r in
          open(Path(od)/"metrics.tsv").read().splitlines()[1:]}
    ref = ROOT/"results/per_batch"/f"{inst}-cpp-static__{ds}.tsv"
    Qr = {int(r.split("\t")[0]): float(r.split("\t")[2]) for r in
          open(ref).read().splitlines()[1:]}
    bs = sorted(b for b in Qi if b in Qr and 1 <= b <= nB)
    return bs, {b: max(0.0, Qr[b] - Qi[b]) for b in bs}, Qi, Qr


def maintenance(bs, gap, rebuilds):
    """Sum of carried drift over batches given a set of rebuild batches."""
    reb = sorted(rebuilds)
    total = 0.0
    last = 0; gap_last = 0.0
    ri = 0
    for b in bs:
        while ri < len(reb) and reb[ri] <= b:
            last = reb[ri]; gap_last = gap.get(last, 0.0); ri += 1
        total += max(0.0, gap[b] - gap_last)
    return total


def optimal_offline(bs, gap, rho):
    """Exact DP: min over rebuild schedules of maintenance + n*rho. Returns (cost, schedule)."""
    n = len(bs)
    # idx by position; allow rebuild at any batch position. dp[i] = min cost for prefix ending
    # with a rebuild at bs[i] (carried drift since bs[i] is 0 at bs[i]).
    INF = float("inf")
    # cost of NOT rebuilding anywhere = maintenance with no rebuild
    best_cost = maintenance(bs, gap, [])  # never
    best_sched = []
    # dp over "last rebuild position"; carried since rebuild r over (r, end]
    # dp[i]: min total cost of handling bs[0..i] with a rebuild at bs[i]
    dp = [INF]*n
    pre = [-1]*n
    for i in range(n):
        # segment from start (no prior rebuild) to bs[i], paying rho for the rebuild at bs[i]
        seg0 = sum(max(0.0, gap[bs[j]] - 0.0) for j in range(i))  # carried before first rebuild
        dp[i] = seg0 + rho
        pre[i] = -1
        for k in range(i):
            seg = sum(max(0.0, gap[bs[j]] - gap[bs[k]]) for j in range(k, i))
            if dp[k] + seg + rho < dp[i]:
                dp[i] = dp[k] + seg + rho; pre[i] = k
    # add tail (from last rebuild to end) for each candidate last rebuild
    for i in range(n):
        tail = sum(max(0.0, gap[bs[j]] - gap[bs[i]]) for j in range(i, n))
        tot = dp[i] + tail
        if tot < best_cost:
            best_cost = tot
            sched = []; k = i
            while k != -1:
                sched.append(bs[k]); k = pre[k]
            best_sched = sorted(sched)
    return best_cost, best_sched


def ski_rental(bs, gap, rho):
    """Online ski-rental: accumulate maintenance (carried drift) since the last rebuild and
    rebuild once it reaches the rebuild cost rho ('rent until rental == buy, then buy').
    This is the classic deterministic 2-competitive rule (thm:competitive)."""
    reb = []; gap_last = 0.0; acc = 0.0
    for b in bs:
        acc += max(0.0, gap[b] - gap_last)
        if acc >= rho:
            reb.append(b); gap_last = gap[b]; acc = 0.0
    return reb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dblp")
    ap.add_argument("--inst", default="leiden")
    ap.add_argument("--batches", type=int, default=105)
    ap.add_argument("--rho", type=float, default=0.0015, help="rebuild cost in drift-equiv units")
    ap.add_argument("--out", default=str(RESULTS/"recompute.csv"))
    args = ap.parse_args()
    bs, gap, Qi, Qr = incr_gap(args.dataset, args.inst, args.batches)
    rho = args.rho
    opt_cost, opt_sched = optimal_offline(bs, gap, rho)

    def row(policy, reb):
        m = maintenance(bs, gap, reb)
        tot = m + len(reb)*rho
        carried = []  # per-batch carried drift for mean
        reb_s = sorted(reb); last = 0.0; ri = 0
        for b in bs:
            while ri < len(reb_s) and reb_s[ri] <= b:
                last = gap.get(reb_s[ri], 0.0); ri += 1
            carried.append(max(0.0, gap[b]-last))
        return dict(policy=policy, dataset=args.dataset, inst=args.inst, rho=f"{rho:.4f}",
                    n_rebuilds=len(reb), mean_gap=f"{statistics.mean(carried):.5f}",
                    max_gap=f"{max(carried):.5f}", total_cost=f"{tot:.5f}",
                    competitive_ratio=f"{tot/opt_cost:.3f}" if opt_cost > 0 else "1.000",
                    batches=len(bs))

    rows = []
    rows.append(row("never", []))
    for K in (4, 8, 16):
        rows.append({**row(f"fixedK{K}", list(range(K, args.batches+1, K))),
                     "policy": f"fixedK{K}"})
    rows.append(row("drift-monitor", ski_rental(bs, gap, rho)))
    rows.append({**row("optimal-offline", opt_sched), "competitive_ratio": "1.000"})
    # always-from-scratch: gap 0, n_rebuilds = nB, cost = nB*rho
    rows.append(dict(policy="always", dataset=args.dataset, inst=args.inst, rho=f"{rho:.4f}",
                     n_rebuilds=len(bs), mean_gap="0.00000", max_gap="0.00000",
                     total_cost=f"{len(bs)*rho:.5f}",
                     competitive_ratio=f"{len(bs)*rho/opt_cost:.3f}" if opt_cost>0 else "",
                     batches=len(bs)))
    # gremod reference (committed per_batch): gap vs from-scratch, never re-optimised
    gm = ROOT/"results/per_batch"/f"gremod__{args.dataset}.tsv"
    if gm.exists():
        Qg = {int(r.split("\t")[0]): float(r.split("\t")[2]) for r in
              open(gm).read().splitlines()[1:]}
        gg = [max(0.0, Qr[b]-Qg[b]) for b in bs if b in Qg and b in Qr]
        if gg:
            rows.append(dict(policy="gremod", dataset=args.dataset, inst="-", rho=f"{rho:.4f}",
                             n_rebuilds=0, mean_gap=f"{statistics.mean(gg):.5f}",
                             max_gap=f"{max(gg):.5f}", total_cost="", competitive_ratio="",
                             batches=len(gg)))

    hdr = ["policy","dataset","inst","rho","n_rebuilds","mean_gap","max_gap","total_cost",
           "competitive_ratio","batches"]
    out = Path(args.out)
    existing = [r for r in csv.DictReader(open(out))] if out.exists() else []
    existing = [r for r in existing if r["dataset"] != args.dataset]   # merge: replace this dataset
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in existing: w.writerow({k: r.get(k, "") for k in hdr})
        for r in rows: w.writerow({k: r.get(k, "") for k in hdr})
    for r in rows:
        print(f"[exp6] {r['policy']:16s} reb={r['n_rebuilds']:>3} mean_gap={r['mean_gap']} "
              f"total={r.get('total_cost','')} CR={r.get('competitive_ratio','')}")
    print(f"[exp6] optimal schedule rebuilds at {opt_sched}; wrote {args.out}")


if __name__ == "__main__":
    main()
