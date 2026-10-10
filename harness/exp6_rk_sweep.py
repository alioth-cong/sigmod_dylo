#!/usr/bin/env python3
"""
R/K calibration sweep: the paper's audit interval K and rebuild threshold R are currently
fixed given values (K=8 per the default-parameters paragraph, R=rho=0.0015 as used in
exp6_recompute.py) with no sweep analogous to Exp-5's lambda knee-finding. exp6_recompute.py's
existing "drift-monitor" policy audits every batch (implicit K=1) and only varies rho -- it
never actually simulates the paper's described policy ("every K batches an audit measures the
gap; DyLo rebuilds once the accumulated gap reaches R"). This script implements that combined
audited-rebuild(K,R) policy and sweeps both knobs jointly across datasets, reusing the same
gap-curve machinery as exp6_recompute.py (gap[b] is computed once per dataset via a single
never-rebuild run; all K/R variants are then free pure-Python simulation over that fixed curve).

Output: results/rk_sweep.csv (long format: dataset,inst,K,rho,n_rebuilds,mean_gap,
total_cost,competitive_ratio) plus results/rk_sweep_summary.csv (per-dataset argmin K at the
paper's reference rho, and whether audited-rebuild ever beats the best fixed-K at that rho).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp6_recompute import incr_gap, maintenance, optimal_offline, ski_rental  # noqa: E402

RESULTS = Path(DYLO_ROOT + "/results")

DATASETS = ["facebook", "cit-hepth", "cit-hepph", "epinions", "dblp", "twitter"]
BATCHES = {"facebook": 9, "cit-hepth": 36, "cit-hepph": 43, "epinions": 41,
           "dblp": 105, "twitter": 135}
INST = "leiden"
REF_RHO = 0.0015          # the value already used as "R" throughout Exp-6 (recompute.csv)
REF_K = 8                 # the paper's declared default audit interval
K_GRID = [1, 2, 4, 8, 16, 32, 64]
RHO_GRID = [0.0005, 0.0015, 0.005, 0.015, 0.05]


def audited_rebuild(bs, gap, rho, K):
    """The policy the paper's prose actually describes: audit only every K-th batch (by
    position in the batch stream); at an audit, if accumulated carried drift since the last
    rebuild has reached rho, rebuild (reset carried drift to 0 as of that audited batch)."""
    reb = []
    gap_last = 0.0
    for i, b in enumerate(bs, start=1):
        if i % K != 0:
            continue
        carried = max(0.0, gap[b] - gap_last)
        if carried >= rho:
            reb.append(b)
            gap_last = gap[b]
    return reb


def row(dataset, inst, bs, gap, opt_cost, policy, K, rho, reb):
    m = maintenance(bs, gap, reb)
    tot = m + len(reb) * rho
    reb_s = sorted(reb)
    carried = []
    last = 0.0
    ri = 0
    for b in bs:
        while ri < len(reb_s) and reb_s[ri] <= b:
            last = gap.get(reb_s[ri], 0.0)
            ri += 1
        carried.append(max(0.0, gap[b] - last))
    return dict(dataset=dataset, inst=inst, policy=policy, K=K, rho=f"{rho:.4f}",
                n_rebuilds=len(reb), mean_gap=f"{statistics.mean(carried):.5f}",
                max_gap=f"{max(carried):.5f}", total_cost=f"{tot:.5f}",
                competitive_ratio=f"{tot/opt_cost:.3f}" if opt_cost > 0 else "1.000",
                batches=len(bs))


def main():
    all_rows = []
    summary_rows = []
    for ds in DATASETS:
        nB = BATCHES[ds]
        print(f"[rk_sweep] computing gap curve: {ds} ({INST}, {nB} batches)...", flush=True)
        bs, gap, Qi, Qr = incr_gap(ds, INST, nB)

        for rho in RHO_GRID:
            opt_cost, opt_sched = optimal_offline(bs, gap, rho)

            best_fixedK = None
            for K in K_GRID:
                reb = list(range(K, nB + 1, K))
                r = row(ds, INST, bs, gap, opt_cost, f"fixedK{K}", K, rho, reb)
                all_rows.append(r)
                cr = float(r["competitive_ratio"])
                if best_fixedK is None or cr < best_fixedK[1]:
                    best_fixedK = (K, cr)

                r2 = row(ds, INST, bs, gap, opt_cost, "audited-rebuild", K, rho,
                         audited_rebuild(bs, gap, rho, K))
                all_rows.append(r2)

            # K=1 "drift-monitor" reference (matches exp6_recompute.py's existing policy)
            r3 = row(ds, INST, bs, gap, opt_cost, "drift-monitor(K=1)", 1, rho,
                     ski_rental(bs, gap, rho))
            all_rows.append(r3)

            if abs(rho - REF_RHO) < 1e-9:
                audited_at_refK = [r for r in all_rows if r["dataset"] == ds
                                    and r["policy"] == "audited-rebuild"
                                    and r["K"] == REF_K and abs(float(r["rho"]) - rho) < 1e-9][0]
                best_audited = min((r for r in all_rows if r["dataset"] == ds
                                     and r["policy"] == "audited-rebuild"
                                     and abs(float(r["rho"]) - rho) < 1e-9),
                                    key=lambda r: float(r["competitive_ratio"]))
                summary_rows.append(dict(
                    dataset=ds, ref_rho=rho, ref_K=REF_K,
                    paper_default_cr=audited_at_refK["competitive_ratio"],
                    best_fixedK=best_fixedK[0], best_fixedK_cr=f"{best_fixedK[1]:.3f}",
                    best_audited_K=best_audited["K"], best_audited_cr=best_audited["competitive_ratio"],
                    drift_monitor_K1_cr=r3["competitive_ratio"],
                ))

    with open(RESULTS / "rk_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with open(RESULTS / "rk_sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nwrote {RESULTS/'rk_sweep.csv'} ({len(all_rows)} rows)")
    print(f"wrote {RESULTS/'rk_sweep_summary.csv'} ({len(summary_rows)} rows)\n")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    main()
