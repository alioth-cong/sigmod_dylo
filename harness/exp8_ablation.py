#!/usr/bin/env python3
"""
Exp-8 (tab:ablation): disable each DyLo-Leiden component in turn (dblp). Each component is
shown necessary by the metric it governs (caption predictions):
  - cost-aware DP (lambda=0): inflates re-summarization cost
  - affected-region bound (re-optimize whole graph = from-scratch): inflates latency
  - batch processing (|dE|=1): inflates both per-edge latency and per-edge cost
  - drift monitor (never rebuild): lets modularity decay over the stream
Assembled from committed CSVs (lambda.csv, latency_compare.csv, resumm_hier.csv,
recompute.csv, batchsize.csv) + a from-scratch latency probe. dblp showcase (LiveJournal
hierarchical maintenance is out of budget). Output: results/ablation.csv.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv, os, statistics, subprocess
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
R = ROOT/"results"
DRIVER = ROOT/"harness/cpp/dylo/dylo_driver"
RUNS = ROOT/"data"/"runs"
DS = "dblp"


def lam(ds, l, col):
    for r in csv.DictReader(open(R/"lambda.csv")):
        if r["instance"] == "leiden" and r["dataset"] == ds and r["lambda"] == l:
            return float(r[col])
    return None


def main():
    # full DyLo-Leiden at default lambda=0.1
    full_Q = lam(DS, "1e-1", "Q"); full_cost = lam(DS, "1e-1", "cost_hier")
    lam0_cost = lam(DS, "0", "cost_hier")
    full_lat = next(float(r["dylo_leiden_ms"]) for r in csv.DictReader(open(R/"latency_compare.csv"))
                    if r["dataset"] == DS)
    # -region: from-scratch latency probe (re-optimize whole graph each batch)
    subprocess.run([str(DRIVER), "leiden-fs", str(RUNS/DS/"initial.txt"), str(RUNS/DS/"inc.txt"),
                    "4", "/tmp/ex8_fs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    fsm = [l.split("\t") for l in open("/tmp/ex8_fs/metrics.tsv")][1:]
    fs_lat = statistics.mean(float(r[1]) for r in fsm[1:]); fs_Q = statistics.mean(float(r[2]) for r in fsm)
    fs_cost = next(float(r["cost_norm_hier"]) for r in csv.DictReader(open(R/"resumm_hier.csv"))
                   if r["method"] == "leiden-cpp-static" and r["dataset"] == DS)
    # -batch: from Exp-7 (|dE|=1 vs default |dE|=1000), per-edge metrics
    bs = {int(r["batch_size"]): r for r in csv.DictReader(open(R/"batchsize.csv"))}
    b1, bk = bs[1], bs[1000]
    # -drift: from Exp-6 never-rebuild vs full (recompute.csv)
    rec = {r["policy"]: r for r in csv.DictReader(open(R/"recompute.csv"))}
    never_gap = float(rec["never"]["mean_gap"]); never_maxgap = float(rec["never"]["max_gap"])

    rows = [
        dict(variant="DyLo-Leiden (full)", latency_ms=f"{full_lat:.3f}", modularity_Q=f"{full_Q:.5f}",
             resumm_cost=f"{full_cost:.5f}", note="default: lambda=0.1, region-bound, batch=1000, drift-monitor"),
        dict(variant="- cost-aware DP (lambda=0)", latency_ms=f"{full_lat:.3f}", modularity_Q=f"{full_Q:.5f}",
             resumm_cost=f"{lam0_cost:.5f}",
             note=f"re-summ cost inflates {lam0_cost/full_cost:.1f}x (commits every dirty community)"),
        dict(variant="- affected-region bound", latency_ms=f"{fs_lat:.1f}", modularity_Q=f"{fs_Q:.5f}",
             resumm_cost=f"{fs_cost:.5f}",
             note=f"re-optimize whole graph (from-scratch): latency inflates {fs_lat/full_lat:.0f}x"),
        dict(variant="- batch processing (|dE|=1)", latency_ms=f"{float(b1['us_per_edge'])/1000*1000:.3f}",
             modularity_Q=b1["modularity_Q"], resumm_cost=f"{float(b1['resumm_cost_per_edge']):.1f}/edge",
             note=(f"edge-at-a-time: per-edge latency {float(b1['us_per_edge']):.2f} vs {float(bk['us_per_edge']):.2f} us "
                   f"({float(b1['us_per_edge'])/float(bk['us_per_edge']):.0f}x), per-edge cost "
                   f"{float(b1['resumm_cost_per_edge']):.0f} vs {float(bk['resumm_cost_per_edge']):.0f} "
                   f"({float(b1['resumm_cost_per_edge'])/float(bk['resumm_cost_per_edge']):.1f}x)")),
        dict(variant="- drift monitor (never rebuild)", latency_ms=f"{full_lat:.3f}",
             modularity_Q=f"{full_Q-never_maxgap:.5f}", resumm_cost=f"{full_cost:.5f}",
             note=f"no rebuild: modularity decays, drift gap grows to +{never_maxgap:.4f} over the stream"),
    ]
    with open(R/"ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","latency_ms","modularity_Q","resumm_cost","note"])
        w.writeheader()
        for r in rows: w.writerow(r)
    for r in rows:
        print(f"[exp8] {r['variant']:32s} lat={r['latency_ms']:>10s} Q={r['modularity_Q']} cost={r['resumm_cost']}")
    print("[exp8] wrote results/ablation.csv")


if __name__ == "__main__":
    main()
