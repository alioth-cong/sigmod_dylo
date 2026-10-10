#!/usr/bin/env python3
"""
Emit the `hit-leiden` baseline rows from the Exp-5 lambda=0 leiden points (lambda.csv).
hit-leiden == hierarchical, cost-oblivious, affected-region incremental Leiden ==
DyLo-Leiden at lambda=0 (a faithful approximation of HIT-Leiden, arXiv:2601.08554; differs
only in that it is DyLo's own Leiden detector run cost-obliviously). It anchors the Pareto
figure: DyLo's lambda-sweep sweeps down-left from this shared point.

Writes hit-leiden rows into quality.csv, resumm.csv, resumm_hier.csv, latency.csv,
replacing any existing hit-leiden rows. Values come from the lambda=0, instance=leiden rows.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
RESULTS = ROOT / "results"
REPO = "this-artifact (harness/cpp/dylo)"
NOTE = ("faithful HIT-Leiden approx (arXiv:2601.08554): hierarchical cost-oblivious "
        "incremental Leiden = DyLo-Leiden at lambda=0 (affected-region + full re-optimization, "
        "no lambda*Cost term); anchors DyLo's lambda-frontier")


def load_lambda0():
    d = {}
    for r in csv.DictReader(open(RESULTS / "lambda.csv")):
        if r["instance"] == "leiden" and r["lambda"] == "0":
            d[r["dataset"]] = r
    return d


def upsert(path, keycols, newrows, header=None):
    path = Path(path)
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    hdr = header or (list(rows[0].keys()) if rows else list(newrows[0].keys()))
    newkeys = {tuple(r[c] for c in keycols) for r in newrows}
    rows = [r for r in rows if tuple(r.get(c) for c in keycols) not in newkeys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in rows + newrows:
            w.writerow({k: r.get(k, "") for k in hdr})
    print(f"[hit-leiden] {path.name}: +{len(newrows)} rows")


def main():
    L0 = load_lambda0()
    q, res, hier, lat = [], [], [], []
    for ds, r in sorted(L0.items()):
        q.append(dict(method="hit-leiden", dataset=ds, value=r["Q"], unit="modularity",
                      language="C++", repo=REPO, commit="0", notes=NOTE))
        res.append(dict(method="hit-leiden", dataset=ds, cost_norm_weight=r["cost_flat"],
                        cost_norm_fromscratch="", cost_norm_fromscratch_leiden="", cost_abs="",
                        fromscratch_ref="leiden-static", language="C++", repo=REPO, commit="0", notes=NOTE))
        hier.append(dict(method="hit-leiden", dataset=ds, cost_norm_hier=r["cost_hier"],
                         cost_abs_hier="", batches=r["batches"], language="C++", repo=REPO,
                         commit="0", notes=NOTE))
        lat.append(dict(method="hit-leiden", dataset=ds, value=r["latency_ms"], unit="ms/batch",
                        language="C++", repo=REPO, commit="0",
                        notes=NOTE + "; hierarchical-mode per-batch time (DP + dendrogram), single-thread"))
    upsert(RESULTS / "quality.csv", ["method", "dataset"], q)
    upsert(RESULTS / "resumm.csv", ["method", "dataset"], res)
    upsert(RESULTS / "resumm_hier.csv", ["method", "dataset"], hier)
    upsert(RESULTS / "latency.csv", ["method", "dataset"], lat)


if __name__ == "__main__":
    main()
