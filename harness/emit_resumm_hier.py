#!/usr/bin/env python3
"""
Populate results/resumm_hier.csv (tab:resumm-hier) across methods:
  - DyLo (3 instances): real hierarchical cost at default lambda=0.1 (from lambda.csv).
  - hit-leiden / leiden-hier / louvain-hier: kept (real multi-level dendrograms).
  - Flat incremental + static baselines (dynamo, batch, gremod, qca, df-louvain, ds-louvain,
    df-leiden, *-static, *-cpp-static): no hierarchy -> their hierarchical re-summarization
    cost is the flat leaf cost as a LOWER BOUND (a flat partition earns no shared-ancestor
    amortization; mapped onto a tree it would only cost MORE). Taken from resumm.csv.
This lets tab:resumm-hier compare every method on the hierarchical axis where DyLo's
shared-ancestor savings show. Idempotent.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
from pathlib import Path

R = Path(DYLO_ROOT + "/results")
KEEP = {"leiden-hier", "louvain-hier", "hit-leiden"}      # real hierarchies already present
FLAT = {"dynamo", "batch", "gremod", "qca", "df-louvain", "ds-louvain", "df-leiden",
        "louvain-static", "leiden-static", "slm-static",
        "louvain-cpp-static", "leiden-cpp-static", "slm-cpp-static"}
FLAT_NOTE = ("flat method, no hierarchy; hierarchical re-summarization cost = flat leaf cost "
             "(lower bound: a flat partition earns no shared-ancestor amortization)")
HDR = ["method", "dataset", "cost_norm_hier", "cost_abs_hier", "batches", "language", "repo", "commit", "notes"]


def main():
    # meta (language/repo/commit) from resumm.csv
    meta = {}
    flat_cost = {}
    for r in csv.DictReader(open(R/"resumm.csv")):
        meta[r["method"]] = (r["language"], r["repo"], r["commit"])
        flat_cost[(r["method"], r["dataset"])] = r["cost_norm_weight"]

    rows = [r for r in csv.DictReader(open(R/"resumm_hier.csv")) if r["method"] in KEEP]

    # DyLo at default lambda=0.1
    lam = {}
    for r in csv.DictReader(open(R/"lambda.csv")):
        if r["lambda"] == "1e-1":
            lam[(f"dylo-{r['instance']}", r["dataset"])] = r
    for (m, ds), r in sorted(lam.items()):
        la, repo, com = meta.get(m, ("C++", "this-artifact (harness/cpp/dylo)", "0"))
        rows.append(dict(method=m, dataset=ds, cost_norm_hier=f"{float(r['cost_hier']):.6f}",
                         cost_abs_hier="", batches=r["batches"], language=la, repo=repo, commit=com,
                         notes="DyLo hierarchical re-summarization cost (ancestor-closure, def:cost) at default lambda=0.1"))

    # flat baselines: hierarchical cost = flat leaf cost (lower bound)
    for (m, ds), c in sorted(flat_cost.items()):
        if m in FLAT:
            la, repo, com = meta[m]
            rows.append(dict(method=m, dataset=ds, cost_norm_hier=c, cost_abs_hier="",
                             batches="", language=la, repo=repo, commit=com, notes=FLAT_NOTE))

    # dedup by (method,dataset) keep last
    seen = {}
    for r in rows: seen[(r["method"], r["dataset"])] = r
    with open(R/"resumm_hier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HDR); w.writeheader()
        for r in sorted(seen.values(), key=lambda r: (r["method"], r["dataset"])):
            w.writerow({k: r.get(k, "") for k in HDR})
    print(f"[resumm_hier] wrote {len(seen)} rows; methods: {sorted({r['method'] for r in seen.values()})}")


if __name__ == "__main__":
    main()
