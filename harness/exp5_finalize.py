#!/usr/bin/env python3
"""
Merge per-job Exp-5 CSVs (/tmp/lam/*.csv) into results/lambda.csv, then append the
GreMod reference point per dataset (instance=gremod, lambda=na) so the frontier figure
can overlay it as a dominated dot. GreMod is flat/add-only: GreMod_hier := GreMod_flat
(a lower bound on its hierarchical cost; flat methods earn no shared-ancestor savings).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv, glob, sys
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
RESULTS = ROOT / "results"
HEADER = ["instance", "dataset", "lambda", "Q", "cost_hier", "cost_flat",
          "dirty_frac", "latency_ms", "num_comms", "batches"]
SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lam2"


def fnum(x):
    return float("inf") if x in ("inf", "Inf") else float(x)


def selfcheck(rows):
    """Verify Q is monotone non-increasing in lambda and cost_hier>=cost_flat (at lambda>0)."""
    by = {}
    for r in rows:
        by.setdefault((r["instance"], r["dataset"]), []).append(r)
    okQ = okC = 0; badQ = []; badC = []
    for key, rr in by.items():
        rr = sorted(rr, key=lambda r: fnum(r["lambda"]))
        Qs = [float(r["Q"]) for r in rr]
        # monotone non-increasing within tolerance (re-detect rebuild adds tiny noise)
        mono = all(Qs[i] <= Qs[i-1] + 2e-3 for i in range(1, len(Qs)))
        (okQ := okQ + 1) if mono else badQ.append(key)
        for r in rr:
            if fnum(r["lambda"]) > 0 and float(r["cost_hier"]) + 1e-4 < float(r["cost_flat"]):
                badC.append((key, r["lambda"])); break
        else:
            okC += 1
    print(f"[selfcheck] Q monotone non-increasing: {okQ}/{len(by)} ok"
          + (f"; VIOLATIONS {badQ}" if badQ else ""))
    print(f"[selfcheck] cost_hier>=cost_flat (lambda>0): {okC}/{len(by)} ok"
          + (f"; VIOLATIONS {badC}" if badC else ""))


def main():
    rows = []
    for f in sorted(glob.glob(f"{SRC}/*.csv")):
        for r in csv.DictReader(open(f)):
            rows.append(r)
    # dedup by (instance,dataset,lambda) keeping last
    seen = {}
    for r in rows:
        seen[(r["instance"], r["dataset"], r["lambda"])] = r
    rows = list(seen.values())

    # GreMod reference per dataset
    gq, gcost = {}, {}
    for r in csv.DictReader(open(RESULTS / "quality.csv")):
        if r["method"] == "gremod" and r["value"]:
            gq[r["dataset"]] = r["value"]
    for r in csv.DictReader(open(RESULTS / "resumm.csv")):
        if r["method"] == "gremod" and r["cost_norm_weight"]:
            gcost[r["dataset"]] = r["cost_norm_weight"]
    dsets = {r["dataset"] for r in rows}
    gremod_rows = []
    for ds in sorted(dsets):
        if ds in gcost:
            gremod_rows.append({"instance": "gremod", "dataset": ds, "lambda": "na",
                                "Q": f"{float(gq[ds]):.5f}", "cost_hier": f"{float(gcost[ds]):.5f}",
                                "cost_flat": f"{float(gcost[ds]):.5f}", "dirty_frac": "nan",
                                "latency_ms": "nan", "num_comms": "", "batches": ""})

    out = RESULTS / "lambda.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["instance"], r["dataset"],
                        float("inf") if r["lambda"] in ("inf", "Inf") else float(r["lambda"]))):
            w.writerow({k: r.get(k, "") for k in HEADER})
        for r in gremod_rows:
            w.writerow(r)
    print(f"[exp5_finalize] wrote {len(rows)} DyLo rows + {len(gremod_rows)} gremod rows to {out}")
    selfcheck(rows)


if __name__ == "__main__":
    main()
