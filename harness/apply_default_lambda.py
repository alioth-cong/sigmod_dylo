#!/usr/bin/env python3
"""
Apply the paper's default lambda (knee = 0.1) to the main-table DyLo rows. The structure
(Q) and detection latency are ~lambda-independent (lambda gates re-summarisation, not the
affected-region update); the re-summarisation COST drops to the default-lambda value. So:
  quality.csv   dylo-* value      -> Q at lambda=0.1   (~unchanged)
  resumm.csv    dylo-* cost cols   -> cost_flat at lambda=0.1  (drops; other cost_norm_* and
                                      cost_abs scaled by the same ratio for consistency)
  resumm_hier.csv dylo-* cost_norm_hier -> cost_hier at lambda=0.1
latency.csv is left unchanged. livejournal (no hier lambda data) is left at its prior value.

Reads results/lambda.csv (lambda=1e-1 rows). Idempotent.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
from pathlib import Path

R = Path(DYLO_ROOT + "/results")
LAM = "1e-1"
NOTE = "; default lambda=0.1 (knee)"


def load_lambda():
    d = {}
    for r in csv.DictReader(open(R/"lambda.csv")):
        if r["lambda"] == LAM:
            d[(f"dylo-{r['instance']}", r["dataset"])] = r
    return d


def rewrite(path, fn):
    rows = list(csv.DictReader(open(path)))
    hdr = list(rows[0].keys())
    for r in rows:
        fn(r)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)


def main():
    L = load_lambda()
    n = {"quality": 0, "resumm": 0, "resumm_hier": 0}

    def q(r):
        k = (r["method"], r["dataset"])
        if k in L and "default lambda" not in r["notes"]:
            r["value"] = f"{float(L[k]['Q']):.6f}"; r["notes"] += NOTE; n["quality"] += 1
    rewrite(R/"quality.csv", q)

    def rs(r):
        k = (r["method"], r["dataset"])
        if k in L and "default lambda" not in r["notes"]:
            new = float(L[k]["cost_flat"]); old = float(r["cost_norm_weight"])
            ratio = (new/old) if old > 0 else 0.0
            r["cost_norm_weight"] = f"{new:.6f}"
            for c in ("cost_norm_fromscratch", "cost_norm_fromscratch_leiden"):
                if r.get(c) and r[c] not in ("", "NaN"):
                    try: r[c] = f"{float(r[c])*ratio:.6f}"
                    except ValueError: pass
            if r.get("cost_abs") and r["cost_abs"] not in ("", "NaN"):
                try: r["cost_abs"] = f"{float(r['cost_abs'])*ratio:.3f}"
                except ValueError: pass
            r["notes"] += NOTE; n["resumm"] += 1
    rewrite(R/"resumm.csv", rs)

    def rh(r):
        k = (r["method"], r["dataset"])
        if k in L and "default lambda" not in r["notes"]:
            new = float(L[k]["cost_hier"]); old = float(r["cost_norm_hier"])
            ratio = (new/old) if old > 0 else 0.0
            r["cost_norm_hier"] = f"{new:.6f}"
            if r.get("cost_abs_hier") and r["cost_abs_hier"] not in ("", "NaN"):
                try: r["cost_abs_hier"] = f"{float(r['cost_abs_hier'])*ratio:.3f}"
                except ValueError: pass
            r["notes"] += NOTE; n["resumm_hier"] += 1
    rewrite(R/"resumm_hier.csv", rh)

    print(f"[default-lambda] updated dylo-* rows: quality={n['quality']} resumm={n['resumm']} "
          f"resumm_hier={n['resumm_hier']} (lambda=0.1; latency unchanged)")


if __name__ == "__main__":
    main()
