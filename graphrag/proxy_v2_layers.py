#!/usr/bin/env python
"""Validation layers 1-3: node-level transfer, batch-level correlation, method-ranking agreement.

Data: results/proxy_v2/<index>.csv from proxy_dryrun_all.sh (one row per real call;
usage of hit=1 rows is ground truth). Calibration uses a single full index (--calib); everything else is test data.

Three definitions of real cost (per call):
  in      usage.prompt_tokens
  in+out  prompt + completion
  $       prompt + R*completion (R = --price-ratio, output price / input price)

Proxies (predicted cost per call):
  old     the paper's current w: leaves count entities, internal nodes count children (r_int=1)
  B1      min(B, a + b*|ent(subtree)|) (counts only; fits in the C++ driver)
  B1+out  B1 + the mean output per call (times 1 or R), i.e. alpha + p_in*T_in + p_out*mean(T_out)
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import kendalltau, spearmanr

D = Path(DYLO_ROOT + "/results/proxy_v2")


def load(name):
    d = pd.read_csv(D / f"{name}.csv")
    d["index"] = name
    return d


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="rebuild327_fresh")
    ap.add_argument("--price-ratio", type=float, default=4.0)
    args = ap.parse_args()
    R = args.price_ratio

    names = sorted(Path(f).stem for f in glob.glob(str(D / "*.csv")))
    all_d = pd.concat([load(n) for n in names], ignore_index=True)
    h = all_d[all_d.hit == 1].copy()

    # ---- calibration (a single full index) ----
    c = h[h["index"] == args.calib]
    y = c.usage_prompt.values.astype(float)
    Z = c.n_ent.values.astype(float)
    fit = least_squares(lambda t: np.minimum(t[0], t[1] + t[2] * Z) - y,
                        [y.max(), y.min(), 80.0]).x
    B, a, b = fit
    tout = c.usage_completion.mean()
    leaf = c.is_leaf == 1
    bo, ao = np.polyfit(np.where(leaf, c.n_ent, c.n_children), y, 1)
    print(f"calibrated on {args.calib} (n={len(c)}): B1 = min({B:.0f}, {a:.0f} + {b:.2f}*|ent|); "
          f"mean output T_out={tout:.0f}; old weights as linear map tok~{ao:.0f}+{bo:.1f}*w")

    def pred(df, kind, cost):
        lf = df.is_leaf == 1
        if kind == "old":
            return np.where(lf, df.n_ent, df.n_children).astype(float)
        tin = np.minimum(B, a + b * df.n_ent.values)
        if kind == "B1":
            return tin
        if kind == "B1+out":
            return tin + {"in": 0.0, "in+out": tout, "$": R * tout}[cost]
        raise ValueError(kind)

    def truth(df, cost):
        if cost == "in":
            return df.usage_prompt.values.astype(float)
        if cost == "in+out":
            return (df.usage_prompt + df.usage_completion).values.astype(float)
        return (df.usage_prompt + R * df.usage_completion).values.astype(float)

    # ---- layer 1: node-level transfer (test set = all hit calls outside the calibration index) ----
    t = h[h["index"] != args.calib]
    full = t[~t["index"].str.contains("norecl|arm_")]
    inc = t[t["index"].str.contains("norecl|arm_")]
    print("\nLayer 1, node level (input tokens; B1 transfers the calibrated parameters as is, old uses the calibrated linear map)")
    print(f"{'test set':<26}{'n':>7} | {'old R2':>7} {'rho':>6} {'MAPE':>6} | {'B1 R2':>7} {'rho':>6} {'MAPE':>6}")
    for nm, s in [("16 full-rebuild indexes", full), ("56 incremental arms", inc), ("all", t)]:
        for sub_nm, sub in [("", s), (" · leaf", s[s.is_leaf == 1]), (" · internal", s[s.is_leaf == 0])]:
            yy = sub.usage_prompt.values
            po = ao + bo * pred(sub, "old", "in")
            pb = pred(sub, "B1", "in")
            print(f"{nm + sub_nm:<26}{len(sub):>7} | {r2(yy, po):7.3f} {spearmanr(yy, po).correlation:6.3f} "
                  f"{np.mean(np.abs(yy - po) / yy):6.1%} | {r2(yy, pb):7.3f} "
                  f"{spearmanr(yy, pb).correlation:6.3f} {np.mean(np.abs(yy - pb) / yy):6.1%}")

    # ---- per arm (one update = one "method x batch") ----
    def setting(n):
        if n.startswith("arm_"):
            m = re.match(r"arm_(.+?)(?:_(\d+c?))?$", n)
            return m.group(1), (m.group(2) or "327")
        if n.startswith("norecl"):
            return "dylo", n.replace("norecl", "") or "327"
        if n.startswith("rebuild"):
            return "rebuild", n.replace("rebuild", "").replace("_fresh", "")
        return None, None

    rows = []
    for n in names:
        meth, st = setting(n)
        if meth is None or n in ("norecl327_g5", "rebuild327"):
            continue  # g5 is a gamma-sweep variant; rebuild327 has an older cache lineage, the 327 group uses rebuild327_fresh
        d = all_d[all_d["index"] == n]
        hh = d[d.hit == 1]
        rec = dict(arm=n, method=meth, setting=st, calls=len(d), hit=len(hh))
        # calls without a hit get their input from the tier-A dry run (linear map with R2=.9995) and the mean output
        p_in = np.where(d.hit == 1, d.usage_prompt, 1.0713 * d.prompt_tok - 73.2)
        p_out = np.where(d.hit == 1, d.usage_completion, tout)
        rec["true_in"] = p_in.sum()
        rec["true_inout"] = (p_in + p_out).sum()
        rec["true_$"] = (p_in + R * p_out).sum()
        for k in ("old", "B1"):
            rec[f"pred_{k}"] = pred(d, k, "in").sum()
        rec["pred_B1out_inout"] = pred(d, "B1+out", "in+out").sum()
        rec["pred_B1out_$"] = pred(d, "B1+out", "$").sum()
        rows.append(rec)
    A = pd.DataFrame(rows)
    A.to_csv(D / "layers_arms.csv", index=False)

    # ---- layer 2: batch level (all "method x batch" points) ----
    print(f"\nLayer 2, batch level: {len(A)} (method x batch) points, predicted total cost vs measured total tokens")
    for tcol, pcols in [("true_in", ["pred_old", "pred_B1"]),
                        ("true_inout", ["pred_old", "pred_B1", "pred_B1out_inout"]),
                        ("true_$", ["pred_old", "pred_B1", "pred_B1out_$"])]:
        for pc in pcols:
            x, yv = A[pc].values, A[tcol].values
            bb, aa = np.polyfit(x, yv, 1)
            print(f"  {tcol:<10} ~ {pc:<17} R²={r2(yv, aa + bb * x):.4f}  "
                  f"ρ={spearmanr(x, yv).correlation:.4f}  "
                  f"median relative error {np.median(np.abs(yv - (aa + bb * x)) / yv):.1%}")

    # ---- layer 3: method ranking (all methods ranked within each batch) ----
    print(f"\nLayer 3, method ranking: Kendall tau between proxy rank and measured rank within each batch (price ratio R={R:g})")
    print(f"{'batch':<6}{'methods':>5} | {'in:old':>7} {'in:B1':>6} | {'io:old':>7} {'io:B1+o':>8} | "
          f"{'$:old':>6} {'$:B1+o':>7} | DyLo measured rank (in/io/$)")
    for st, g in A.groupby("setting"):
        if len(g) < 4:
            continue
        tau = lambda p, tc: kendalltau(g[p], g[tc]).correlation
        rk = lambda tc: int(g[tc].rank().loc[g.method == "dylo"].iloc[0]) if (g.method == "dylo").any() else -1
        print(f"{st:<6}{len(g):>5} | {tau('pred_old', 'true_in'):7.3f} {tau('pred_B1', 'true_in'):6.3f} | "
              f"{tau('pred_old', 'true_inout'):7.3f} {tau('pred_B1out_inout', 'true_inout'):8.3f} | "
              f"{tau('pred_old', 'true_$'):6.3f} {tau('pred_B1out_$', 'true_$'):7.3f} | "
              f"{rk('true_in')}/{rk('true_inout')}/{rk('true_$')} of {len(g)}")


if __name__ == "__main__":
    main()
