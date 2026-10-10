#!/usr/bin/env python
"""Node-level fidelity: the old structural weights vs the tier-B closed form vs the tier-A dry run.

Input: per-community CSVs from proxy_dryrun.py (only rows with a cache-key hit; ground truth = usage.prompt_tokens).

Models (one model fitted to all nodes, then reported for leaf / internal / all):
  M0   the paper's current weights r_int=1: leaves count entities, internal nodes count children; tok = a + b*w
  M0r  same, with internal weights multiplied by a fitted r_int (the "rescaled" row of Table 10)
  B1   tier B, counts only: tok = min(B, a + b_e*|ent(subtree)|)
  B2   tier B, counts + edges: tok = min(B, a + b_e*|ent(subtree)| + b_r*|E_ctx|)
  B3   tier B with text length: tok = min(B, a + b*raw), raw = sum of tokens of each entity/relationship row in the subtree
       (incrementally maintainable); the cap B is set by max_input_length
  A    tier-A dry run: tok = a + b*prompt_tok (the rebuilt full prompt)

--train selects the calibration index (default: each file calibrates itself); other files form the transfer test set.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import spearmanr


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - p) ** 2).sum() / sst if sst > 0 else float("nan")
    rho = spearmanr(y, p).correlation if len(y) > 2 else float("nan")
    mape = np.mean(np.abs(y - p) / y)
    return r2, rho, mape


def fit_models(d: pd.DataFrame) -> dict:
    y = d.usage_prompt.values.astype(float)
    leaf = d.is_leaf.values == 1
    m = {}
    # M0: a single linear fit, w = entities (leaf) / children (internal)
    w0 = np.where(leaf, d.n_ent, d.n_children).astype(float)
    b, a = np.polyfit(w0, y, 1)
    m["M0"] = ("lin", dict(a=a, b=b))
    # M0r: internal weights multiplied by r_int
    X = np.c_[np.ones(len(y)), np.where(leaf, d.n_ent, 0), np.where(leaf, 0, d.n_children)]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    m["M0r"] = ("m0r", dict(a=c[0], bl=c[1], bi=c[2]))

    def capfit(cols, name):
        Z = d[cols].values.astype(float)
        def f(t):
            return np.minimum(t[0], t[1] + Z @ t[2:])
        x0 = np.r_[y.max(), y.min(), np.full(len(cols), (y.mean() - y.min()) / max(Z.mean(), 1))]
        r = least_squares(lambda t: f(t) - y, x0)
        m[name] = ("cap", dict(B=r.x[0], a=r.x[1], coef=r.x[2:], cols=cols))
    capfit(["n_ent"], "B1")
    capfit(["n_ent", "n_edges_ctx"], "B2")
    capfit(["raw_tok"], "B3")
    b, a = np.polyfit(d.prompt_tok, y, 1)
    m["A"] = ("lin_p", dict(a=a, b=b))
    return m


def predict(kind_par, d: pd.DataFrame) -> np.ndarray:
    kind, p = kind_par
    leaf = d.is_leaf.values == 1
    if kind == "lin":
        return p["a"] + p["b"] * np.where(leaf, d.n_ent, d.n_children)
    if kind == "m0r":
        return p["a"] + np.where(leaf, p["bl"] * d.n_ent, p["bi"] * d.n_children)
    if kind == "cap":
        Z = d[p["cols"]].values.astype(float)
        return np.minimum(p["B"], p["a"] + Z @ p["coef"])
    if kind == "lin_p":
        return p["a"] + p["b"] * d.prompt_tok
    raise ValueError(kind)


def report(models, d, title):
    print(f"\n### {title}  (n={len(d)}: leaf {int(d.is_leaf.sum())} / internal {int((d.is_leaf == 0).sum())})")
    print(f"{'model':<6} | {'leaf R²':>8} {'ρ':>6} {'MAPE':>6} | {'int R²':>8} {'ρ':>6} {'MAPE':>6} | "
          f"{'all R²':>8} {'ρ':>6} {'MAPE':>6}")
    out = {}
    for name, mp in models.items():
        p = predict(mp, d)
        y = d.usage_prompt.values
        res = []
        for sub in (d.is_leaf == 1, d.is_leaf == 0, np.ones(len(d), bool)):
            res += list(metrics(y[sub], p[sub]))
        out[name] = res
        print(f"{name:<6} | {res[0]:8.3f} {res[1]:6.3f} {res[2]:6.1%} | {res[3]:8.3f} {res[4]:6.3f} "
              f"{res[5]:6.1%} | {res[6]:8.3f} {res[7]:6.3f} {res[8]:6.1%}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--train", default="", help="calibration CSV (default: each file calibrates itself)")
    args = ap.parse_args()
    load = lambda f: pd.read_csv(f).query("hit == 1").reset_index(drop=True)
    if args.train:
        tr = load(args.train)
        models = fit_models(tr)
        for k, (kind, p) in models.items():
            print(k, kind, {a: (np.round(b, 3) if not isinstance(b, list) else b) for a, b in p.items()})
        for f in args.csvs:
            report(models, load(f), f"{f.split('/')[-1]}  [calibrated on {args.train.split('/')[-1]}]")
    else:
        for f in args.csvs:
            d = load(f)
            models = fit_models(d)
            for k, (kind, p) in models.items():
                print(k, kind, {a: (np.round(b, 3) if not isinstance(b, list) else b) for a, b in p.items()})
            report(models, d, f.split("/")[-1] + "  [self-calibrated]")


if __name__ == "__main__":
    main()
