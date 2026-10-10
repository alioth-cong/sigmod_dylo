#!/usr/bin/env python3
"""Bradley-Terry with a length covariate: separate "longer answer" from "better method".

Motivation: DyLo's median answer has 2,538 characters versus 1,750-2,058 for the others. Judges do
prefer longer answers on the three quality dimensions ("longest ranked first" in 57%/52%/30% of cases), so
the standard BT strength s_i mixes "better content" with "longer writing".

Model (conditional logit):

    P(i beats j | question q) = sigma( s_i - s_j + beta * Delta_{ijq} )

    Delta_{ijq} = ( log len(i,q) - log len(j,q) ) / sd   (standardized over all pairs)

  beta measures how much the odds rise per standard deviation of log-length difference;
  s_i becomes the method strength **at equal length**.

Identifiability: beta is identified by **length-rank reversals of the same method across questions**. DyLo is
longest on only 25% of questions and shorter than each rival on 25-35%, so the overlap is ample and beta is not collinear with s.

Important caveat: length here is **not a pure confounder** but a **mediator** of the mechanism under test.
DyLo's answers are longer because its index exposes 558 reports at level<=1 (others 453-486),
so global search has more evidence to synthesize. Conditioning on a mediator **underestimates** the total effect
(post-treatment / mediator adjustment bias). Both quantities are therefore reported:

    standard BT s_i       = total effect of deploying DyLo (including the "hence longer" path)
    covariate-adjusted s_i = direct effect, i.e. "is the content better at equal length"

The former answers "is switching to DyLo worth it"; the latter answers "is the advantage only more words".

Usage: arena_bt_length.py <judge-pack dir> [--dims quality|all]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import glob
import itertools
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TMP = Path(DYLO_SCRATCH + "")
DIMS = ["comprehensiveness", "diversity", "empowerment", "directness"]
QUALITY = ["comprehensiveness", "diversity", "empowerment"]


def load_rankings(pack: Path, key) -> dict:
    """{(qid, dim, order): [methods from best to worst]}"""
    arms, letters = key["arms"], key["letters"]
    out = {}
    for o in (1, 2):
        omap = {int(k): v for k, v in key["orders"][o - 1].items()}
        for f in sorted(glob.glob(f"{pack}/order{o}-part*.csv")):
            for r in csv.DictReader(open(f, encoding="utf-8-sig")):
                kcol = [x for x in r if x.endswith("qid")][0]
                q = int(r[kcol][1:])
                dim = (r["criterion"] or "").strip().lower()
                seq = (r["ranking"] or "").strip().split(">")
                if q not in omap or dim not in DIMS or sorted(seq) != sorted(letters):
                    continue
                out[(q, dim, o)] = [arms[omap[q][letters.index(x)]] for x in seq]
    return out


def build_pairs(rk, arms, lens, dims):
    """Count a win only if both orders agree; return [(i, j, delta)] with i beating j."""
    pairs = []
    qs = sorted({q for q, _, _ in rk})
    for q in qs:
        for dim in dims:
            r1, r2 = rk.get((q, dim, 1)), rk.get((q, dim, 2))
            if not r1 or not r2:
                continue
            p1 = {a: i for i, a in enumerate(r1)}
            p2 = {a: i for i, a in enumerate(r2)}
            for a, b in itertools.combinations(arms, 2):
                w1, w2 = p1[a] < p1[b], p2[a] < p2[b]
                if w1 != w2:
                    continue            # the two orders disagree -> tie, dropped
                i, j = (a, b) if w1 else (b, a)
                d = math.log(max(lens[i][q], 1)) - math.log(max(lens[j][q], 1))
                pairs.append((i, j, d))
    return pairs


def fit(pairs, arms, use_beta=True, iters=400, lr=0.5):
    """Newton-gradient hybrid: maximum likelihood over s (zero mean) and beta."""
    idx = {a: k for k, a in enumerate(arms)}
    s = [0.0] * len(arms)
    beta = 0.0
    for _ in range(iters):
        gs = [0.0] * len(arms)
        gb = 0.0
        hs = [1e-6] * len(arms)
        hb = 1e-6
        for i, j, d in pairs:
            z = s[idx[i]] - s[idx[j]] + (beta * d if use_beta else 0.0)
            p = 1.0 / (1.0 + math.exp(-z))
            r = 1.0 - p                      # the observation is always "i wins"
            w = p * (1 - p)
            gs[idx[i]] += r; gs[idx[j]] -= r
            hs[idx[i]] += w; hs[idx[j]] += w
            if use_beta:
                gb += r * d
                hb += w * d * d
        for k in range(len(arms)):
            s[k] += lr * gs[k] / hs[k]
        if use_beta:
            beta += lr * gb / hb
        m = sum(s) / len(s)
        s = [x - m for x in s]
    return {a: s[idx[a]] for a in arms}, beta


def boot(pairs_by_q, arms, use_beta, n=400, seed=20260828):
    rng = random.Random(seed)
    qs = list(pairs_by_q)
    S = defaultdict(list); B = []
    for _ in range(n):
        pick = [rng.choice(qs) for _ in qs]
        pp = [p for q in pick for p in pairs_by_q[q]]
        try:
            s, b = fit(pp, arms, use_beta, iters=120)
        except Exception:  # noqa: BLE001
            continue
        for a in arms:
            S[a].append(s[a])
        B.append(b)
    def ci(v):
        v = sorted(v)
        return (v[int(.025 * len(v))], v[int(.975 * len(v))]) if len(v) >= 20 else (float("nan"),) * 2
    return {a: ci(S[a]) for a in arms}, (ci(B) if use_beta else (float("nan"),) * 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("--dims", default="quality", choices=["quality", "all"])
    ap.add_argument("--range", default="")
    a = ap.parse_args()
    pack = Path(a.pack)
    key = json.load(open(pack / "key.json"))
    arms = key["arms"]
    dims = QUALITY if a.dims == "quality" else DIMS

    lens = {arm: [len(x["answer"]) if x else 0
                  for x in json.load(open(TMP / "sense_ans" / f"{arm}.json"))]
            for arm in arms}

    rk = load_rankings(pack, key)
    if a.range:
        lo, hi = (int(x) for x in a.range.split(":"))
        rk = {k: v for k, v in rk.items() if lo <= k[0] < hi}
    pairs = build_pairs(rk, arms, lens, dims)
    if not pairs:
        sys.exit("no usable consistent verdicts")

    # standardize Delta
    sd = statistics.pstdev([d for _, _, d in pairs]) or 1.0
    pairs = [(i, j, d / sd) for i, j, d in pairs]
    pbq = defaultdict(list)
    for (q, dim, o) in rk:
        pass
    # group by question for the bootstrap
    qs = sorted({q for q, _, _ in rk})
    for q in qs:
        sub = build_pairs({k: v for k, v in rk.items() if k[0] == q}, arms, lens, dims)
        pbq[q] = [(i, j, d / sd) for i, j, d in sub]

    print(f"usable consistent verdicts: {len(pairs)} pairs over {len(qs)} questions, dims {dims}")
    print(f"sd of log-length difference = {sd:.4f}\n")

    s0, _ = fit(pairs, arms, use_beta=False)
    s1, beta = fit(pairs, arms, use_beta=True)
    ci0, _ = boot(pbq, arms, False)
    ci1, cib = boot(pbq, arms, True)

    print(f"length effect beta = {beta:+.4f}  95% CI [{cib[0]:+.4f}, {cib[1]:+.4f}]")
    print(f"  i.e. +1 sd of log-length difference ({sd:.3f}) multiplies the odds by {math.exp(beta):.3f}"
          f", a win rate of {1/(1+math.exp(-beta)):.1%}\n")

    print(f"{'method':<24}{'std BT':>10}{'95% CI':>20}{'eq-len BT':>10}{'95% CI':>20}{'diff':>8}")
    for arm in sorted(arms, key=lambda x: -s0[x]):
        print(f"{arm:<24}{s0[arm]:>+10.3f}[{ci0[arm][0]:+6.2f},{ci0[arm][1]:+6.2f}]"
              f"{s1[arm]:>+10.3f}[{ci1[arm][0]:+6.2f},{ci1[arm][1]:+6.2f}]"
              f"{s1[arm]-s0[arm]:>+8.3f}")

    json.dump({"beta": beta, "beta_ci": cib, "sd_logdiff": sd,
               "bt_plain": s0, "bt_plain_ci": ci0,
               "bt_length_adjusted": s1, "bt_length_adjusted_ci": ci1},
              open(pack / "bt_length.json", "w"), indent=1)
    print(f"\nsaved: {pack/'bt_length.json'}")


if __name__ == "__main__":
    main()
