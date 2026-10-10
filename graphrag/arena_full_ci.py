#!/usr/bin/env python3
"""Arena robustness for Set B: CIs for all eight arms plus multiple-comparison correction.

Reviewers noted four open points; this script closes two of them (pure post-processing, no re-judging):
  (1) confidence intervals were shown only for DyLo -> report 95% CIs of BT strength for all eight arms
  (2) multiple comparisons were not addressed -> Holm-Bonferroni over all C(8,2)=28 pairwise comparisons,
      reporting both raw and adjusted significance, i.e. whether DyLo's lead survives correction

The bootstrap resamples **questions** as clusters (the 28 pairwise verdicts of a question are not independent), as in the main table.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import glob, itertools, json, math, sys
from pathlib import Path
sys.path.insert(0, DYLO_ROOT + "/graphrag")
# load_rankings lives in arena_bt_length.py; arena_bt.py only has the fit and the CI
from arena_bt import wins_from_rankings, fit_bt, bt_with_ci  # noqa
from arena_bt_length import load_rankings, QUALITY  # noqa

import os
_SET = os.environ.get("ARENA_SET", "B")
PACKS = ({"GPT-5.6 Sol": "arena8_60", "Opus 5": "arena8_60_opus"} if _SET == "A"
         else {"GPT-5.6 Sol": "new60_gpt", "Opus 5": "new60_opus", "Gemini 3.7": "new60_gemini"})
B = Path(DYLO_MHRAG + "/clean_chain")

def holm(pvals):
    """Holm-Bonferroni: return adjusted p-values (order preserving)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i])
        out[i] = min(1.0, run)
    return out

def main():
    for judge, pack in PACKS.items():
        d = B / pack
        if not d.exists():
            print(f"{judge}: missing {pack}"); continue
        key = json.load(open(d / "key.json"))
        arms = key["arms"]
        rk3 = load_rankings(d, key)   # {(qid,dim,order): ranking}
        # As in the main table: a win counts only if the **pair** agrees in both orders (not the whole ranking). The order-1
        # ranking represents the question in BT; pairs that disagree cancel out in the win counts.
        rk = {(q, dim): seq for (q, dim, o), seq in rk3.items()
              if o == 1 and dim in QUALITY}
        if not rk:
            print(f"{judge}: no usable verdicts"); continue
        res = bt_with_ci(rk, arms, n_boot=1000)   # {method: (strength, lower, upper)}
        s = {k: v[0] for k, v in res.items()}
        ci = {k: (v[1], v[2]) for k, v in res.items()}
        print(f"  questions with usable verdicts {len({q for q,_ in rk})}, (question,dim) consistent in both orders {len(rk)}")
        print(f"\n===== {judge} =====")
        print(f"{'method':<24}{'BT strength':>10}{'95% CI':>22}")
        for a in sorted(arms, key=lambda x: -s[x]):
            print(f"{a:<24}{s[a]:>+10.3f}   [{ci[a][0]:+.3f}, {ci[a][1]:+.3f}]")
        # pairwise win rates + binomial test + Holm
        w = wins_from_rankings(rk, arms)
        pairs, ps, info = [], [], []
        for a, b in itertools.combinations(arms, 2):
            na, nb = w.get((a, b), 0), w.get((b, a), 0)
            n = na + nb
            if n == 0: continue
            # two-sided binomial test, H0: p=0.5 (normal approximation, n is large)
            z = (na - n / 2) / math.sqrt(n / 4)
            p = math.erfc(abs(z) / math.sqrt(2))
            pairs.append((a, b)); ps.append(p)
            info.append((a, b, na, nb, na / n, p))
        adj = holm(ps)
        sig_raw = sum(1 for p in ps if p < .05)
        sig_adj = sum(1 for p in adj if p < .05)
        print(f"\n  {len(ps)} pairwise comparisons: {sig_raw} significant unadjusted, {sig_adj} after Holm")
        dy = [i for i, a in enumerate(arms) if "norecl" in a]
        if dy:
            tgt = arms[dy[0]]
            print(f"  {tgt}'s 7 comparisons (adjusted):")
            for (a, b, na, nb, wr, p), pa in zip(info, adj):
                if tgt not in (a, b): continue
                other = b if a == tgt else a
                rate = wr if a == tgt else 1 - wr
                print(f"    vs {other:<22} win rate {rate:.1%}  p={p:.2e}  Holm p={pa:.2e}"
                      f"  {'significant' if pa < .05 else 'not significant'}")

if __name__ == "__main__":
    main()
