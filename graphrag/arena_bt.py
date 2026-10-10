#!/usr/bin/env python3
"""Bradley-Terry scores for arena rankings: fit pairwise wins into one strength scale.

Why not average rank:
  1) Average rank ignores *whom* one loses to -- losing to the strongest and to the weakest cost the same;
  2) An equal-weight average over four dimensions mixes in directness, which is a **control dimension**
     on which DyLo is expected to rank low (its answers are longer and broader). Including it penalizes
     DyLo for a strength.

Therefore:
  - each of the three quality dimensions (comprehensiveness / diversity / empowerment) gets its own
    Bradley-Terry fit, plus a pooled scale;
  - directness is **reported separately and excluded from the total**; it falsifies the alternative "the answers are just longer":
    if DyLo wins the three quality dimensions but not directness, the length explanation fails.

BT model: P(i beats j) = exp(s_i) / (exp(s_i)+exp(s_j)), fitted by MM iterations,
with the reference fixed at 0; confidence intervals by bootstrap over questions.
"""
import math
import random
from collections import defaultdict


def fit_bt(wins: dict[tuple[str, str], int], items: list[str],
           iters: int = 500, tol: float = 1e-9) -> dict[str, float]:
    """Fit Bradley-Terry with the MM algorithm. wins[(a,b)] = number of times a beats b."""
    p = {i: 1.0 for i in items}
    tot_w = defaultdict(float)
    for (a, b), c in wins.items():
        tot_w[a] += c
    for _ in range(iters):
        new = {}
        for i in items:
            denom = 0.0
            for j in items:
                if i == j:
                    continue
                n_ij = wins.get((i, j), 0) + wins.get((j, i), 0)
                if n_ij:
                    denom += n_ij / (p[i] + p[j])
            new[i] = (tot_w[i] / denom) if denom > 0 else p[i]
        # normalize to avoid drift
        g = sum(new.values()) / len(new)
        new = {k: v / g for k, v in new.items()}
        if max(abs(new[k] - p[k]) for k in items) < tol:
            p = new
            break
        p = new
    # convert to log scale and subtract the mean
    s = {k: math.log(max(v, 1e-12)) for k, v in p.items()}
    m = sum(s.values()) / len(s)
    return {k: v - m for k, v in s.items()}


def wins_from_rankings(rankings: dict, items: list[str]) -> dict:
    """rankings[(qid, dim)] = [methods from best to worst] -> pairwise win counts."""
    w = defaultdict(int)
    for seq in rankings.values():
        pos = {a: i for i, a in enumerate(seq)}
        for a in items:
            for b in items:
                if a != b and pos.get(a, 1e9) < pos.get(b, 1e9):
                    w[(a, b)] += 1
    return w


def bt_with_ci(rankings: dict, items: list[str], n_boot: int = 400,
               seed: int = 42) -> dict[str, tuple[float, float, float]]:
    """Return {method: (strength, lower, upper)}; bootstrap resamples questions."""
    base = fit_bt(wins_from_rankings(rankings, items), items)
    qs = sorted({k for k, _ in rankings})
    rng = random.Random(seed)
    boots = defaultdict(list)
    for _ in range(n_boot):
        pick = [rng.choice(qs) for _ in qs]
        sub = {}
        for n, q in enumerate(pick):
            for (qq, dim), seq in rankings.items():
                if qq == q:
                    sub[(n, dim)] = seq
        try:
            f = fit_bt(wins_from_rankings(sub, items), items)
        except Exception:  # noqa: BLE001
            continue
        for k, v in f.items():
            boots[k].append(v)
    out = {}
    for k in items:
        b = sorted(boots[k])
        if len(b) >= 20:
            lo, hi = b[int(0.025 * len(b))], b[int(0.975 * len(b))]
        else:
            lo = hi = base[k]
        out[k] = (base[k], lo, hi)
    return out
