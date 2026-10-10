#!/usr/bin/env python3
"""
Hierarchical re-summarization cost, split into the two terms of Definition
def:cost (structure-altering vs structure-preserving updates).

  restructuring cost : a node whose membership or child-set changed -- and every
                       ancestor of such a node -- has a summary that describes a
                       member set that no longer exists. It must be regenerated
                       in full, at weight w(c) (leaf: #members, internal:
                       #children). This is exactly what hier_cost.py computes.

  content-refresh cost : a node whose membership and child-set are UNCHANGED but
                       whose internal edge weight grew. Its summary is still
                       correct, only incomplete -- one further relation among
                       entities it already covers -- so the deployment appends
                       rather than rewrites. Charged KAPPA = 1 unit, not w(c).

Total = sum_{c dirty} w(c) + KAPPA * |{c refreshed}|, normalised by the same
total tree weight as hier_cost.py, so the two are directly comparable and the
difference is attributable solely to the content-refresh term.

The rule is applied uniformly to every method, so it neither favours nor
penalises any of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dirty_cost import read_flat_lists, build_index  # noqa
from hier_cost import read_levels, _parents  # noqa

KAPPA = 1.0


def load_batch_edges(run_dir: Path, k: int):
    """Edges ARRIVING in batch k (k=0 -> none; the initial graph is the baseline).

    Only the batch delta is needed: a structure-preserving node has an unchanged
    member set, so its internal edge weight changed in batch k iff at least one
    edge of inc_k has both endpoints inside it. Re-reading the whole accumulated
    edge set per batch would be quadratic in the stream length and gives the
    same answer for exactly the nodes this term charges.
    """
    if k <= 0:
        return []
    fp = run_dir / f"inc_{k}.txt"
    if not fp.exists():
        return []
    out = []
    with open(fp) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                out.append((int(p[0]), int(p[1])))
    return out


def _match(prev: list[list[int]], curr: list[list[int]]):
    """Per curr community: (self_dirty, best_prev_index or None).

    self_dirty <=> best Jaccard to prev < 1.0 (membership/child-set changed or new).
    """
    if not prev:
        return [True] * len(curr), [None] * len(curr)
    prev_sizes = [len(p) for p in prev]
    pidx = build_index(prev)
    sd, match = [], []
    for c in curr:
        lc = len(c)
        cnt: dict[int, int] = {}
        for n in c:
            pi = pidx.get(n)
            if pi is not None:
                cnt[pi] = cnt.get(pi, 0) + 1
        best, best_pi = 0.0, None
        for pi, o in cnt.items():
            j = o / (lc + prev_sizes[pi] - o)
            if j > best:
                best, best_pi = j, pi
        sd.append(best < 1.0)
        match.append(best_pi)
    return sd, match


def _gained(levels, batch_edges):
    """gained[i][c] = True iff a batch-k edge landed strictly inside level-i node c."""
    out = []
    for lev in levels:
        idx = build_index(lev)
        hit = [False] * len(lev)
        for a, b in batch_edges:
            ca = idx.get(a)
            if ca is not None and ca == idx.get(b):
                hit[ca] = True
        out.append(hit)
    return out


def hier_cost_batch_edges(prev_levels, curr_levels, gained):
    """(cost_abs, cost_norm, restructuring_abs, refresh_abs) for one batch.

    `gained[i][c]` says whether a batch edge landed inside level-i node c.
    """
    if not curr_levels:
        return 0.0, 0.0, 0.0, 0.0
    nlev = len(curr_levels)
    sd, match = [], []
    for i in range(nlev):
        prev_i = prev_levels[i] if (prev_levels and i < len(prev_levels)) else []
        a, b = _match(prev_i, curr_levels[i])
        sd.append(a)
        match.append(b)
    parents = _parents(curr_levels)
    nchildren = [[0] * len(curr_levels[i]) for i in range(nlev)]
    for i in range(nlev - 1):
        for c, par in enumerate(parents[i]):
            if par >= 0:
                nchildren[i + 1][par] += 1
    dirty = [list(sd[i]) for i in range(nlev)]
    for i in range(nlev - 1):
        for c in range(len(curr_levels[i])):
            if dirty[i][c]:
                par = parents[i][c]
                if par >= 0:
                    dirty[i + 1][par] = True

    restr = refresh = total = 0.0
    for i in range(nlev):
        for c in range(len(curr_levels[i])):
            w = len(curr_levels[i][c]) if i == 0 else nchildren[i][c]
            total += w
            if dirty[i][c]:
                restr += w
            elif match[i][c] is not None and gained[i][c]:
                # structure-preserving: same members/children, more relations
                refresh += KAPPA
    cost = restr + refresh
    return cost, (cost / total if total > 0 else 0.0), restr, refresh


def stream_hier_cost_edges(out_dir: Path, run_dir: Path, n_batches: int):
    """[(batch, cost_abs, cost_norm, restr_abs, refresh_abs)]."""
    rows = []
    prev = None
    for k in range(0, n_batches + 1):
        if not (out_dir / f"com_{k}_L0.txt").exists():
            break
        curr = read_levels(out_dir, k)
        if prev is not None:
            gained = _gained(curr, load_batch_edges(run_dir, k))
            ca, cn, r, f = hier_cost_batch_edges(prev, curr, gained)
            rows.append((k, ca, cn, r, f))
        prev = curr
    return rows


def _selftest():
    # Same fixture as hier_cost._selftest, with an extra edge inside the
    # unchanged community {0,1} so it must be content-refreshed for KAPPA.
    A = [[[0, 1], [2, 3], [4, 5]], [[0, 1, 2, 3], [4, 5]]]
    B = [[[0, 1], [2, 3, 5], [4]], [[0, 1, 2, 3, 5], [4]]]
    # the batch's new edge (0,1) lands strictly inside the unchanged leaf {0,1}
    gained = _gained(B, [(0, 1)])
    ca, cn, r, f = hier_cost_batch_edges(A, B, gained)
    # restructuring is unchanged from hier_cost: 7
    assert abs(r - 7.0) < 1e-9, r
    # {0,1} is the only clean node that gained an internal edge -> +1
    assert abs(f - 1.0) < 1e-9, f
    assert abs(ca - 8.0) < 1e-9, ca
    assert abs(cn - 8.0 / 9.0) < 1e-9, cn
    # no new internal edge -> identical to hier_cost
    _, _, r2, f2 = hier_cost_batch_edges(A, B, _gained(B, []))
    assert f2 == 0.0 and abs(r2 - 7.0) < 1e-9, (r2, f2)
    print("hier_cost_edges selftest OK: restr=7, refresh=1, total=8/9")


if __name__ == "__main__":
    _selftest()
