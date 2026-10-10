#!/usr/bin/env python3
"""
Hierarchical re-summarization cost (P1).

A dendrogram method emits, per batch k, a sequence of level partitions over the
ORIGINAL vertices:
    com_<k>_L0.txt   finest communities  (these are the LEAVES of the summary tree)
    com_<k>_L1.txt   communities of L0 communities
    ...
    com_<k>_L<top>.txt
Each line = one community = tab/space-separated vertex ids. Level i+1 is coarser
(nests level i).

Cost model (matches the writing-agent / paper spec):
  - Match communities across consecutive batches, per level, by best Jaccard >= 0.5
    of their vertex-member sets.
  - A node is SELF-dirty if its matched overlap < 1.0 (membership changed) or it has
    no >= 0.5 match (new/split).
  - DIRTY = self-dirty OR any child is dirty (propagate leaf->root).
  - weight w(c) = #vertices            for a leaf (level 0)
                  #children            for an internal node (level >= 1)
  - cost(batch) = sum of w(c) over the dirty set (leaves + dirtied ancestors).
  - cost_norm_weight = cost / (sum of w(c) over ALL nodes)   [worst case 1.0]

A single-level input degrades to the flat leaf cost (so flat methods are consistent).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dirty_cost import read_flat_lists, build_index  # noqa


def read_levels(out_dir: Path, k: int) -> list[list[list[int]]]:
    """Return [L0, L1, ...]; Li = list of communities (each a list of vertex ids)."""
    levels = []
    i = 0
    while True:
        f = out_dir / f"com_{k}_L{i}.txt"
        if not f.exists():
            break
        levels.append(read_flat_lists(f))
        i += 1
    return levels


def _self_dirty(prev: list[list[int]], curr: list[list[int]]) -> list[bool]:
    """Per curr community: True iff best Jaccard to prev < 1.0 (changed or new)."""
    if not prev:
        return [True] * len(curr)
    prev_sizes = [len(p) for p in prev]
    pidx = build_index(prev)
    out = []
    for c in curr:
        lc = len(c)
        cnt: dict[int, int] = {}
        for n in c:
            pi = pidx.get(n)
            if pi is not None:
                cnt[pi] = cnt.get(pi, 0) + 1
        best = 0.0
        for pi, o in cnt.items():
            j = o / (lc + prev_sizes[pi] - o)
            if j > best:
                best = j
        out.append(best < 1.0)
    return out


def _parents(levels: list[list[list[int]]]) -> list[list[int]]:
    """parents[i][c] = index of the level-(i+1) community containing level-i community c.
    Top level has no parents -> empty list at that index."""
    parents = []
    for i in range(len(levels) - 1):
        up_index = build_index(levels[i + 1])          # vertex -> level-(i+1) comm
        par = []
        for c in levels[i]:
            par.append(up_index.get(c[0], -1))         # nested: first vertex decides
        parents.append(par)
    parents.append([])  # top level
    return parents


def hier_cost_batch(prev_levels, curr_levels) -> tuple[float, float]:
    """Return (cost_abs, cost_norm_weight) for one batch."""
    if not curr_levels:
        return 0.0, 0.0
    nlev = len(curr_levels)
    # self-dirty per level
    sd = []
    for i in range(nlev):
        prev_i = prev_levels[i] if (prev_levels and i < len(prev_levels)) else []
        sd.append(_self_dirty(prev_i, curr_levels[i]))
    parents = _parents(curr_levels)
    # children counts per node
    nchildren = [[0] * len(curr_levels[i]) for i in range(nlev)]
    for i in range(nlev - 1):
        for c, par in enumerate(parents[i]):
            if par >= 0:
                nchildren[i + 1][par] += 1
    # propagate dirty leaf->root
    dirty = [list(sd[i]) for i in range(nlev)]
    for i in range(nlev - 1):
        for c in range(len(curr_levels[i])):
            if dirty[i][c]:
                par = parents[i][c]
                if par >= 0:
                    dirty[i + 1][par] = True
    # weights + cost
    cost = 0.0
    total = 0.0
    for i in range(nlev):
        for c in range(len(curr_levels[i])):
            w = len(curr_levels[i][c]) if i == 0 else nchildren[i][c]
            total += w
            if dirty[i][c]:
                cost += w
    return cost, (cost / total if total > 0 else 0.0)


def stream_hier_cost(out_dir: Path, n_batches: int) -> list[tuple[int, float, float]]:
    rows = []
    prev = None
    for k in range(0, n_batches + 1):
        if not (out_dir / f"com_{k}_L0.txt").exists():
            break
        curr = read_levels(out_dir, k)
        if prev is not None:
            ca, cn = hier_cost_batch(prev, curr)
            rows.append((k, ca, cn))
        prev = curr
    return rows


def _selftest():
    # 6 vertices. Batch A: L0 = {0,1},{2,3},{4,5}; L1 = {0,1,2,3},{4,5}
    # Batch B: vertex 5 moves from {4,5} to {2,3}: L0 = {0,1},{2,3,5},{4}; L1 = {0,1,2,3,5},{4}
    A = [[[0,1],[2,3],[4,5]], [[0,1,2,3],[4,5]]]
    B = [[[0,1],[2,3,5],[4]], [[0,1,2,3,5],[4]]]
    ca, cn = hier_cost_batch(A, B)
    # L0: {0,1} unchanged(not dirty); {2,3,5} changed; {4} new -> dirty leaves weight 3+1=4
    # L1: {0,1,2,3,5} changed + has dirty child -> dirty (2 children); {4} new -> dirty (1 child)
    #     internal dirty weight = 2 + 1 = 3
    # total weight = (2+3+1) leaves + (2+1) internal = 6 + 3 = 9 ; cost = 4 + 3 = 7
    assert abs(ca - 7.0) < 1e-9, ca
    assert abs(cn - 7.0/9.0) < 1e-9, cn
    # identical batch -> zero cost
    ca2, cn2 = hier_cost_batch(A, A)
    assert ca2 == 0.0 and cn2 == 0.0, (ca2, cn2)
    # single-level (flat) consistency: leaf-only
    ca3, cn3 = hier_cost_batch([[[0,1],[2,3],[4,5]]], [[[0,1],[2,3,5],[4]]])
    assert abs(ca3 - 4.0) < 1e-9, ca3   # dirty leaves {2,3,5},{4} weight 3+1
    print("hier_cost selftest OK: cost=7/9, identity=0, flat=4")


if __name__ == "__main__":
    _selftest()
