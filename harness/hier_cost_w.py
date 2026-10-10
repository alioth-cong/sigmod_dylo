#!/usr/bin/env python3
"""From each batch's partitions, compute the re-summarization cost under two node-weight models, plus the number of summaries rewritten.

The dirty set is decided exactly as in hier_cost.py / dirty_cost.py (best Jaccard < 1 at the same level means self-dirty,
then propagated up the parent chain; flat methods have only the leaf level). Only the node weight changes:
  old  leaf = number of vertices, internal = number of children (the paper's Definition 2.1)
  b1   every node = min(CAP, A + B*|subtree entities|) (input tokens of one summary call,
       measured in rerun/results/proxy_v2; scale-free after normalization)
Each batch's cost is divided by that batch's total tree weight ("full rebuild = 1") and averaged over the stream.
calls = number of dirty nodes in the batch (i.e. summaries rewritten), calls_norm = calls / number of nodes in the batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dirty_cost import read_flat_lists  # noqa: E402
from hier_cost import _parents, _self_dirty, read_levels  # noqa: E402

B1 = dict(A=2026.0, B=89.54, CAP=6239.0)


def b1w(n: int) -> float:
    return min(B1["CAP"], B1["A"] + B1["B"] * n)


def read_any(out_dir: Path, k: int, hier: bool):
    if hier:
        lv = read_levels(out_dir, k)
        if lv:
            return lv
    f = out_dir / f"com_{k}.txt"
    return [read_flat_lists(f)] if f.exists() else []


def batch_costs(prev, curr) -> dict:
    nlev = len(curr)
    sd = [_self_dirty(prev[i] if (prev and i < len(prev)) else [], curr[i]) for i in range(nlev)]
    parents = _parents(curr)
    nchildren = [[0] * len(curr[i]) for i in range(nlev)]
    for i in range(nlev - 1):
        for c, par in enumerate(parents[i]):
            if par >= 0:
                nchildren[i + 1][par] += 1
    dirty = [list(x) for x in sd]
    for i in range(nlev - 1):
        for c in range(len(curr[i])):
            if dirty[i][c] and parents[i][c] >= 0:
                dirty[i + 1][parents[i][c]] = True
    r = dict(cost_old=0.0, tot_old=0.0, cost_b1=0.0, tot_b1=0.0, calls=0, nodes=0)
    for i in range(nlev):
        for c in range(len(curr[i])):
            size = len(curr[i][c])            # every level is a vertex list: subtree entity count
            wo = size if i == 0 else nchildren[i][c]
            wb = b1w(size)
            r["tot_old"] += wo
            r["tot_b1"] += wb
            r["nodes"] += 1
            if dirty[i][c]:
                r["cost_old"] += wo
                r["cost_b1"] += wb
                r["calls"] += 1
    return r


def stream(out_dir: Path, n_batches: int, hier: bool) -> list[dict]:
    rows, prev = [], None
    for k in range(0, n_batches + 1):
        curr = read_any(out_dir, k, hier)
        if not curr:
            break
        if prev is not None:
            r = batch_costs(prev, curr)
            r["k"] = k
            r["norm_old"] = r["cost_old"] / r["tot_old"] if r["tot_old"] else 0.0
            r["norm_b1"] = r["cost_b1"] / r["tot_b1"] if r["tot_b1"] else 0.0
            r["calls_norm"] = r["calls"] / r["nodes"] if r["nodes"] else 0.0
            rows.append(r)
        prev = curr
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    return dict(batches=n,
                cost_old=sum(r["norm_old"] for r in rows) / n,
                cost_b1=sum(r["norm_b1"] for r in rows) / n,
                calls=sum(r["calls"] for r in rows) / n,
                calls_norm=sum(r["calls_norm"] for r in rows) / n,
                nodes=sum(r["nodes"] for r in rows) / n)


if __name__ == "__main__":
    out, nb, hier = Path(sys.argv[1]), int(sys.argv[2]), (sys.argv[3] == "hier")
    print(summarize(stream(out, nb, hier)))
