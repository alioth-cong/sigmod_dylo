#!/usr/bin/env python3
"""
Re-summarization cost with an EDGE-SENSITIVE dirty rule.

The shipped rule (dirty_cost.py) marks a community dirty only when its MEMBER
SET changes: best-Jaccard against the previous batch < 1.0. A community whose
membership is unchanged but which gained internal edges therefore costs zero.
Measured on Facebook batch 1, 18 of 20 communities fall in exactly that case,
so almost the whole index is free under that rule.

That is a real modelling gap. A community summary describes the relationships
among its members, not just the roster: adding a relationship between two
entities already in C changes what C's summary should say, so a pipeline that
never regenerates C is serving a stale summary. This module therefore adds the
second condition:

    dirty(C)  <=>  members(C) changed  OR  the internal edge weight of C changed

Everything else is identical to dirty_cost.py -- same weights (|members| for a
leaf), same normalisation, same matching -- so the two can be compared directly
and the delta is attributable solely to the edge-sensitivity.

The rule is applied uniformly to every method, so it neither favours nor
penalises any of them; it raises the absolute floor for all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dirty_cost import read_flat_lists, build_index  # noqa

# Cost charged for a structure-preserving refresh (members unchanged, internal
# edges added). One unit, not |members|: the existing summary stays correct and
# is extended with one more relation among entities it already covers, so
# pricing it as a full rewrite would overstate it by the community's size.
UNIT_REFRESH = 1.0


def load_batch_edges(run_dir: Path, k: int):
    """Edges ARRIVING in batch k (k<=0 -> none; the initial graph is the baseline).

    Only the batch delta is needed: a structure-preserving community has an
    unchanged member set, so its internal edge weight changed in batch k iff at
    least one edge of inc_k has both endpoints inside it. Re-reading the whole
    accumulated edge set per batch would be quadratic in the stream length and
    gives the same answer for exactly the communities this term charges.
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


def gained_internal(comms, batch_edges, node2comm):
    """gained[c] = True iff a batch edge landed strictly inside community c."""
    hit = [False] * len(comms)
    for a, b in batch_edges:
        ca = node2comm.get(a)
        if ca is not None and ca == node2comm.get(b):
            hit[ca] = True
    return hit


def stream_cost_edges(out_dir: Path, run_dir: Path, n_batches: int):
    """[(batch, cost_abs, cost_norm)] under the edge-sensitive rule."""
    rows = []
    prev = prev_idx = prev_sizes = None
    for k in range(0, n_batches + 1):
        cf = out_dir / f"com_{k}.txt"
        if not cf.exists():
            break
        curr = read_flat_lists(cf)
        c_idx = build_index(curr)
        gained = gained_internal(curr, load_batch_edges(run_dir, k), c_idx)
        if prev is not None:
            cost = 0.0
            total = 0
            for ci, c in enumerate(curr):
                lc = len(c)
                total += lc
                # best-Jaccard match on member sets (same as dirty_cost.py)
                cnt = {}
                for n in c:
                    pi = prev_idx.get(n)
                    if pi is not None:
                        cnt[pi] = cnt.get(pi, 0) + 1
                best, best_pi = 0.0, None
                for pi, o in cnt.items():
                    j = o / (lc + prev_sizes[pi] - o)
                    if j > best:
                        best, best_pi = j, pi
                if best < 1.0:
                    # structure-altering: the old summary describes a member set
                    # that no longer exists, so it must be regenerated in full.
                    cost += lc
                elif best_pi is not None and gained[ci]:
                    # structure-preserving: the old summary is still CORRECT,
                    # only incomplete -- one more relation among members it
                    # already covers. Charging |members| would price that as a
                    # full rewrite; the incremental refresh is a unit cost.
                    cost += UNIT_REFRESH
            rows.append((k, cost, cost / total if total else 0.0))
        prev, prev_idx = curr, c_idx
        prev_sizes = [len(c) for c in curr]
    return rows


if __name__ == "__main__":
    import argparse, statistics
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-batches", type=int, required=True)
    a = ap.parse_args()
    rows = stream_cost_edges(Path(a.out_dir), Path(a.run_dir), a.n_batches)
    print(f"batches={len(rows)} mean_cost_norm="
          f"{statistics.mean(r[2] for r in rows) if rows else float('nan'):.6f}")
