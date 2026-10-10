#!/usr/bin/env python3
"""
Re-summarization cost across a stream of partitions.

Per-batch cost (relative to previous batch):
  - Match each new community to an old one by best Jaccard overlap of member sets.
  - "Same community" if overlap >= 0.5.
  - A community is "dirty" if (a) its member set changed at all (matched overlap < 1.0)
    or (b) it is new/split (no >=0.5 match).
  - For flat (single-level) partitions, the dirty set is just the dirty leaves.
  - For hierarchical partitions, the dirty set propagates to all ancestors.
  - weight(c) = |members|  for a leaf
                |children| for an internal node.
  - cost = sum of weight(c) over dirty set.
  - We also output cost normalized to "from-scratch" — from-scratch relabels everything,
    so its dirty set is the entire tree (worst case = 1.0 after normalization).

Community file format (dynamo native, also produced by our SLM/Leiden adapters):
  one community per line, tab-separated node IDs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# ---------- io ----------

def read_flat_partition(path: Path) -> list[set[int]]:
    """Read a flat partition file. Returns list of member-id sets."""
    comms: list[set[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            members = {int(x) for x in line.split() if x}
            if members:
                comms.append(members)
    return comms


def read_flat_lists(path: Path) -> list[list[int]]:
    """Read a flat partition file as a list of member-id lists.

    Partition lines hold disjoint, unique node IDs, so we skip the per-line set
    construction (a big win on multi-million-node graphs); membership/overlap is
    recovered via an inverted node->community index instead of set intersection.
    """
    comms: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            members = [int(x) for x in line.split()]
            if members:
                comms.append(members)
    return comms


def build_index(comms: list[list[int]]) -> dict[int, int]:
    """Inverted index: node id -> its community index in `comms`."""
    idx: dict[int, int] = {}
    for ci, c in enumerate(comms):
        for n in c:
            idx[n] = ci
    return idx


def score_against_index(prev_sizes: list[int], prev_idx: dict[int, int],
                        curr: list[list[int]]) -> tuple[float, float]:
    """One-batch dirty-set cost using a prebuilt prev index. O(sum|curr|).

    For each curr community c, best-Jaccard to prev is found by counting member
    overlaps per prev community (only those c actually touches), then
    jaccard = overlap / (|c| + |prev| - overlap). c is dirty iff best < 1.0
    (membership changed) — which also covers new/split (best stays 0.0). This is
    exactly the semantics of the O(C^2) matcher, computed in linear time.
    """
    cost = 0.0
    total = 0
    for c in curr:
        lc = len(c)
        total += lc
        if not prev_sizes:
            cost += lc
            continue
        cnt: dict[int, int] = {}
        g = cnt.get
        for n in c:
            pi = prev_idx.get(n)
            if pi is not None:
                cnt[pi] = g(pi, 0) + 1
        best = 0.0
        for pi, o in cnt.items():
            j = o / (lc + prev_sizes[pi] - o)
            if j > best:
                best = j
                if best >= 1.0:
                    break
        if best < 1.0:
            cost += lc
    return cost, (cost / total if total > 0 else 0.0)


# ---------- metric core ----------

def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b: return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    if union == 0: return 0.0
    return inter / union


def match_and_score_flat(prev: list[set[int]], curr: list[set[int]]) -> tuple[float, float, dict]:
    """Compute one-batch dirty-set cost for FLAT partitions.

    Strategy: for each new (curr) community c, find the prev community with maximum Jaccard.
    Mark c "dirty" iff best overlap < 1.0 (membership changed) or < 0.5 (new/split).
    NOTE: for flat partitions there's no hierarchy, so weight(c) = |members| of c.
          The dirty SET is *new* communities that are dirty; we do NOT count old communities
          that vanished, since the user defined dirty in terms of new->old matching.

    For from-scratch normalization: total weight of the new partition = sum |c| = total #nodes.
    cost_norm = cost / total_weight.

    Returns: (cost_abs, cost_norm, debug)
    """
    if not curr:
        return 0.0, 0.0, {}
    prev_lists = [list(p) for p in prev]
    prev_sizes = [len(p) for p in prev_lists]
    prev_idx = build_index(prev_lists)
    curr_lists = [list(c) for c in curr]
    ca, cn = score_against_index(prev_sizes, prev_idx, curr_lists)
    return ca, cn, {}


# ---------- streaming over a method's output directory ----------

def stream_cost(out_dir: Path, n_batches: int) -> list[tuple[int, float, float]]:
    """Walk com_0.txt..com_<n_batches>.txt; return [(batch, cost_abs, cost_norm)] for batches 1..n_batches."""
    rows: list[tuple[int, float, float]] = []
    prev_idx: dict[int, int] | None = None
    prev_sizes: list[int] | None = None
    for k in range(0, n_batches + 1):
        cf = out_dir / f"com_{k}.txt"
        if not cf.exists():
            sys.stderr.write(f"[dirty_cost] missing {cf}\n")
            break
        curr = read_flat_lists(cf)
        if prev_idx is not None:
            ca, cn = score_against_index(prev_sizes, prev_idx, curr)
            rows.append((k, ca, cn))
        # the current partition's index/sizes become "prev" for the next batch
        prev_sizes = [len(c) for c in curr]
        prev_idx = build_index(curr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-batches", type=int, required=True)
    ap.add_argument("--write-tsv", default=None)
    args = ap.parse_args()

    rows = stream_cost(Path(args.out_dir), args.n_batches)
    if args.write_tsv:
        with open(args.write_tsv, "w") as g:
            g.write("batch\tcost_abs\tcost_norm\n")
            for r in rows:
                g.write(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}\n")
    avg_abs = sum(r[1] for r in rows) / max(1, len(rows))
    avg_norm = sum(r[2] for r in rows) / max(1, len(rows))
    print(json.dumps({"batches": len(rows), "avg_cost_abs": avg_abs, "avg_cost_norm": avg_norm}))


if __name__ == "__main__":
    main()
