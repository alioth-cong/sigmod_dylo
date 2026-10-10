#!/usr/bin/env python
"""DyLo stand-in for cluster_graph.

GraphRAG's cluster_graph(graph, max_cluster_size, use_lcc, seed) returns
  [(level, cluster_id, parent_cluster_id, [node names, ...]), ...]
where level 0 is the coarsest and larger levels are finer (graspologic hierarchical_leiden's convention).

DyLo's driver emits com_<b>_L0/L1/... per batch with L0 the finest and larger levels coarser -- the opposite direction,
so the levels must be reversed here, otherwise every community report gets the wrong level.

DyLo is incremental across batches while cluster_graph is stateless, so the partition sequence is produced
offline once by dylo_stream.py; here we only look up the batch and convert the format. The batch comes from DYLO_BATCH.

Vertex-set alignment: DyLo runs on the full graph (its stream only grows), while GraphRAG with use_lcc=True
clusters only the largest connected component. Conversion filters by the nodes of the given graph and drops empty communities, so both arms cover
exactly the same vertices.

Hierarchy semantics alignment (DYLO_HIER_MODE, default ragged)
--------------------------------------------
Both sides use the Louvain/SLM/Leiden family but build hierarchies differently, so injecting directly mismatches the accounting:

  graspologic hierarchical_leiden -- **top-down recursive splitting with a size cutoff**. Once a community
    is below max_cluster_size it becomes a leaf and is not refined, so deep levels only hold descendants of branches
    still over the limit: a ragged tree. Measured on the base index: level0 has 65 communities covering 58.9%, level2 862
    covering 46.1%, level4 only 6 covering 0.4%; the largest community has 531 members.

  DyLo driver -- a classic **bottom-up aggregated** dendrogram in which every level is a full partition of the graph. Measured
    after injection: all 6 levels cover the same 7,562 entities, the community count rises monotonically 20 -> 1165,
    and level0 has a giant community of 2,757 entities.

Three consequences: (1) max_cluster_size has no effect on DyLo, and the giant community's context cannot fit
the max_input_length=4000 window, so its report is vague; (2) at community_level=1 the selectable communities are
536 vs 41, 13x fewer, and too few retrieval candidates leave questions unanswered; (3) DyLo covers the same entities at every level,
so its report count and token cost are not comparable with base.

The default is therefore ragged: DyLo's dendrogram is re-cut by graspologic's rule -- start from the coarsest level;
a community within max_cluster_size is a leaf and is not expanded; only an oversized one emits its
children from the next finer level at level+1. Both sides then share hierarchy semantics, and none of DyLo's partitions change.
Set DYLO_HIER_MODE=full to restore the old behavior (full dendrogram, to reproduce earlier results).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import hashlib
import json
import os
from pathlib import Path

import networkx as nx

from graphrag.index.utils.stable_lcc import stable_largest_connected_component

Communities = list[tuple[int, int, int, list[str]]]

DYLO_DIR = Path(os.environ.get("DYLO_DIR", DYLO_ROOT + "/graphrag/dylo_arm"))


def _read_levels(batch: int) -> list[list[list[int]]]:
    """Return [L0, L1, ...]; each level is a list of communities, each a list of leaf vertex ids."""
    lv = []
    i = 0
    while True:
        p = DYLO_DIR / f"com_{batch}_L{i}.txt"
        if not p.exists():
            break
        cur = []
        for line in p.open():
            row = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
            if row:
                cur.append(row)
        lv.append(cur)
        i += 1
    if lv:
        return lv

    # Flat baselines write a single-level com_<b>.txt. DynaMo/QCA/GreMod/DF-Louvain/Delta-screen
    # maintain no hierarchy; HIT-Leiden has P levels internally but its driver's write_com stores only level-0
    # labels. We inject the single level as is, so GraphRAG sees only level 0 -- the real shape of these methods
    # when used as a GraphRAG index; we do not fabricate a hierarchy for them.
    flat = DYLO_DIR / f"com_{batch}.txt"
    if flat.exists():
        cur = []
        for line in flat.open():
            row = [int(x) for x in line.split() if x.lstrip("-").isdigit()]
            if row:
                cur.append(row)
        if cur:
            return [cur]

    msg = f"partition missing: neither {DYLO_DIR}/com_{batch}_L0.txt nor com_{batch}.txt exists"
    raise FileNotFoundError(msg)


def _load_names() -> dict[int, str]:
    m = json.loads((DYLO_DIR / "id2name.json").read_text())
    return {int(k): v for k, v in m.items()}


def dylo_cluster_graph(
    graph: nx.Graph,
    max_cluster_size: int,
    use_lcc: bool,
    seed: int | None = None,  # noqa: ARG001
) -> Communities:
    """Return DyLo's hierarchical partition in GraphRAG's Communities contract."""
    if len(graph.nodes) == 0:
        return []
    if use_lcc:
        graph = stable_largest_connected_component(graph)
    keep = set(graph.nodes)

    batch = int(os.environ["DYLO_BATCH"])
    levels = _read_levels(batch)
    id2name = _load_names()

    # reverse the level order: DyLo's last (coarsest) level becomes GraphRAG's level 0
    ordered = list(reversed(levels))

    # drop trailing duplicate levels (after DyLo converges, L1==L2 can repeat)
    dedup: list[list[list[int]]] = []
    for lvl in ordered:
        sig = sorted(tuple(sorted(c)) for c in lvl)
        if dedup and sig == sorted(tuple(sorted(c)) for c in dedup[-1]):
            continue
        dedup.append(lvl)

    # Community ids must be determined by **member-set content**, not by enumeration order.
    # Downstream decides "rewrite the summary or not" by member set; if ids drift with enumeration order,
    # structurally unchanged communities are flagged as changed. We hit this: the freeze arm's driver had zero changes on all eight levels,
    # yet GraphRAG reported 1,377 reports to rewrite (1,119 on level 5), all artifacts of id drift.
    # digest_size must be <= 6: GraphRAG's read_indexer_entities has a
    #   nodes_df["community"] = nodes_df["community"].fillna(-1)   # -> float64
    #   ... [str(int(i)) for i in x]
    # fillna that turns the integer column into float64, which has only a 53-bit mantissa. With 7 bytes (56 bits)
    # 982 of 1455 community ids (67.5%) came back as different integers after the round trip, entities'
    # community_ids then matched no report, every occurrence weight was 0, and retrieval's
    # _compute_community_weights hit max_weight=0 -> ZeroDivisionError --
    # which is how all 60 questions of the norecl arm failed. The rebuild arm's ids are small integers 0..1519 and unaffected.
    # 6 bytes = 48 bits < 2^53 round-trips exactly; with thousands of communities the collision probability is ~1e-9.
    def stable_id(level: int, names: list[str]) -> int:
        h = hashlib.blake2b(
            (str(level) + "|" + "|".join(sorted(names))).encode(),
            digest_size=6,
        )
        return int.from_bytes(h.digest(), "big")

    def to_names(members: list[int]) -> list[str]:
        return [id2name[v] for v in members if v in id2name and id2name[v] in keep]

    mode = os.environ.get("DYLO_HIER_MODE", "ragged").lower()

    # Flat (single-level) baselines go through GraphRAG's own recursive refinement rather than raw injection.
    #
    # GraphRAG does not use bare Leiden but hierarchical_leiden: after one Leiden run,
    # every community above max_cluster_size is split further on its subgraph until all are below the threshold.
    # Injecting a baseline's raw partition would compare "one modularity optimization" against "Leiden + recursive refinement" --
    # measured: DF-Louvain cuts the same 7,883-node LCC into only 56 communities (median 118,
    # max 849), while Leiden yields 1,518 and DyLo 1,465, a 26x granularity gap,
    # leaving an order of magnitude fewer usable reports. That is an artifact of the wiring, not of the method.
    #
    # Here the baseline's partition is passed as starting_communities to GraphRAG's own
    # hierarchical_leiden: the incremental method maintains the top-level partition, refinement follows GraphRAG's native logic,
    # fully symmetric with the Leiden arm.
    if len(dedup) == 1 and os.environ.get("DYLO_FLAT_SUBDIVIDE", "1") != "0":
        return _emit_subdivided(dedup[0], graph, to_names, max_cluster_size, seed)

    if mode == "full":
        return _emit_full(dedup, to_names, stable_id)
    return _emit_ragged(dedup, to_names, stable_id, max_cluster_size)


def _emit_subdivided(comms, graph, to_names, max_cluster_size, seed) -> Communities:
    """Seed GraphRAG's own hierarchical_leiden recursive refinement with the baseline's partition."""
    from graspologic.partition import hierarchical_leiden

    start: dict[str, int] = {}
    for ci, members in enumerate(comms):
        for nm in to_names(members):
            start[nm] = ci
    # nodes not covered by the baseline form singleton classes, so graspologic does not fail on missing keys
    nxt = len(comms)
    for n in graph.nodes:
        if n not in start:
            start[n] = nxt
            nxt += 1

    # graspologic's native hierarchical_leiden requires starting_communities labels
    # to be contiguous 0..k-1 -- it indexes internal arrays by label. to_names drops nodes not in keep/id2name,
    # so some ci lose all members and the label sequence has holes, while nxt still numbers from
    # len(comms); the largest label then exceeds the number of labels in use and the native layer panics
    # ("index out of bounds: the len is 615 but the index is 616"). Labels are compacted here.
    dense = {c: i for i, c in enumerate(sorted(set(start.values())))}
    start = {n: dense[c] for n, c in start.items()}

    mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size,
        starting_communities=start, random_seed=seed,
    )
    lvl: dict[int, dict[int, list[str]]] = {}
    parent: dict[int, int] = {}
    for p in mapping:
        lvl.setdefault(p.level, {}).setdefault(p.cluster, []).append(p.node)
        parent[p.cluster] = p.parent_cluster if p.parent_cluster is not None else -1

    out: Communities = []
    for level in sorted(lvl):
        for cid, names in lvl[level].items():
            if names:
                out.append((level, cid, parent.get(cid, -1), names))
    return out


def _emit_full(dedup, to_names, stable_id) -> Communities:
    """Old behavior: every level is a full partition of the graph, and all levels are emitted."""
    results: Communities = []
    prev_owner: dict[int, int] = {}  # leaf vertex -> cluster_id on the previous level
    for level, comms in enumerate(dedup):
        owner: dict[int, int] = {}
        for members in comms:
            names = to_names(members)
            if not names:
                continue
            cid = stable_id(level, names)
            # parent = the cluster on the previous level that contains any member of this community; -1 at the top
            parent = -1
            if level > 0:
                for v in members:
                    if v in prev_owner:
                        parent = prev_owner[v]
                        break
            results.append((level, cid, parent, names))
            for v in members:
                owner[v] = cid
        prev_owner = owner
    return results


def _emit_ragged(dedup, to_names, stable_id, max_cluster_size: int) -> Communities:
    """Re-cut DyLo's dendrogram by graspologic's rule.

    Walk down from the coarsest level: a community within max_cluster_size is a leaf and is not expanded;
    only an oversized one emits its children from the next finer level at level+1.

    Children's members are always **intersected with the parent** before emission. DyLo's levels are not strictly nested (audits and reclamation
    move vertices across communities); without the intersection the same finer community would hang under two coarser ones --
    measured: 12 duplicate (level, cid) pairs and 11 "child not a subset of parent" violations.
    With the intersection every vertex belongs to one community per level and the tree is a strict refinement.
    """
    # per level, build an index vertex -> community position on that level, used to find children
    idx: list[dict[int, int]] = []
    for lvl in dedup:
        m: dict[int, int] = {}
        for ci, members in enumerate(lvl):
            for v in members:
                m[v] = ci
        idx.append(m)

    results: Communities = []
    emitted: set[tuple[int, int]] = set()
    # expansion queue: (dedup level, members of this community, GraphRAG level to emit at, parent cluster_id)
    queue: list[tuple[int, list[int], int, int]] = [
        (0, members, 0, -1) for members in dedup[0]
    ]
    while queue:
        dl, members, out_level, parent = queue.pop()
        names = to_names(members)
        if not names:
            continue
        cid = stable_id(out_level, names)
        if (out_level, cid) in emitted:
            continue
        emitted.add((out_level, cid))
        results.append((out_level, cid, parent, names))

        # already small enough, or no finer level to descend into -> leaf
        if len(names) <= max_cluster_size or dl + 1 >= len(dedup):
            continue

        mset = set(members)

        def children_at(nd: int) -> list[list[int]]:
            groups: dict[int, list[int]] = {}
            for v in members:
                ci = idx[nd].get(v)
                if ci is not None:
                    groups.setdefault(ci, []).append(v)
            return [g for g in groups.values() if g]

        # search finer levels until it actually splits (some adjacent levels are the same partition)
        nd = dl + 1
        kids = children_at(nd)
        while len(kids) <= 1 and nd + 1 < len(dedup):
            nd += 1
            kids = children_at(nd)
        if len(kids) <= 1 or (len(kids) == 1 and len(kids[0]) == len(mset)):
            continue          # it cannot be split further down, so treat it as a leaf
        for g in kids:
            queue.append((nd, g, out_level + 1, cid))
    return results


def install() -> None:
    """Install the stand-in at both GraphRAG reference points (the module and the name already imported by create_communities)."""
    import graphrag.index.operations.cluster_graph as op
    import graphrag.index.workflows.create_communities as cc

    op.cluster_graph = dylo_cluster_graph
    cc.cluster_graph = dylo_cluster_graph
