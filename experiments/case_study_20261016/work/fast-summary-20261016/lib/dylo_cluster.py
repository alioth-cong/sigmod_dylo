# Path-portable copy; original and published SHA-256 in provenance.json.
"""cluster_graph 的 DyLo 替身。

GraphRAG 的 cluster_graph(graph, max_cluster_size, use_lcc, seed) 返回
  [(level, cluster_id, parent_cluster_id, [节点名, ...]), ...]
其中 level 0 最粗、层号越大越细（graspologic hierarchical_leiden 的约定）。

DyLo 的驱动每批发射 com_<b>_L0/L1/... ，L0 最细、层号越大越粗 —— 方向相反，
这里必须反转，否则社区报告的层级全错。

DyLo 是跨批增量的，而 cluster_graph 是无状态的；因此分区序列由 dylo_stream.py
离线一次性跑出，这里只按批次号查表并转换格式。批次号从环境变量 DYLO_BATCH 读。

顶点集对齐：DyLo 在全图上跑（它的流是只增的），而 GraphRAG 在 use_lcc=True 时
只聚类最大连通分量。转换时按传入 graph 的节点集过滤，空社区丢弃，于是两臂覆盖
完全相同的顶点。

层次语义对齐（DYLO_HIER_MODE，默认 ragged）
--------------------------------------------
两边虽然都用 Louvain/SLM/Leiden 家族，造层次的方式却不同，直接注入会让口径失配：

  graspologic hierarchical_leiden —— **自顶向下递归切分 + 尺寸截止**。社区一旦
    小于 max_cluster_size 就成为叶子、不再下沉，所以深层只装着仍然超标的分支的
    后代，是棵参差不齐的树。实测 base 索引：level0 65 个覆盖 58.9%、level2 862 个
    覆盖 46.1%、level4 只剩 6 个覆盖 0.4%，最大社区 531。

  DyLo 驱动 —— **自底向上聚合**的经典树状图，每一层都是全图的完整划分。实测
    注入后：6 层每层都覆盖同样的 7,562 个实体，社区数 20→1165 单调增，
    level0 有个 2,757 个实体的巨型社区。

后果有三：(1) max_cluster_size 对 DyLo 完全失效，巨型社区的上下文塞不进
max_input_length=4000 的窗口，报告只能空泛；(2) community_level=1 时可选社区数
536 对 41，差 13 倍，检索层候选太少会直接答不出来；(3) DyLo 每层重复覆盖同一批
实体，报告数与 token 成本的口径跟 base 不可直接比。

所以默认改为 ragged：拿 DyLo 的树状图按 graspologic 的规则重新裁剪 —— 从最粗层
出发，社区不超过 max_cluster_size 即为叶子、不再展开；超标才把它在下一细层里的
孩子发射到 level+1。这样两边的层次语义一致，DyLo 自身的划分结果一个都没改。
设 DYLO_HIER_MODE=full 可退回旧行为（完整树状图，用于复现早先的结果）。
"""

def _artifact_path(relative):
    from pathlib import Path as _P
    import os as _O
    root = next((p for p in _P(__file__).resolve().parents if (p / 'ARTIFACT_SNAPSHOT.json').exists()))
    return str(root / relative)

def _artifact_env(name, default):
    import os as _O
    return _O.environ.get(name, default)
import hashlib
import json
import os
from pathlib import Path
import networkx as nx
from graphrag.index.utils.stable_lcc import stable_largest_connected_component
Communities = list[tuple[int, int, int, list[str]]]
DYLO_DIR = Path(os.environ.get('DYLO_DIR', _artifact_path('experiments/case_study_20261016/work/fast-summary-20261016/lib/dylo_arm')))

def _read_levels(batch: int) -> list[list[list[int]]]:
    """返回 [L0, L1, ...]，每层是社区列表，社区是叶子顶点 id 列表。"""
    lv = []
    i = 0
    while True:
        p = DYLO_DIR / f'com_{batch}_L{i}.txt'
        if not p.exists():
            break
        cur = []
        for line in p.open():
            row = [int(x) for x in line.split() if x.lstrip('-').isdigit()]
            if row:
                cur.append(row)
        lv.append(cur)
        i += 1
    if lv:
        return lv
    flat = DYLO_DIR / f'com_{batch}.txt'
    if flat.exists():
        cur = []
        for line in flat.open():
            row = [int(x) for x in line.split() if x.lstrip('-').isdigit()]
            if row:
                cur.append(row)
        if cur:
            return [cur]
    msg = f'分区缺失: {DYLO_DIR}/com_{batch}_L0.txt 与 com_{batch}.txt 均不存在'
    raise FileNotFoundError(msg)

def _load_names() -> dict[int, str]:
    m = json.loads((DYLO_DIR / 'id2name.json').read_text())
    return {int(k): v for (k, v) in m.items()}

def dylo_cluster_graph(graph: nx.Graph, max_cluster_size: int, use_lcc: bool, seed: int | None=None) -> Communities:
    """按 GraphRAG 的 Communities 契约返回 DyLo 的分层划分。"""
    if len(graph.nodes) == 0:
        return []
    if use_lcc:
        graph = stable_largest_connected_component(graph)
    keep = set(graph.nodes)
    batch = int(os.environ['DYLO_BATCH'])
    levels = _read_levels(batch)
    id2name = _load_names()
    ordered = list(reversed(levels))
    dedup: list[list[list[int]]] = []
    for lvl in ordered:
        sig = sorted((tuple(sorted(c)) for c in lvl))
        if dedup and sig == sorted((tuple(sorted(c)) for c in dedup[-1])):
            continue
        dedup.append(lvl)

    def stable_id(level: int, names: list[str]) -> int:
        h = hashlib.blake2b((str(level) + '|' + '|'.join(sorted(names))).encode(), digest_size=6)
        return int.from_bytes(h.digest(), 'big')

    def to_names(members: list[int]) -> list[str]:
        return [id2name[v] for v in members if v in id2name and id2name[v] in keep]
    mode = os.environ.get('DYLO_HIER_MODE', 'ragged').lower()
    if len(dedup) == 1 and os.environ.get('DYLO_FLAT_SUBDIVIDE', '1') != '0':
        return _emit_subdivided(dedup[0], graph, to_names, max_cluster_size, seed)
    if mode == 'full':
        return _emit_full(dedup, to_names, stable_id)
    return _emit_ragged(dedup, to_names, stable_id, max_cluster_size)

def _emit_subdivided(comms, graph, to_names, max_cluster_size, seed) -> Communities:
    """用 baseline 的划分做种子，交给 GraphRAG 自己的 hierarchical_leiden 递归细分。"""
    from graspologic.partition import hierarchical_leiden
    start: dict[str, int] = {}
    for (ci, members) in enumerate(comms):
        for nm in to_names(members):
            start[nm] = ci
    nxt = len(comms)
    for n in graph.nodes:
        if n not in start:
            start[n] = nxt
            nxt += 1
    dense = {c: i for (i, c) in enumerate(sorted(set(start.values())))}
    start = {n: dense[c] for (n, c) in start.items()}
    mapping = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, starting_communities=start, random_seed=seed)
    lvl: dict[int, dict[int, list[str]]] = {}
    parent: dict[int, int] = {}
    for p in mapping:
        lvl.setdefault(p.level, {}).setdefault(p.cluster, []).append(p.node)
        parent[p.cluster] = p.parent_cluster if p.parent_cluster is not None else -1
    out: Communities = []
    for level in sorted(lvl):
        for (cid, names) in lvl[level].items():
            if names:
                out.append((level, cid, parent.get(cid, -1), names))
    return out

def _emit_full(dedup, to_names, stable_id) -> Communities:
    """旧行为：每层都是全图的完整划分，逐层全部发射。"""
    results: Communities = []
    prev_owner: dict[int, int] = {}
    for (level, comms) in enumerate(dedup):
        owner: dict[int, int] = {}
        for members in comms:
            names = to_names(members)
            if not names:
                continue
            cid = stable_id(level, names)
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
    """按 graspologic 的规则裁剪 DyLo 的树状图。

    从最粗层往下走：一个社区若不超过 max_cluster_size 就是叶子，停止展开；
    超标才把它在下一细层里的孩子发射到 level+1。

    孩子的成员一律**与父亲取交集**再发射。DyLo 的层间并非严格嵌套（审计与回收会
    让顶点跨社区移动），不取交集的话同一个细层社区会同时挂到两个粗层社区下面 ——
    实测会产生 12 个重复的 (level, cid) 和 11 处"孩子不是父亲子集"的违规。
    取交集后每个顶点在每层只归属一个社区，树是严格的划分细化。
    """
    idx: list[dict[int, int]] = []
    for lvl in dedup:
        m: dict[int, int] = {}
        for (ci, members) in enumerate(lvl):
            for v in members:
                m[v] = ci
        idx.append(m)
    results: Communities = []
    emitted: set[tuple[int, int]] = set()
    queue: list[tuple[int, list[int], int, int]] = [(0, members, 0, -1) for members in dedup[0]]
    while queue:
        (dl, members, out_level, parent) = queue.pop()
        names = to_names(members)
        if not names:
            continue
        cid = stable_id(out_level, names)
        if (out_level, cid) in emitted:
            continue
        emitted.add((out_level, cid))
        results.append((out_level, cid, parent, names))
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
        nd = dl + 1
        kids = children_at(nd)
        while len(kids) <= 1 and nd + 1 < len(dedup):
            nd += 1
            kids = children_at(nd)
        if len(kids) <= 1 or (len(kids) == 1 and len(kids[0]) == len(mset)):
            continue
        for g in kids:
            queue.append((nd, g, out_level + 1, cid))
    return results

def install() -> None:
    """把替身装进 GraphRAG 的两个引用点（模块内 + create_communities 已导入的名字）。"""
    import graphrag.index.operations.cluster_graph as op
    import graphrag.index.workflows.create_communities as cc
    op.cluster_graph = dylo_cluster_graph
    cc.cluster_graph = dylo_cluster_graph
