#!/usr/bin/env python
"""Export the leaf partition of GraphRAG's initial index as a warm start for the DyLo driver.

Why: the DyLo driver otherwise always cold-starts its own clustering, which has nothing to do with GraphRAG's partition
(measured 783 communities vs 6,564); inheriting summaries by member set then almost never hits, and the measured "re-summarization cost"
would be an artifact of the wiring rather than a property of the algorithm. With the leaf level fed in, both arms share batch 0, and later differences
come purely from the incremental maintenance policy.

Leaf level: each entity belongs to the **deepest** community containing it. GraphRAG's hierarchy recursively splits oversized
communities (max_cluster_size), so the deepest one is its actual finest granularity.

Usage: export_leaf_part.py <communities.parquet> <entities.parquet> <id2name.json> <output>
"""
import collections
import json
import sys

import pandas as pd


def main(comm_pq: str, ent_pq: str, id2name_json: str, out: str) -> None:
    c = pd.read_parquet(comm_pq)
    e = pd.read_parquet(ent_pq, columns=["id", "title"])
    eid2title = dict(zip(e["id"], e["title"]))
    name2vid = {v: int(k) for k, v in json.load(open(id2name_json)).items()}

    deep: dict[str, tuple[int, int]] = {}
    for lv, cm, ents in zip(c["level"], c["community"], c["entity_ids"]):
        if ents is None:
            continue
        for x in ents:
            cur = deep.get(x)
            if cur is None or lv > cur[0]:
                deep[x] = (lv, cm)

    part = collections.defaultdict(set)
    miss = 0
    for x, (_lv, cm) in deep.items():
        v = name2vid.get(eid2title.get(x))
        if v is None:
            miss += 1
            continue
        part[cm].add(v)

    n = 0
    with open(out, "w") as f:
        for vs in part.values():
            if vs:
                f.write("\t".join(map(str, sorted(vs))) + "\n")
                n += 1
    print(f"leaf partition -> {out}: {n} communities, {sum(len(v) for v in part.values())} vertices, "
          f"unmapped {miss}", flush=True)
    if miss:
        print(f"  warning: {miss} entities have no vertex in the edge stream (usually isolated entities without relationships)")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
