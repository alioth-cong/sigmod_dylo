#!/usr/bin/env python
"""Extract DyLo's edge stream from the official arm's per-batch snapshots.

Both arms share exactly the same entity/relationship extraction, so DyLo's input comes straight from the official snapshots:
  batch_00 relationships -> initial.txt
  relationships new in batch_kk relative to batch_(kk-1) -> inc_k.txt
Vertex ids are assigned in order of first appearance; id2name.json is written for dylo_cluster.py to map back.

Edge weights use GraphRAG's own relationships.weight (create_graph builds the graph with it too),
so DyLo optimizes the same weighted graph that the official Leiden sees.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import json
import sys
from pathlib import Path

import pandas as pd

R = Path(DYLO_ROOT + "/graphrag")
SNAP = R / "snapshots_official"
OUT = R / "dylo_arm"
NB = 20


def rel_set(b: int) -> pd.DataFrame:
    p = SNAP / f"batch_{b:02d}" / "relationships.parquet"
    df = pd.read_parquet(p, columns=["source", "target", "weight"])
    df["key"] = df.apply(
        lambda r: (r["source"], r["target"]) if r["source"] <= r["target"]
        else (r["target"], r["source"]), axis=1)
    return df.drop_duplicates("key")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ids: dict[str, int] = {}

    def vid(name: str) -> int:
        if name not in ids:
            ids[name] = len(ids)
        return ids[name]

    base = rel_set(0)
    seen = set()
    with (OUT / "initial.txt").open("w") as f:
        for _, r in base.iterrows():
            a, b = vid(r["source"]), vid(r["target"])
            if a == b:
                continue          # self-loop: meaningless for Leiden's modularity, and baseline implementations crash on it
            seen.add(r["key"])
            f.write(f"{a} {b} {float(r['weight'])}\n")
    print(f"initial: {len(seen)} edges, {len(ids)} vertices", flush=True)

    for k in range(1, NB + 1):
        cur = rel_set(k)
        n = 0
        with (OUT / f"inc_{k}.txt").open("w") as f:
            for _, r in cur.iterrows():
                if r["key"] in seen:
                    continue
                a, b = vid(r["source"]), vid(r["target"])
                if a == b:
                    continue
                seen.add(r["key"])
                f.write(f"{a} {b} {float(r['weight'])}\n")
                n += 1
        print(f"inc_{k}: {n} new edges, {len(ids)} vertices so far", flush=True)

    (OUT / "id2name.json").write_text(
        json.dumps({str(v): k for k, v in ids.items()}, ensure_ascii=False))
    # the driver finds files by <inc-base> plus batch number, so give it a base name
    (OUT / "inc.txt").write_text("")
    print(f"done: {len(ids)} vertices, {len(seen)} edges -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
