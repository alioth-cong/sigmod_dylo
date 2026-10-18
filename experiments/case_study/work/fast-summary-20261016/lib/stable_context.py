"""Preserve the existing index's record order; append new records deterministically.

GraphRAG selects the first incident relationship before ranking the context. A
groupby during merge must not silently select different old evidence. This is
an input ordering policy, not a relaxation of summary freshness checks.
"""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
from functools import lru_cache
from pathlib import Path
import pandas as pd
POLICY = 'base-anchored-order-strict-input-v1'

@lru_cache(maxsize=8)
def anchor(root, name):
    keys = ['title'] if name == 'entities' else ['source', 'target']
    table = pd.read_parquet(Path(root) / 'data/base' / f'{name}.parquet', columns=['id', *keys])
    assert table.id.is_unique
    return {str(r.id): (i, tuple((str(getattr(r, k)) for k in keys))) for (i, r) in enumerate(table.itertuples())}

def stable_rows(table, root, name):
    """Only reorder rows. All descriptions, IDs, degrees and new facts survive."""
    assert table.id.is_unique, (name, 'duplicate record ids')
    old = anchor(str(Path(root).resolve()), name)
    keys = ['title'] if name == 'entities' else ['source', 'target']
    order = []
    for row in table.itertuples():
        identity = str(row.id)
        semantic_key = tuple((str(getattr(row, k)) for k in keys))
        if identity in old:
            (position, expected) = old[identity]
            assert semantic_key == expected, (name, identity, 'old record identity changed')
            order.append((0, position))
        else:
            order.append((1, *semantic_key, int(row.human_readable_id), identity))
    positions = sorted(range(len(table)), key=order.__getitem__)
    return table.iloc[positions].reset_index(drop=True)

def stable_tables(entities, relationships, root):
    return (stable_rows(entities, root, 'entities'), stable_rows(relationships, root, 'relationships'))
