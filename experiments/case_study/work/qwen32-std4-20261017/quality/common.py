def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import json, hashlib, time
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parent
W = R.parents[2]
O = W / 'outputs/qwen32-std4-20261017/quality'
METHODS = ['dylo', 'batch', 'rebuild']
QWEN = str(W / _artifact_env('QWEN_MODEL', 'Qwen/Qwen2.5-32B-Instruct'))

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_suffix('.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    t.replace(path)

def read(p):
    return json.loads(p.read_text())

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def index(arm):
    return R.parent / 'runs' / arm / 'index'

def ready(arm):
    return (index(arm).parent / 'verified.json').exists() and (index(arm) / 'published.json').exists()

def cut(comms, level=0):
    """A covering antichain; primary global condition is C0 (root reports)."""
    rows = {int(r.community): r for r in comms.itertuples()}
    roots = [cid for (cid, r) in rows.items() if int(r.parent) == -1]
    stack = list(roots)
    chosen = []
    while stack:
        cid = stack.pop()
        r = rows[cid]
        children = list(map(int, r.children))
        if int(r.level) < level and children:
            stack.extend(children)
        else:
            chosen.append(cid)
    root_members = set().union(*(set(rows[c].entity_ids) for c in roots))
    covered = set()
    for cid in chosen:
        m = set(rows[cid].entity_ids)
        assert not m & covered, ('Overlapping cut', cid)
        covered |= m
    assert covered == root_members, 'Global level cut omitted covered entities'
    return (sorted(chosen), len(covered))

class Tokenizer:

    def __init__(self):
        from transformers import AutoTokenizer
        self.inner = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)

    def encode(self, s):
        return self.inner.encode(s, add_special_tokens=False)

    def decode(self, t):
        return self.inner.decode(t, skip_special_tokens=True)

    def num_tokens(self, s):
        return len(self.encode(s))

def layout(arm):
    path = index(arm)
    comms = pd.read_parquet(path / 'communities.parquet')
    reports = pd.read_parquet(path / 'community_reports.parquet')
    (ids, covered) = cut(comms)
    selected = reports[reports.community.isin(ids)].copy()
    assert set(selected.community) == set(ids) and selected.community.is_unique
    from graphrag.query.indexer_adapters import read_indexer_reports
    objects = read_indexer_reports(selected, comms[comms.community.isin(ids)], None, dynamic_community_selection=True)
    return (objects, ids, covered)
