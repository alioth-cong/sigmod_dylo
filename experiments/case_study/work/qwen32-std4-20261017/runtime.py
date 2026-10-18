"""Isolated32B model routing and accounting; old experiment sources are read-only."""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import os, sys, json, time, hashlib, importlib.util
from pathlib import Path
import httpx, pandas as pd
R = Path(__file__).resolve().parent
W = R.parents[1]
O = W / 'outputs/qwen32-std4-20261017'
F = W / 'work/fast-summary-20261016'
MODEL = 'Qwen/Qwen2.5-32B-Instruct'
sys.path.insert(0, str(F))
import quick_run as q
original_totals = q.totals
sys.path.insert(0, str(R))
from qwen32_model import ActualModel
q.f.b.ActualModel = ActualModel
spec = importlib.util.spec_from_file_location('q32_isolation', R / 'isolation.py')
isolation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(isolation)
Guard = isolation.Guard

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_name(path.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(value, indent=2, default=str))
    t.replace(path)

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(tag):
    return {n: pd.read_parquet(R / 'data' / tag / f'{n}.parquet') for n in ['documents', 'entities', 'relationships', 'text_units', 'communities', 'community_reports']}

def contexts(path):
    return [(int(level), pd.DataFrame(rows)) for (level, rows) in read(path)]

def records(path):
    if not Path(path).exists():
        return []
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def usage(rows):
    result = original_totals(rows)
    for (key, status) in [('successful', 'ok'), ('failed', 'error')]:
        result[key]['unknown_usage_calls'] = sum((r.get('status') == status and (not all((k in r for k in ['prompt_tokens', 'completion_tokens']))) for r in rows))
    return result

def assert_initial32(base):
    reports = base['community_reports']
    assert set(reports.summary_model) == {MODEL}, 'Mixed or old7B initial reports'
    assert reports.community.is_unique and set(reports.community) == set(base['communities'].community)
    assert (R / 'initial/complete.json').exists(), 'Uniform32B initial index not completed'

def load_vector_cache():
    import lancedb
    db = lancedb.connect(str(R / 'initial/index/lancedb'))
    return {n: {r['text']: r['vector'] for r in db.open_table(n).to_arrow().to_pylist()} for n in rowsets(load('base'))}

def rowsets(tables):
    return {'default-entity-description': [(str(r.id), str(r.title) + ':' + str(r.description)) for r in tables['entities'].itertuples()], 'default-text_unit-text': [(str(r.id), str(r.text)) for r in tables['text_units'].itertuples()], 'default-community-full_content': [(str(r.id), str(r.full_content)) for r in tables['community_reports'].itertuples()]}

async def materialize(tables, dest, cache=None):
    import lancedb
    from graphrag.query.indexer_adapters import read_indexer_reports, read_indexer_entities
    dest.mkdir(parents=True, exist_ok=True)
    for (name, table) in tables.items():
        table.to_parquet(dest / f'{name}.parquet', index=False)
    db = lancedb.connect(str(dest / 'lancedb'))
    stats = {}
    async with httpx.AsyncClient(timeout=600) as client:
        for (name, rows) in rowsets(tables).items():
            old = (cache or {}).get(name, {})
            need = [(i, text) for (i, text) in rows if text not in old]
            vectors = {}
            for k in range(0, len(need), 128):
                block = need[k:k + 128]
                reply = await client.post('http://127.0.0.1:18721/v1/embeddings', json={'input': [t for (_, t) in block]})
                reply.raise_for_status()
                values = reply.json()['data']
                assert len(values) == len(block)
                vectors.update({text: v['embedding'] for ((_, text), v) in zip(block, values)})
            data = [{'id': i, 'text': text, 'vector': old[text] if text in old else vectors[text], 'attributes': '{}'} for (i, text) in rows]
            tab = db.create_table(name, data=data, mode='overwrite')
            assert tab.count_rows() == len(rows)
            assert len(tab.search(data[0]['vector']).limit(1).to_list()) == 1
            stats[name] = {'rows': len(rows), 'embedded_rows': len(need), 'reused_vectors': len(rows) - len(need)}
    assert read_indexer_reports(tables['community_reports'], tables['communities'], 1)
    assert read_indexer_entities(tables['entities'], tables['communities'], 1)
    dump(dest / 'published.json', {'status': 'queryable', 'published_at': time.time(), 'tables': stats})
    return stats
q.f.e.materialize = materialize
q.f.e.load_vector_cache = load_vector_cache
q.totals = usage
