def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import os, asyncio, time
from runtime import *

async def main():
    if (R / 'initial/complete.json').exists():
        assert_initial32(load('base'))
        return
    assert read(O / 'stability.json')['status'] == 'passed'
    q.f.b.CONCURRENCY = int(os.environ['QWEN32_CONCURRENCY'])
    base = load('base')
    dest = R / 'initial'
    dest.mkdir(exist_ok=True)
    started = time.perf_counter()
    phases = {}
    try:
        (fresh, n, _) = await q.f.b.reports(base['communities'], base['entities'], base['relationships'], (R / 'community_report_graph.txt').read_text(), dest, 17, phases, checkpoint=True, contexts=contexts(R / 'plans/initial_contexts.json'))
        assert len(fresh) == n == 1554 and fresh.community.is_unique and (set(fresh.community) == set(base['communities'].community))
        fresh['summary_model'] = MODEL
        fresh.to_parquet(R / 'data/base/community_reports.parquet', index=False)
        base['community_reports'] = fresh
        publication_started = time.perf_counter()
        vectors = await materialize(base, dest / 'index')
        publication_s = time.perf_counter() - publication_started
        calls = records(dest / 'requests.jsonl')
        assert all((r.get('model') == MODEL for r in calls))
        result = {'status': 'complete', 'model': MODEL, 'initial_documents': 304, 'reports': len(fresh), 'originally_missing_reports_now_regenerated': 134, 'all_reports_regenerated_with32B': True, 'initial_report_sha256': sha(R / 'data/base/community_reports.parquet'), 'tokens': usage(calls), 'attempt_wall_s': time.perf_counter() - started, 'publication_s': publication_s, 'vector_tables': vectors, 'cost_scope': 'initial304 setup only; all excluded from maintenance token and time', 'at': time.time()}
        dump(dest / 'complete.json', result)
        dump(O / 'initial_setup.json', result)
    finally:
        with (dest / 'attempt_times.jsonl').open('a') as f:
            f.write(json.dumps({'attempt_wall_s': time.perf_counter() - started, 'at': time.time(), 'complete': (dest / 'complete.json').exists()}) + '\n')
if __name__ == '__main__':
    asyncio.run(main())
