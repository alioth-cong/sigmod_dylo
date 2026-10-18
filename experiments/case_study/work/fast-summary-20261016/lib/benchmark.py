"""Measured graph-to-queryable-index benchmark on one shared insertion batch.

No historical LLM response cache is read. Old index reports may be reused only
when represented members AND their evidence are unchanged. Real requests and
native model usage are logged individually; phase timings use wall-clock time.
"""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import asyncio, csv, hashlib, json, logging, math, os, shutil, subprocess, sys, time
from uuid import uuid4, uuid5, NAMESPACE_URL
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).resolve().parent / 'cache/numba'))
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
import httpx
import numpy as np
import pandas as pd
import tiktoken
from citations import audit, allowed_ids
from citation_schema import source_schema, render_report
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'outputs/e123-20261015'
SRC = Path(_artifact_path('experiments/case_study/data/legacy'))
H = Path(_artifact_path('harness'))
STREAM = ROOT / 'data/stream'
ARMS = ['dylo', 'batch', 'hit-leiden', 'dynamo', 'qca', 'df-louvain', 'ds-louvain', 'rebuild']
CONCURRENCY = 1
MAX_OUTPUT = 2048
TEMPERATURES = [0, 0, 0]
REPAIR_INSTRUCTIONS = 'The previous attempt failed validation. Return complete JSON following the original report requirements. Use only record IDs from the actual CSV tables under Real Data, never from the examples. Use the human_readable_id column for citations. Do not invent record IDs or dataset names. Close every data citation and JSON string, and avoid repeating number sequences. Keep the original report length and findings requirements. Validation feedback: '
ENC = tiktoken.get_encoding('cl100k_base')
sys.path.insert(0, _artifact_path('experiments/case_study/work/fast-summary-20261016/lib'))
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from community_coverage import create_communities, entity_only_context
import graphrag.index.workflows.create_communities as cc
from graphrag.index.workflows.create_community_reports import _prep_nodes, _prep_edges
from graphrag.index.operations.summarize_communities.explode_communities import explode_communities
from graphrag.index.operations.summarize_communities.graph_context.context_builder import build_local_context, build_level_context
from graphrag.index.operations.summarize_communities.community_reports_extractor import CommunityReportsExtractor, CommunityReportResponse
from graphrag.index.operations.summarize_communities.typing import CommunityReport, Finding
from graphrag.index.operations.finalize_community_reports import finalize_community_reports
from graphrag.index.operations.summarize_communities.utils import get_levels
from graphrag.index.utils.stable_lcc import stable_largest_connected_component
import dylo_cluster

class Tokenizer:

    def encode(self, s):
        return ENC.encode(s)

    def decode(self, t):
        return ENC.decode(t)

    def num_tokens(self, s):
        return len(ENC.encode(s))

def say(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def load_tables(tag):
    return {n: pd.read_parquet(ROOT / f'data/{tag}/{n}.parquet') for n in ['documents', 'entities', 'relationships', 'text_units', 'communities', 'community_reports']}

def native(arm, dest):
    dest.mkdir(parents=True, exist_ok=True)
    env = {k: v for (k, v) in os.environ.items() if not k.startswith('DYLO_')}
    env['OMP_NUM_THREADS'] = '1'
    init = str(STREAM / 'initial_padded.txt')
    inc = str(STREAM / 'inc.txt')
    if arm == 'dylo':
        env.update(DYLO_EMIT_HIER='1', DYLO_LAMBDA='0.1', DYLO_AUDIT_K='8', DYLO_REBUILD_R='1e18', DYLO_SIGMA_ONLY='1', DYLO_SIGMA='0.2', DYLO_FINAL_AUDIT='1', DYLO_TOPDOWN='0', DYLO_HIER_GAMMA='5.0', DYLO_PRUNE='1', DYLO_PRUNE_MODE='leaf', DYLO_DEFER='1', DYLO_DEFER_EPS='0.05', DYLO_DEFER_MODE='quality', DYLO_TREE_DP='1', DYLO_LAZY_CERT='1', DYLO_INC_TREE='1', DYLO_LAZY_RECLAIM='1', DYLO_INIT_PART=str(STREAM / 'init_part.txt'))
        env['DYLO_BENCHMARK_SOURCE_STATE'] = str(dest / 'source_state.tsv')
        cmd = [str(ROOT / 'native/dylo_driver'), 'leiden', str(STREAM / 'initial.txt'), inc, '1', str(dest), '1']
    elif arm in ['batch', 'qca', 'dynamo']:
        java = _artifact_env('JAVA', 'java')
        cp = str(H / 'java/build') + ':' + (H / 'java/build/.classpath').read_text().strip()
        cmd = [java, '-Xmx16g', '-cp', cp, 'BaselineDriver', arm, init]
        if arm != 'dynamo':
            cmd.append(str(STREAM / 'baseline_init_part.txt'))
        cmd.extend([str(STREAM / ('incq.txt' if arm == 'qca' else 'inc.txt')), '1', str(dest)])
    elif arm == 'hit-leiden':
        env['DYLO_P'] = '3'
        cmd = [str(H / 'cpp/dylo/hit_leiden_driver_inc'), init, inc, '1', str(dest), '5.0']
    else:
        cmd = [str(H / 'cpp/dylo_driver'), 'df' if arm == 'df-louvain' else 'ds', init, inc, '1', str(dest), '1']
    start = time.perf_counter()
    with (dest / 'native.log').open('w') as log:
        subprocess.run(cmd, env=env, cwd=dest, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    elapsed = time.perf_counter() - start
    shutil.copy2(STREAM / 'id2name.json', dest / 'id2name.json')
    metrics = pd.read_csv(dest / 'metrics.tsv', sep='\t')
    row = metrics.iloc[-1].to_dict()
    assert int(row['batch']) == 1
    return (elapsed, row, cmd)

def title_members(comms, entities):
    names = dict(zip(entities.id, entities.title))
    return {int(r.community): frozenset((names[x] for x in r.entity_ids if x in names)) for r in comms.itertuples()}

def evidence(entities, relationships):
    en = {str(r.title): (str(r.description), tuple(sorted(map(str, r.text_unit_ids)))) for r in entities.itertuples()}
    edges = defaultdict(list)
    for r in relationships.itertuples():
        key = tuple(sorted((str(r.source), str(r.target))))
        edges[key].append((str(r.description), str(r.weight), tuple(sorted(map(str, r.text_unit_ids)))))
    return (en, {k: tuple(sorted(v)) for (k, v) in edges.items()})

def signature(members, en, edges):
    value = (tuple(sorted(((n, en.get(n)) for n in members))), tuple(sorted(((k, v) for (k, v) in edges.items() if k[0] in members and k[1] in members))))
    return hashlib.sha256(repr(value).encode()).hexdigest()

def modularity_contribution(members, edges):
    pairs = [k for k in edges if k[0] != k[1]]
    count = len(pairs)
    internal = sum((a in members and b in members for (a, b) in pairs))
    volume = sum((int(a in members) + int(b in members) for (a, b) in pairs))
    return internal / count - (volume / (2 * count)) ** 2 if count else 0

def make_plan(arm, base, newcomms, merged, native_dir, old_context_hashes, new_context_hashes):
    oldm = title_members(base['communities'], base['entities'])
    newm = title_members(newcomms, merged['entities'])
    (be, br) = evidence(base['entities'], base['relationships'])
    (me, mr) = evidence(merged['entities'], merged['relationships'])
    old_reports = {int(r.community): pd.Series(r._asdict()) for r in base['community_reports'].itertuples(index=False)}
    by_members = defaultdict(list)
    for (cid, m) in oldm.items():
        if cid in old_reports:
            by_members[m].append(cid)
    reused = []
    changed = []
    reasons = defaultdict(int)
    deferred = {}
    state = native_dir / 'source_state.tsv'
    if arm == 'dylo' and state.exists():
        names = json.loads((STREAM / 'id2name.json').read_text())
        for line in state.read_text().splitlines():
            (cid, qref, qnow, dirty, oldids, newids) = line.split('\t')
            before = frozenset((names[v] for v in oldids.split(',') if v and v in names))
            after = frozenset((names[v] for v in newids.split(',') if v and v in names))
            (qref, qnow) = (float(qref), float(qnow))
            drift = abs(qnow - qref) / abs(qref) if qref else 0 if not qnow else math.inf
            actual_ref = modularity_contribution(before, br)
            actual_now = modularity_contribution(after, mr)
            independent_drift = abs(actual_now - actual_ref) / abs(actual_ref) if actual_ref else 0 if not actual_now else math.inf
            if drift < 0.05 and independent_drift >= 0.05:
                reasons['native_source_drift_check_blocked'] += 1
            if not int(dirty) and after and (after < before) and (drift < 0.05) and (independent_drift < 0.05):
                deferred[after] = (before, drift, int(cid))
    for r in newcomms.itertuples():
        cid = int(r.community)
        m = newm[cid]
        candidates = by_members.get(m, [])
        found = None
        if arm != 'rebuild':
            for oldid in candidates:
                if signature(m, be, br) == signature(m, me, mr) and old_context_hashes[str(oldid)] == new_context_hashes[str(cid)]:
                    found = oldid
                    break
        if found is not None:
            row = old_reports[found].copy()
            row['community'] = cid
            row['human_readable_id'] = cid
            row['level'] = int(r.level)
            row['id'] = uuid5(NAMESPACE_URL, f"t123/{cid}/{row['id']}").hex
            for key in ['parent', 'children', 'size', 'period']:
                row[key] = getattr(r, key)
            reused.append(row)
        else:
            pending = deferred.get(m)
            if pending is not None:
                (before, drift, source) = pending
                candidates_old = by_members.get(before, [])
                if candidates_old and signature(m, be, br) == signature(m, me, mr) and (old_context_hashes[str(candidates_old[0])] == new_context_hashes[str(cid)]):
                    row = old_reports[candidates_old[0]].copy()
                    row['community'] = cid
                    row['human_readable_id'] = cid
                    row['level'] = int(r.level)
                    row['id'] = uuid5(NAMESPACE_URL, f"t123/{cid}/{row['id']}").hex
                    for key in ['parent', 'children', 'size', 'period']:
                        row[key] = getattr(r, key)
                    reused.append(row)
                    reasons['deferred_source'] += 1
                    continue
            changed.append(cid)
            reasons['rebuild' if arm == 'rebuild' else 'evidence_changed' if candidates else 'membership_changed'] += 1
    return (pd.DataFrame(reused), changed, dict(reasons))
from qwen_model import ActualModel

def prepare_contexts(comms, entities, relationships):
    from stable_context import stable_tables
    (entities, relationships) = stable_tables(entities, relationships, ROOT)
    callback = NoopWorkflowCallbacks()
    tok = Tokenizer()
    nodes = _prep_nodes(explode_communities(comms, entities.copy()))
    edges = _prep_edges(relationships.copy())
    local = build_local_context(nodes, edges, None, tok, callback, 4000)
    hierarchy = comms.explode('children').rename(columns={'children': 'sub_community'})[['community', 'level', 'sub_community']].dropna()
    levels = get_levels(nodes)
    contexts = [(level, build_level_context(pd.DataFrame(), hierarchy, local, tok, int(level), 4000)) for level in levels]
    isolated = {int(r.community): r for r in comms.itertuples() if len(r.relationship_ids) == 0}
    for (_, frame) in contexts:
        for (index, record) in frame.iterrows():
            cid = int(record.community)
            if cid in isolated:
                text = entity_only_context(isolated[cid], entities, tok, 4000)
                frame.at[index, 'context_string'] = text
                frame.at[index, 'context_size'] = tok.num_tokens(text)
                frame.at[index, 'context_exceed_limit'] = False
    ids = [int(r.community) for (_, frame) in contexts for r in frame.itertuples()]
    assert len(ids) == len(set(ids)) and set(ids) == set(map(int, comms.community)), 'Missing or duplicate summary contexts'
    return contexts

def context_hashes(contexts):
    return {str(int(r.community)): hashlib.sha256(r.context_string.encode()).hexdigest() for (_, frame) in contexts for r in frame.itertuples()}

async def reports(comms, entities, relationships, prompt, outdir, seed, phases, checkpoint=False, contexts=None):
    if contexts is None:
        start = time.perf_counter()
        contexts = prepare_contexts(comms, entities, relationships)
        phases['context_prepare_s'] = time.perf_counter() - start
    target = set(map(int, comms.community))
    contexts = [(level, frame[frame.community.isin(target)]) for (level, frame) in contexts]
    results = []
    level_times = []
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    planned = sum((len(c) for (_, c) in contexts))
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)) as client:
        with (outdir / 'requests.jsonl').open('a' if checkpoint else 'w') as log:

            async def one(record, level):
                nonlocal done
                async with sem:
                    saved = outdir / 'checkpoint' / f"{record['community']}.json"
                    if checkpoint and saved.exists():
                        done += 1
                        return CommunityReport(**json.loads(saved.read_text()))
                    result = None
                    feedback = ''
                    for attempt in range(len(TEMPERATURES)):
                        model = ActualModel(client, log, record['community'], level, attempt, seed, feedback)
                        extractor = CommunityReportsExtractor(model, extraction_prompt=prompt, max_report_length=2000)
                        result = await extractor(record['context_string'])
                        if result.structured_output is not None:
                            break
                        feedback = model.last_error
                    done += 1
                    if done % 25 == 0 or done == planned:
                        say(f'{outdir.name}: summaries {done}/{planned}')
                    r = result.structured_output
                    if r is None:
                        return None
                    report = CommunityReport(community=int(record['community']), full_content=result.output, level=int(level), rank=r.rating, title=r.title, rating_explanation=r.rating_explanation, summary=r.summary, findings=[Finding(explanation=f.explanation, summary=f.summary) for f in r.findings], full_content_json=r.model_dump_json(indent=4))
                    if checkpoint:
                        saved.parent.mkdir(exist_ok=True)
                        tmp = saved.with_suffix('.tmp')
                        tmp.write_text(json.dumps(report))
                        tmp.replace(saved)
                    return report
            start = time.perf_counter()
            for (level, context) in contexts:
                say(f'{outdir.name}: level {level}, {len(context)} reports')
                level_start = time.perf_counter()
                results.extend(await asyncio.gather(*(one(r, level) for r in context.to_dict('records'))))
                level_times.append({'level': int(level), 'tasks': len(context), 'wall_s': time.perf_counter() - level_start})
            phases['lm_generate_s'] = time.perf_counter() - start
    (outdir / 'level_timings.json').write_text(json.dumps(level_times, indent=2))
    start = time.perf_counter()
    valid = [r for r in results if r is not None]
    final = finalize_community_reports(pd.DataFrame(valid), comms) if valid else pd.DataFrame()
    phases['report_finalize_s'] = time.perf_counter() - start
    if planned - len(valid):
        raise RuntimeError(f'{planned - len(valid)} reports still missing after the common recovery policy; incomplete maintenance is not a valid timing trial')
    return (final, planned, planned - len(valid))

async def embed_and_store(tables, dest):
    """Materialize standard GraphRAG vector tables, using the original BGE model."""
    import lancedb
    from graphrag.query.indexer_adapters import read_indexer_reports, read_indexer_entities
    db = lancedb.connect(str(dest / 'lancedb'))
    datasets = {'default-entity-description': [(str(r.id), str(r.title) + ':' + str(r.description)) for r in tables['entities'].itertuples()], 'default-text_unit-text': [(str(r.id), str(r.text)) for r in tables['text_units'].itertuples()], 'default-community-full_content': [(str(r.id), str(r.full_content)) for r in tables['community_reports'].itertuples()]}
    async with httpx.AsyncClient(timeout=600) as client:
        for (name, rows) in datasets.items():
            data = []
            for k in range(0, len(rows), 128):
                block = rows[k:k + 128]
                response = await client.post('http://127.0.0.1:18621/v1/embeddings', json={'model': 'BAAI/bge-small-en-v1.5', 'input': [x[1] for x in block]})
                response.raise_for_status()
                vectors = response.json()['data']
                assert len(vectors) == len(block)
                data.extend(({'id': ident, 'text': text, 'vector': vector['embedding'], 'attributes': '{}'} for ((ident, text), vector) in zip(block, vectors)))
            db.create_table(name, data=data, mode='overwrite')
    r = read_indexer_reports(tables['community_reports'], tables['communities'], 1)
    e = read_indexer_entities(tables['entities'], tables['communities'], 1)
    assert r and e
    return {'reports_at_level_le1': len(r), 'retrieval_entities': len(e), 'vector_tables': list(datasets)}

async def run(arm, repeat):
    outdir = ROOT / 'runs' / f'r{repeat}-{arm}'
    if (outdir / 'result.json').exists():
        say(f'skip completed {outdir.name}')
        return
    outdir.mkdir(parents=True, exist_ok=True)
    (base, merged) = (load_tables('base'), load_tables('merged'))
    assert set(base['community_reports'].community) == set(base['communities'].community)
    old_context_hashes = json.loads((ROOT / 'initial/context_hashes.json').read_text())
    phases = {}
    start = time.perf_counter()
    native_row = {}
    cmd = []
    if arm != 'rebuild':
        (phases['native_driver_s'], native_row, cmd) = native(arm, outdir / 'native')
        dylo_cluster.DYLO_DIR = outdir / 'native'
        os.environ.update(DYLO_BATCH='1', DYLO_HIER_MODE='ragged', DYLO_FLAT_SUBDIVIDE='1')
        old_cluster = cc.cluster_graph
        cc.cluster_graph = dylo_cluster.dylo_cluster_graph
    else:
        phases['native_driver_s'] = 0
        old_cluster = cc.cluster_graph
    t = time.perf_counter()
    try:
        comms = create_communities(merged['entities'].copy(), merged['relationships'].copy(), 10, True, 3735928559)
    finally:
        cc.cluster_graph = old_cluster
    phases['hierarchy_adapter_s'] = time.perf_counter() - t
    t = time.perf_counter()
    contexts = prepare_contexts(comms, merged['entities'], merged['relationships'])
    phases['context_prepare_s'] = time.perf_counter() - t
    new_context_hashes = context_hashes(contexts)
    (outdir / 'context_hashes.json').write_text(json.dumps(new_context_hashes, indent=2))
    t = time.perf_counter()
    (inherited, changed, reasons) = make_plan(arm, base, comms, merged, outdir / 'native', old_context_hashes, new_context_hashes)
    phases['refresh_plan_s'] = time.perf_counter() - t
    say(f'{outdir.name}: communities={len(comms)}, reuse={len(inherited)}, scheduled={len(changed)}, reasons={reasons}')
    prompt = (ROOT / 'community_report_graph.txt').read_text()
    (fresh, planned, failed) = await reports(comms[comms.community.isin(changed)], merged['entities'], merged['relationships'], prompt, outdir, 20261012 + repeat, phases, contexts=contexts)
    t = time.perf_counter()
    parts = [d for d in [inherited, fresh] if len(d)]
    combined = pd.concat(parts, ignore_index=True)
    assert not combined.community.duplicated().any()
    assert not combined.id.duplicated().any()
    assert set(map(int, combined.community)) == set(map(int, comms.community)), 'Every current community must have a report before maintenance completes'
    tables = {**merged, 'communities': comms, 'community_reports': combined}
    index = outdir / 'index'
    index.mkdir(exist_ok=True)
    for (name, table) in tables.items():
        table.to_parquet(index / f'{name}.parquet', index=False)
    phases['index_write_s'] = time.perf_counter() - t
    t = time.perf_counter()
    checks = await embed_and_store(tables, index)
    phases['embedding_and_vector_store_s'] = time.perf_counter() - t
    elapsed = time.perf_counter() - start
    reqs = [json.loads(s) for s in (outdir / 'requests.jsonl').read_text().splitlines()]
    success = [r for r in reqs if r['status'] == 'ok']
    errors = [r for r in reqs if r['status'] != 'ok']
    assert len(success) == planned and len({r['community'] for r in success}) == planned
    result = {'arm': arm, 'repeat': repeat, 'total_s': elapsed, 'phases': phases, 'phase_residual_s': elapsed - sum(phases.values()), 'communities': len(comms), 'inherited_reports': len(inherited), 'scheduled_communities': len(changed), 'report_tasks': planned, 'generated_reports': len(fresh), 'failed_reports': failed, 'lm_requests': len(reqs), 'retries': len(reqs) - planned, 'repair_requests': sum((bool(r.get('generation_repair')) for r in reqs)), 'native_input_tokens': sum((r.get('prompt_tokens', 0) for r in reqs)), 'native_output_tokens': sum((r.get('completion_tokens', 0) for r in reqs)), 'input_cl100k': sum((r['input_cl100k'] for r in reqs)), 'request_time_sum_s': sum((r['request_s'] for r in reqs)), 'cache_hits': 0, 'finish_length': sum((r.get('finish_reason') == 'length' for r in reqs)), 'model': 'openai/gpt-oss-120b', 'reasoning_effort': 'low', 'concurrency': CONCURRENCY, 'successful_input_tokens': sum((r.get('prompt_tokens', 0) for r in success)), 'successful_output_tokens': sum((r.get('completion_tokens', 0) for r in success)), 'successful_input_cl100k': sum((r['input_cl100k'] for r in success)), 'failed_requests': len(errors), 'failed_input_tokens': sum((r.get('prompt_tokens', 0) for r in errors)), 'failed_output_tokens': sum((r.get('completion_tokens', 0) for r in errors)), 'citation_warnings': sum((sum(r['citation_audit']['warnings'].values()) for r in success)), 'initial_index_repair_tasks': 0, 'context_freshness': 'exact full-graph rendered-context SHA256 equality in addition to membership and evidence equality', 'primary_token_scope': 'one JSON-complete, citation-ID-grounded successful summary per updated community; failed attempts excluded', 'native_metrics': native_row, 'native_command': cmd, 'refresh_reasons': reasons, 'checks': checks, 'lm_level_times': json.loads((outdir / 'level_timings.json').read_text()), 'materialization': 'native source state with epsilon=.05, independently checked against initial report membership and graph; unchanged retained evidence and exact source-to-report mapping required'}
    (outdir / 'result.json').write_text(json.dumps(result, indent=2, default=str))
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outdir / 'result.json', OUT / f'{outdir.name}.json')
    say(f"RESULT {arm} r{repeat}: total={elapsed:.2f}s, LM={phases['lm_generate_s']:.2f}s, reports={len(fresh)}/{planned}, requests={len(reqs)}, tokens={result['native_input_tokens']}/{result['native_output_tokens']}")

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', nargs='+', default=ARMS)
    ap.add_argument('--repeat', type=int, default=1)
    a = ap.parse_args()
    for arm in a.arms:
        await run(arm, a.repeat)
if __name__ == '__main__':
    logging.basicConfig(filename=str(ROOT / 'benchmark-errors.log'), level=logging.ERROR)
    asyncio.run(main())
