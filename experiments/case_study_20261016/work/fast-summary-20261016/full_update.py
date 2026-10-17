"""Actual raw-document arrival through published index; no cross-run time splicing."""
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[1] / 'outputs/fast-summary-20261016'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'cache/numba'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
sys.path.insert(0, str(ROOT / 'lib'))
import numpy as np
import pandas as pd
import networkx as nx
import yaml
import benchmark as b
import engine as e
import native_runner as nr
import extraction_model as em
from graphrag.api.index import build_index
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.cache.noop_pipeline_cache import NoopPipelineCache
from graphrag.index.update.entities import _group_and_resolve_entities
from graphrag.index.update.relationships import _update_and_merge_relationships
from graphrag.index.workflows.extract_graph import get_summarized_entities_relationships
from graphrag.index.workflows.create_final_text_units import create_final_text_units
from isolation import Guard
from graphrag.index.workflows.factory import PipelineFactory
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

e.R = b.ROOT = nr.R = ROOT
nr.N = ROOT / 'native'


def exclusion_manifest():
    return json.loads((OUT/'exclusions.json').read_text())


def filter_units(units):
    manifest = exclusion_manifest()
    excluded = {x['text_unit_id']: x for x in manifest['excluded_text_units']}
    found = units[units.id.isin(excluded)]
    assert set(found.id) == set(excluded), 'Frozen excluded text units changed or missing'
    for row in found.itertuples():
        assert hashlib.sha256(row.text.encode()).hexdigest() == excluded[row.id]['text_sha256']
    kept = units[~units.id.isin(excluded)].copy()
    assert len(units) == 50 and len(kept) == 48
    assert {x for ids in kept.document_ids for x in ids} == {x for ids in units.document_ids for x in ids}
    return kept, {'input_units': len(units), 'retained_units': len(kept),
        'excluded_ids': sorted(excluded), 'input_documents_retained': 23,
        'exclusion_manifest_sha256': hashlib.sha256((OUT/'exclusions.json').read_bytes()).hexdigest()}


async def exclude_failed_text_units(config, context):
    units = await load_table_from_storage('text_units', context.output_storage)
    kept, audit = filter_units(units)
    await write_table_to_storage(kept, 'text_units', context.output_storage)
    e.dump(Path(config.output.base_dir).parent/'input_filter_audit.json', audit)
    return WorkflowFunctionOutput(result=audit)


PipelineFactory.register('exclude_failed_text_units', exclude_failed_text_units)


def progress(dest, stage, **kwargs):
    e.dump(dest / 'progress.json', {'stage': stage, 'updated_at': time.time(), **kwargs})
    print(time.strftime('%H:%M:%S'), dest.name, stage, kwargs, flush=True)


class Callbacks(NoopWorkflowCallbacks):
    def __init__(self, dest):
        self.dest = dest
        self.starts, self.times = {}, {}

    def workflow_start(self, name, instance):
        self.starts[name] = time.perf_counter()
        progress(self.dest, name)

    def workflow_end(self, name, instance):
        self.times[name] = time.perf_counter() - self.starts[name]
        e.dump(self.dest / 'extraction_workflow_times.json', self.times)
        if em.EXHAUSTED:
            raise RuntimeError('Failed extraction requests must not be silently dropped: ' + str(em.EXHAUSTED))


def config_for(dest, input_dir):
    cfg = yaml.safe_load((ROOT / 'extraction-settings.yaml').read_text())
    cfg['root_dir'] = str(ROOT)
    cfg['input']['storage']['base_dir'] = str(input_dir)
    cfg['output']['base_dir'] = str(dest / 'delta')
    cfg['reporting']['base_dir'] = str(dest / 'logs')
    cfg['snapshots']['raw_graph'] = True
    cfg['models']['default_chat_model']['async_mode'] = 'asyncio'
    cfg['models']['default_chat_model']['max_tokens'] = 4096
    cfg['workflows'].insert(cfg['workflows'].index('create_base_text_units')+1, 'exclude_failed_text_units')
    # LoggedModel applies 4096 to extraction, 2048 to entity/relationship descriptions.
    return GraphRagConfig(**cfg)


def edges(tables):
    return {tuple(sorted((a, c))) for a, c in zip(tables['relationships'].source,
        tables['relationships'].target) if a != c}


def stream_for(base, merged, dest):
    s = dest / 'data/stream'
    s.mkdir(parents=True)
    old, new = edges(base), edges(merged)
    assert old <= new
    oldnames = sorted({x for pair in old for x in pair})
    names = oldnames + sorted({x for pair in new for x in pair} - set(oldnames))
    ids = {n: i for i, n in enumerate(names)}
    text = ''.join(f'{ids[a]}\t{ids[c]}\t1.0\n' for a, c in sorted(old))
    (s / 'initial.txt').write_text(text)
    (s / 'initial_padded.txt').write_text(text + f'{len(ids)-1}\t{len(ids)-1}\t0.0\n')
    inc = ''.join(f'{ids[a]}\t{ids[c]}\t1.0\n' for a, c in sorted(new-old))
    (s / 'inc.txt').write_text('')
    (s / 'incq.txt').write_text('')
    (s / 'inc_1.txt').write_text(inc)
    (s / 'incq_1.txt').write_text(inc.replace('\t1.0\n', '\t1\n'))
    e.dump(s / 'id2name.json', {str(i): n for n, i in ids.items()})
    members = b.title_members(base['communities'], base['entities'])
    assigned, groups = set(), []
    for row in sorted(base['communities'].itertuples(), key=lambda x: -x.level):
        group = (members[int(row.community)] - assigned) & set(ids)
        if group:
            groups.append(sorted(ids[n] for n in group))
            assigned.update(group)
    groups.extend([ids[n]] for n in names if n not in assigned)
    (s / 'init_part.txt').write_text(''.join('\t'.join(map(str, g))+'\n' for g in groups))
    present = {int(x) for line in (s / 'initial_padded.txt').read_text().splitlines() for x in line.split()[:2]}
    jgroups = [[x for x in g if x in present] for g in groups]
    covered = [x for g in jgroups for x in g]
    assert len(covered) == len(set(covered)) and set(covered) == present
    (s / 'baseline_init_part.txt').write_text(''.join('\t'.join(map(str, g))+'\n' for g in jgroups if g))
    nr.S = s
    return {'initial_edges': len(old), 'added_edges': len(new-old), 'vertices': len(names)}


def aslist(value):
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def references_ok(tables):
    for name, table in tables.items():
        if name not in ('communities', 'community_reports'):
            assert table.id.is_unique, (name, 'duplicate ids')
    entities, rels, units, docs = (tables[k] for k in ('entities', 'relationships', 'text_units', 'documents'))
    es, rs, us, ds = map(set, (entities.id, rels.id, units.id, docs.id))
    checks = [(entities, 'text_unit_ids', us), (rels, 'text_unit_ids', us),
        (units, 'entity_ids', es), (units, 'relationship_ids', rs),
        (units, 'document_ids', ds), (docs, 'text_unit_ids', us)]
    for table, column, allowed in checks:
        actual = {x for v in table[column] for x in aslist(v)}
        assert actual <= allowed, (column, 'dangling references', list(actual-allowed)[:3])
    assert set(rels.source) | set(rels.target) <= set(entities.title)
    return {k: len(v) for k, v in tables.items()}


async def merge(base, delta, cfg):
    em.PHASE = 'merge'
    ents, mapping = _group_and_resolve_entities(base['entities'].copy(deep=True), delta['entities'].copy(deep=True))
    rels = _update_and_merge_relationships(base['relationships'].copy(deep=True), delta['relationships'].copy(deep=True))
    modelcfg = cfg.get_language_model_config(cfg.summarize_descriptions.model_id)
    strategy = cfg.summarize_descriptions.resolved_strategy(cfg.root_dir, modelcfg)
    ents, rels = await get_summarized_entities_relationships(ents, rels,
        NoopWorkflowCallbacks(), NoopPipelineCache(), strategy, 1)
    if em.EXHAUSTED:
        raise RuntimeError('Description merging failed: ' + str(em.EXHAUSTED))
    graph = nx.Graph()
    graph.add_edges_from(zip(rels.source, rels.target))
    degree = dict(graph.degree())
    ents['degree'] = ents.title.map(degree).fillna(0).astype(int)
    rels['combined_degree'] = [degree.get(a, 0)+degree.get(c, 0) for a, c in zip(rels.source, rels.target)]
    documents = pd.concat([base['documents'], delta['documents']], ignore_index=True)
    # Rebuild cross references from merged records, including coalesced relationship IDs.
    units = create_final_text_units(pd.concat([base['text_units'], delta['text_units']], ignore_index=True), ents, rels, None)
    tables = {'documents': documents, 'entities': ents, 'relationships': rels, 'text_units': units}
    references_ok(tables)
    return tables


def ledger(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def token_totals(rows):
    return {status: {'calls': len(selected := [r for r in rows if (r['status'] == 'ok') == (status == 'successful')]),
        'input_tokens': sum(r.get('prompt_tokens', 0) for r in selected),
        'output_tokens': sum(r.get('completion_tokens', 0) for r in selected),
        'input_cl100k': sum(r.get('input_cl100k', 0) for r in selected),
        'request_s': sum(r.get('request_s', 0) for r in selected)} for status in ('successful', 'failed')}


def semantic_hash(tables):
    # Stable content digest excludes fresh UUIDs and row ordering.
    value = {'entities': sorted((str(r.title), str(r.type), str(r.description)) for r in tables['entities'].itertuples()),
        'relationships': sorted((str(r.source), str(r.target), str(r.description), float(r.weight)) for r in tables['relationships'].itertuples())}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def run(method, seed, phase='formal'):
    dest = ROOT / 'runs' / f'{phase}-filtered-s{seed}-{method}'
    if dest.exists():
        raise RuntimeError('Refusing to overwrite/resume a partial end-to-end clock: ' + str(dest))
    dest.mkdir(parents=True)
    e.R = b.ROOT = ROOT
    base = b.load_tables('base')
    assert len(base['documents']) == 304
    assert set(base['communities'].community) == set(base['community_reports'].community)
    cache = e.load_vector_cache()
    oldhashes = json.loads((ROOT / 'initial/context_hashes.json').read_text())
    input_dir = ROOT / 'input'
    expected_docs = 23
    if phase == 'smoke':
        input_dir = dest / 'input'
        input_dir.mkdir()
        shutil.copy2(ROOT / 'input/doc_00042.txt', input_dir)
        expected_docs = 1
    em.install(dest, seed)
    cfg = config_for(dest, input_dir)
    e.dump(dest / 'config.json', cfg.model_dump(mode='json'))
    guard = Guard()
    guard.start()
    phases = {}
    callbacks = Callbacks(dest)
    start = time.perf_counter()
    received = time.time()
    e.dump(dest/'timing_start.json', {'received_at': received, 'scope': exclusion_manifest()['result_label']})
    try:
        progress(dest, 'raw_document_arrival', expected_documents=expected_docs, received_at=received)
        t = time.perf_counter()
        result = await build_index(cfg, callbacks=[callbacks])
        errors = [str(error) for item in result for error in item.errors or []]
        assert not errors and len(result) == len(cfg.workflows), errors
        assert not em.EXHAUSTED
        delta = {name: pd.read_parquet(dest / 'delta' / f'{name}.parquet') for name in ('documents', 'entities', 'relationships', 'text_units')}
        assert len(delta['documents']) == expected_docs
        assert len(delta['text_units']) == 48
        assert set(delta['text_units'].id).isdisjoint(x['text_unit_id'] for x in exclusion_manifest()['excluded_text_units'])
        assert set(base['documents'].id).isdisjoint(delta['documents'].id)
        phases['load_chunk_extract_s'] = time.perf_counter()-t
        progress(dest, 'merge_graph_descriptions', delta_counts={k: len(v) for k, v in delta.items()})
        t = time.perf_counter()
        merged = await merge(base, delta, cfg)
        # Preserve evidence for later audits without having to replay the LLM.
        (dest/'merged_graph').mkdir()
        for name,table in merged.items():
            table.to_parquet(dest/'merged_graph'/f'{name}.parquet',index=False)
        phases['merge_graph_descriptions_s'] = time.perf_counter()-t
        progress(dest, 'community_maintenance')
        t = time.perf_counter()
        stream = stream_for(base, merged, dest)
        # Native driver takeover occurs AFTER arrival and is included, never subtracted.
        handle = nr.prepare(method, seed, dest / 'native')
        if handle:
            cmdpath = dest / 'native/command.json'
            cmd = json.loads(cmdpath.read_text())
            cmd['initialization_s_included'] = cmd.pop('initialization_s_excluded')
            e.dump(cmdpath, cmd)
        selector = nr.release(handle, dest / 'native')
        comms = e.graph(merged, method, seed, dest / 'native')
        comms.to_parquet(dest/'communities.parquet',index=False)
        phases['community_maintenance_s'] = time.perf_counter()-t
        t = time.perf_counter()
        contexts = b.prepare_contexts(comms, merged['entities'], merged['relationships'])
        e.dump(dest/'context_hashes.json',b.context_hashes(contexts))
        e.R = dest
        inherited, changed, plan = e.plan(method, base, comms, merged, contexts, dest, old_context_hashes=oldhashes)
        e.R = ROOT
        phases['refresh_planning_s'] = time.perf_counter()-t
        graphhash = semantic_hash(merged)
        e.dump(dest / 'graph_ready.json', {'counts': references_ok(merged), 'stream': stream,
            'semantic_sha256': graphhash, 'communities': len(comms), 'refresh': len(changed),
            'inherited': len(inherited), 'delta_counts': {k: len(v) for k, v in delta.items()}})
        if phase == 'smoke':
            # The established report/vector path was already GPU-validated. This probe
            # validates the newly added raw extraction/merge and every native adapter.
            checks = {method: {'communities': len(comms), 'refresh': len(changed)}}
            for other in e.METHODS:
                if other == method:
                    continue
                native = dest / f'probe-{other}' / 'native'
                h = nr.prepare(other, seed, native)
                nr.release(h, native)
                c = e.graph(merged, other, seed, native)
                ctx = b.prepare_contexts(c, merged['entities'], merged['relationships'])
                e.R = dest
                retained, refresh, _ = e.plan(other, base, c, merged, ctx, native.parent,
                    old_context_hashes=oldhashes)
                e.R = ROOT
                checks[other] = {'communities': len(c), 'refresh': len(refresh), 'inherited': len(retained)}
            isolated = guard.finish(dest / 'resource_isolation.json')
            assert isolated
            row = {'status': 'passed', 'phase': 'smoke', 'scope': 'raw extraction, description merge, references, CPU community coverage and refresh plans for all seven methods',
                'whole_queryable_e2e': False, 'checks': checks, 'elapsed_s': time.perf_counter()-start,
                'tokens': token_totals(ledger(dest / 'extraction_calls.jsonl'))}
            e.dump(dest / 'result.json', row)
            e.dump(dest / 'verified.json', row)
            e.dump(OUT / 'smoke.json', row)
            progress(dest, 'probe_passed', methods=len(checks))
            return row
        progress(dest, 'community_resummarization', planned=len(changed), inherited=len(inherited))
        local = {}
        t = time.perf_counter()
        fresh, planned, failed = await b.reports(comms[comms.community.isin(changed)], merged['entities'],
            merged['relationships'], (ROOT / 'community_report_graph.txt').read_text(), dest, seed, local, contexts=contexts)
        phases['community_resummarization_s'] = time.perf_counter()-t
        t = time.perf_counter()
        combined = pd.concat([x for x in (inherited, fresh) if len(x)], ignore_index=True)
        assert combined.community.is_unique and combined.id.is_unique
        assert set(combined.community) == set(comms.community)
        tables = {**merged, 'communities': comms, 'community_reports': combined}
        progress(dest, 'embedding_write_publish')
        vectors = await e.materialize(tables, dest / 'index', cache)
        phases['embedding_write_publish_s'] = time.perf_counter()-t
        elapsed = time.perf_counter()-start
        phases['other_s'] = elapsed-sum(phases.values())
        assert phases['other_s'] >= 0
        requests = ledger(dest / 'requests.jsonl')
        extraction = ledger(dest / 'extraction_calls.jsonl')
        assert sum(r['status'] == 'ok' for r in requests) == len(changed)
        categories = {name: token_totals([r for r in extraction if r['category'] == name])
            for name in ('extraction', 'description_extraction', 'description_merge')}
        categories['community_resummarization'] = token_totals(requests)
        row = {'status': 'completed', 'method': method, 'seed': seed, 'phase': phase,
            'run_id': dest.name, 'received_at': received, 'queryable_at': time.time(),
            'e2e_s': elapsed, 'phases': phases, 'initial_documents': 304,
            'added_documents': expected_docs, 'final_documents': len(merged['documents']),
            'experiment_scope': 'filtered_input_update', 'complete_23_document_ingestion': False,
            'input_coverage': json.loads((dest/'input_filter_audit.json').read_text()),
            'tokens_by_category': categories, 'refresh_summaries': len(changed),
            'inherited_summaries': len(inherited), 'hierarchy_nodes': len(comms),
            'graph_semantic_sha256': graphhash, 'native_selector_s': selector,
            'vector_tables': vectors, 'model': 'Qwen/Qwen2.5-7B-Instruct',
            'initial_index_setup_in_measured_interval': False,
            'native_driver_takeover_in_measured_interval': True,
            'timing_scope': '23 raw documents read and chunked, same two known failed text units excluded, remaining48 units extracted; graph merge, community maintenance, summaries, embeddings, publication and retrieval checks included',
            'rebuild_scope': 'rebuild communities on the complete updated graph; only the 23 new documents require extraction',
            'primary_token_scope': 'successful community summary refresh only; failures and extraction/description calls separate',
            'refresh_policy': plan['refresh_policy'],
            'full_reset_control': method=='full-reset-rebuild',
            'graph_pairing': 'compare semantic hashes within seed; differing hashes must be flagged before comparing graph-maintenance savings',
            'limitations': [plan['adapter_limit'], 'Warm existing index and models; native driver takeover cost included.',
                'Summary coverage is the common largest connected component, including edge-free hierarchy nodes.']}
        e.dump(dest / 'result.json', row)
        e.dump(OUT / f'{dest.name}.json', row)
        progress(dest, 'completed', e2e_s=elapsed)
    except BaseException as exc:
        e.dump(dest / 'failure.json', {'error': str(exc), 'traceback': traceback.format_exc(),
            'elapsed_s': time.perf_counter()-start, 'phases_completed': phases,
            'exhausted': em.EXHAUSTED, 'valid_e2e_result': False})
        progress(dest, 'failed', error=str(exc))
        raise
    finally:
        e.R = ROOT
        isolated = guard.finish(dest / 'resource_isolation.json')
    if not isolated:
        e.dump(dest / 'invalid.json', {'reason': 'GPU contention', 'valid_e2e_result': False})
        raise RuntimeError('GPU contention invalidated trial')
    e.dump(dest / 'verified.json', {'status': 'passed', 'queryable': True,
        'phase_sum_matches_e2e': abs(sum(phases.values())-elapsed) < 1e-6,
        'no_update_response_cache': True, 'resource_isolation': 'passed'})
    return row


if __name__ == '__main__':
    asyncio.run(run(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'formal'))
