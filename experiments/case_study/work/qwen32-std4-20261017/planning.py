"""Re-execute original planning inside each actual maintenance interval."""

def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import os, sys, json, time, hashlib, shutil, subprocess, collections
from pathlib import Path
import pandas as pd
from uuid import uuid5, NAMESPACE_URL
R = Path(__file__).resolve().parent
W = R.parents[1]
F = W / 'work/fast-summary-20261016'
C = Path(_artifact_path('experiments/case_study/data/legacy'))
sys.path.insert(0, str(F))
import quick_run as q
from graphrag.index.workflows import create_communities as cc
from graphrag.index.operations.cluster_graph import cluster_graph as stock_cluster
SEED = 3735928559
original = R / 'native/dylo_driver_std4'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, v):
    q.f.e.dump(p, v)

def contexts(comms, tables):
    b = q.f.b
    nodes = b._prep_nodes(b.explode_communities(comms, tables['entities'].copy()))
    edges = b._prep_edges(tables['relationships'].copy())
    tok = b.Tokenizer()
    local = b.build_local_context(nodes, edges, None, tok, b.NoopWorkflowCallbacks(), 4000)
    hierarchy = comms.explode('children').rename(columns={'children': 'sub_community'})[['community', 'level', 'sub_community']].dropna()
    result = [(level, b.build_level_context(pd.DataFrame(), hierarchy, local, tok, int(level), 4000)) for level in b.get_levels(nodes)]
    ids = [int(row.community) for (_, frame) in result for row in frame.itertuples()]
    assert set(ids) == set(map(int, comms.community)) and len(ids) == len(set(ids))
    return result

def build(arm, base, merged, dest):
    oldby = {}
    for row in base['communities'].itertuples():
        oldby.setdefault(frozenset(row.entity_ids), int(row.community))
    oldreports = {int(row.community): row._asdict() for row in base['community_reports'].itertuples(index=False)}
    checks = {}
    dest.mkdir(parents=True, exist_ok=True)
    native = dest / 'native'
    native.mkdir(exist_ok=True)
    graph_started = time.perf_counter()
    if arm == 'dylo':
        stream = C / 'dylo/stream'
        env = {k: v for (k, v) in os.environ.items() if not k.startswith(('DYLO_', 'E1_'))}
        env.update(DYLO_EMIT_HIER='1', DYLO_LAMBDA='1e-1', DYLO_AUDIT_K='8', DYLO_REBUILD_R='1e18', DYLO_SIGMA_ONLY='1', DYLO_SIGMA='0.2', DYLO_FINAL_AUDIT='1', DYLO_TOPDOWN='0', DYLO_HIER_GAMMA='5.0', DYLO_PRUNE='1', DYLO_PRUNE_MODE='leaf', DYLO_DEFER='0', DYLO_LAZY_CERT='1', DYLO_INC_TREE='1', DYLO_LAZY_RECLAIM='1', DYLO_INIT_PART=str(stream / 'init_part.txt'))
        args = [str(R / 'native/dylo_driver_std4'), 'leiden', str(stream / 'initial.txt'), str(stream / 'inc.txt'), '1', str(native)]
        with (native / 'native.log').open('w') as log:
            subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        shutil.copy2(stream / 'id2name.json', native / 'id2name.json')
        dump(native / 'command.json', {'command': args, 'environment': {k: v for (k, v) in env.items() if k.startswith('DYLO_')}, 'original_binary_sha256': sha(original)})
    elif arm == 'batch':
        q.f.stream_for(base, merged, dest)
        q.f.nr.N = F / 'native'
        handle = q.f.nr.prepare('batch', SEED, native)
        q.f.nr.release(handle, native)
    if arm in ['dylo', 'batch']:
        q.f.b.dylo_cluster.DYLO_DIR = native
        os.environ.update(DYLO_BATCH='1', DYLO_HIER_MODE='ragged', DYLO_FLAT_SUBDIVIDE='1')
        cc.cluster_graph = q.f.b.dylo_cluster.dylo_cluster_graph
    else:
        cc.cluster_graph = stock_cluster
    try:
        comms = cc.create_communities(merged['entities'].copy(), merged['relationships'].copy(), 10, True, SEED)
    finally:
        cc.cluster_graph = stock_cluster
    graph_wall = time.perf_counter() - graph_started
    if arm == 'dylo':
        sig = lambda frame: {(int(row.level), frozenset(row.entity_ids)) for row in frame.itertuples()}
        assert sig(comms) == sig(merged['communities']), ('Original std4 replay differs from archived DyLo partition', len(comms), len(merged['communities']))
    selection_started = time.perf_counter()
    retained = []
    changed = []
    repair = []
    reasons = {}
    for row in comms.itertuples():
        cid = int(row.community)
        old = oldby.get(frozenset(row.entity_ids))
        if arm != 'rebuild' and old is not None and (old in oldreports):
            rec = dict(oldreports[old])
            rec.update(community=cid, human_readable_id=cid, level=int(row.level), id=uuid5(NAMESPACE_URL, f"table5-std4/{arm}/{cid}/{rec['id']}").hex)
            for key in ['parent', 'children', 'size', 'period']:
                rec[key] = getattr(row, key)
            retained.append(rec)
            reasons[str(cid)] = 'same_members_with_old_report'
        else:
            changed.append(cid)
            reasons[str(cid)] = 'full_rebuild' if arm == 'rebuild' else 'old_report_missing' if old is not None else 'membership_changed'
            if arm != 'rebuild' and old is not None:
                repair.append(cid)
    assert len(retained) + len(changed) == len(comms)
    comms.to_parquet(dest / 'communities.parquet', index=False)
    pd.DataFrame(retained).to_parquet(dest / 'inherited.parquet', index=False)
    selection_wall = time.perf_counter() - selection_started
    context_started = time.perf_counter()
    ctx = contexts(comms, merged)
    context_wall = time.perf_counter() - context_started
    dump(dest / 'contexts.json', [(int(level), frame.to_dict('records')) for (level, frame) in ctx])
    dump(dest / 'context_hashes.json', q.f.b.context_hashes(ctx))
    dump(dest / 'refresh_plan.json', {'changed': changed, 'initial_missing_repairs': repair, 'net_update_targets': sorted(set(changed) - set(repair)), 'inherited': len(retained), 'refresh_reasons': reasons, 'reuse_rule': 'Original dylo_patch: first old community with identical entity_id set and an available report; no input/evidence check', 'full_rebuild': arm == 'rebuild'})
    checks[arm] = {'communities': len(comms), 'merged_refresh_targets': len(changed), 'new_document_delta_summary_targets': 26 if arm != 'rebuild' else 0, 'table5_style_total_targets': len(changed) + (26 if arm != 'rebuild' else 0), 'initial_missing_repairs': len(repair), 'net_update_targets_including_delta': len(changed) - len(repair) + (26 if arm != 'rebuild' else 0), 'inherited': len(retained)}
    print(arm, checks[arm], flush=True)
    return (comms, ctx, {'changed': changed, 'initial_missing_repairs': repair, 'net_update_targets': sorted(set(changed) - set(repair)), 'refresh_reasons': reasons}, pd.DataFrame(retained), {'community_update_or_rebuild_s': graph_wall, 'refresh_selection_s': selection_wall, 'context_prepare_s': context_wall})
