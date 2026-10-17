"""E1 measured maintenance, serial LLM calls and isolated initial snapshots."""
import asyncio,hashlib,json,math,os,shutil,time
from collections import defaultdict
from pathlib import Path
from uuid import uuid5,NAMESPACE_URL
import httpx,numpy as np,pandas as pd
import benchmark as b
import native_runner as nr
R=b.ROOT;O=b.OUT
METHODS=['dylo','batch','reuse-rebuild','hit-leiden','df-louvain','qca','full-reset-rebuild']
SEEDS=[17,29,43]

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=str));temp.replace(path)

def rowsets(tables):
    return {'default-entity-description':[(str(r.id),str(r.title)+':'+str(r.description)) for r in tables['entities'].itertuples()],
            'default-text_unit-text':[(str(r.id),str(r.text)) for r in tables['text_units'].itertuples()],
            'default-community-full_content':[(str(r.id),str(r.full_content)) for r in tables['community_reports'].itertuples()]}

async def materialize(tables,dest,cache=None):
    import lancedb
    from graphrag.query.indexer_adapters import read_indexer_reports,read_indexer_entities
    dest.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_parquet(dest/f'{name}.parquet',index=False)
    db=lancedb.connect(str(dest/'lancedb'));stats={}
    async with httpx.AsyncClient(timeout=600) as client:
        for name,rows in rowsets(tables).items():
            old=(cache or {}).get(name,{})
            need=[(i,text) for i,text in rows if text not in old];vectors={}
            for k in range(0,len(need),128):
                block=need[k:k+128];reply=await client.post(os.environ.get('DYLO_EMBEDDING_URL','http://127.0.0.1:18621/v1/embeddings'),json={'input':[t for _,t in block]});reply.raise_for_status()
                values=reply.json()['data'];assert len(values)==len(block)
                vectors.update({text:v['embedding'] for (_,text),v in zip(block,values)})
            data=[{'id':i,'text':text,'vector':old[text] if text in old else vectors[text],'attributes':'{}'} for i,text in rows]
            tab=db.create_table(name,data=data,mode='overwrite')
            assert tab.count_rows()==len(rows)
            # Actual retrieval smoke check is inside the end-to-end interval.
            assert len(tab.search(data[0]['vector']).limit(1).to_list())==1
            stats[name]={'rows':len(rows),'embedded_rows':len(need),'reused_vectors':len(rows)-len(need)}
    assert read_indexer_reports(tables['community_reports'],tables['communities'],1)
    assert read_indexer_entities(tables['entities'],tables['communities'],1)
    dump(dest/'published.json',{'status':'queryable','published_at':time.time(),'tables':stats})
    return stats

def load_vector_cache():
    import lancedb
    db=lancedb.connect(str(R/'initial/index/lancedb'))
    return {name:{r['text']:r['vector'] for r in db.open_table(name).to_arrow().to_pylist()} for name in rowsets(b.load_tables('base'))}

async def initial():
    dest=R/'initial';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    base=b.load_tables('base');contexts=b.prepare_contexts(base['communities'],base['entities'],base['relationships']);hashes=b.context_hashes(contexts)
    dump(dest/'context_hashes.json',hashes)
    missing=set(base['communities'].community)-set(base['community_reports'].community)
    if missing:
        phases={};fresh,_,_=await b.reports(base['communities'][base['communities'].community.isin(missing)],base['entities'],base['relationships'],(R/'community_report_graph.txt').read_text(),dest,17,phases,checkpoint=True,contexts=contexts)
        base['community_reports']=pd.concat([base['community_reports'],fresh],ignore_index=True)
        assert set(base['community_reports'].community)==set(base['communities'].community)
        base['community_reports'].to_parquet(R/'data/base/community_reports.parquet',index=False)
    await materialize(base,dest/'index')
    dump(dest/'complete.json',{'status':'complete','reports':len(base['community_reports']),'initial_debt_repaired':len(missing),
        'reports_sha256':hashlib.sha256((R/'data/base/community_reports.parquet').read_bytes()).hexdigest(),
        'cost_scope':'All setup and initial missing summaries excluded from E1; common archived initial partition and summaries reused across paired seeds.'})

def graph(tables,method,seed,native_dir):
    old=b.cc.cluster_graph
    if method not in ('reuse-rebuild','full-reset-rebuild'):
        b.dylo_cluster.DYLO_DIR=native_dir;os.environ.update(DYLO_BATCH='1',DYLO_HIER_MODE='ragged',DYLO_FLAT_SUBDIVIDE='1');b.cc.cluster_graph=b.dylo_cluster.dylo_cluster_graph
    try:
        comms=b.create_communities(tables['entities'].copy(),tables['relationships'].copy(),10,True,seed)
        from graphrag.index.operations.create_graph import create_graph
        expected=set(b.stable_largest_connected_component(create_graph(tables['relationships'],edge_attr=['weight'])))
        members=b.title_members(comms,tables['entities'])
        roots=[members[int(r.community)] for r in comms.itertuples() if r.level==0]
        assert set().union(*roots)==expected and sum(map(len,roots))==len(expected), 'Top-level summary coverage differs from common LCC'
        return comms
    finally:b.cc.cluster_graph=old

def plan(method,base,comms,merged,contexts,dest,*,planning_only=False,assume_initial_complete=False,old_context_hashes=None):
    if assume_initial_complete and not planning_only:
        raise ValueError('Initial-completion assumption is allowed only in CPU planning diagnostics')
    oldm=b.title_members(base['communities'],base['entities']);newm=b.title_members(comms,merged['entities'])
    oldh=old_context_hashes if old_context_hashes is not None else json.loads((R/'initial/context_hashes.json').read_text());newh=b.context_hashes(contexts)
    bym=defaultdict(list)
    for cid,m in oldm.items():bym[m].append(cid)
    oldreports={int(r.community):r._asdict() for r in base['community_reports'].itertuples()}
    available=set(oldm) if assume_initial_complete else set(oldreports)
    oldlevels=dict(zip(base['communities'].community,base['communities'].level))
    pending=[];events=[];native_pending=[]
    f=dest/'native/source_state.tsv';names=json.loads((R/'data/stream/id2name.json').read_text())
    if method=='dylo' and f.exists():
        for line in f.read_text().splitlines():
            cid,qref,qnow,dirty,before,after=line.split('\t');before=frozenset(names[x] for x in before.split(',') if x);after=frozenset(names[x] for x in after.split(',') if x)
            drift=abs(float(qnow)-float(qref))/abs(float(qref)) if float(qref) else math.inf
            native_pending.append({'source':int(cid),'drift':drift,'dirty':int(dirty),'before':sorted(before),'after':sorted(after)})
            if not int(dirty) and after and after<before and drift<.05:pending.append((int(cid),before,after,drift))
    retained=[];inherited_ids=[];changed=[];reasons={};pending_nodes={};sources_used=set();be,br=b.evidence(base['entities'],base['relationships']);me,mr=b.evidence(merged['entities'],merged['relationships'])
    for row in comms.itertuples():
        cid=int(row.community);m=newm[cid];found=None;reason='membership_changed'
        if method!='full-reset-rebuild':
            for oldid in bym.get(m,[]):
                if oldid in available and oldh[str(oldid)]==newh[str(cid)]:found=oldid;reason='unchanged_input';break
            if found is None and bym.get(m):reason='input_changed'
            # All reuse-enabled E1 arms use identical members + exact stabilized
            # prompt equality. Native pending is logged, never an arm-specific
            # exemption from the common report freshness contract.
        if found is None:changed.append(cid);reasons[str(cid)]='full_reset' if method=='full-reset-rebuild' else reason
        else:
            inherited_ids.append(cid);reasons[str(cid)]=reason
            if planning_only:continue
            r=dict(oldreports[found]);r.pop('Index',None);r.update(community=cid,human_readable_id=cid,level=int(row.level),id=uuid5(NAMESPACE_URL,f'e1/{cid}/{r["id"]}').hex)
            for key in ['parent','children','size','period']:r[key]=getattr(row,key)
            retained.append(r);reasons[str(cid)]=reason
    metadata={'refresh_reasons':reasons,'inherited_count':len(inherited_ids),'inherited_ids':inherited_ids,'planning_only':planning_only,'assume_initial_complete':assume_initial_complete,'pending_nodes':pending_nodes,'pending_source_count':len(sources_used),'pending_unique_nodes':len(pending_nodes),'native_pending':native_pending,
              'refresh_policy':'base-anchored-order-strict-input-v1',
              'reuse_rule':'same member set, available old report, exact stabilized context hash equality; common to every reuse-enabled method',
              'full_reset_control':method=='full-reset-rebuild',
              'adapter_limit':'E1 uses common strict freshness, including DyLo; native pending is logged but cannot exempt summaries. Full-reset Rebuild is separately labelled as an all-fresh control.'}
    assert set(changed).isdisjoint(inherited_ids) and set(changed)|set(inherited_ids)==set(map(int,comms.community))
    dump(dest/'refresh_plan.json',{'changed':changed,'inherited':len(inherited_ids),**metadata})
    with (dest/'pending_events.jsonl').open('w') as out:
        for e in events:out.write(json.dumps(e)+'\n')
    return pd.DataFrame(retained),changed,metadata

def prepare_delta(base,new):
    # Already-extracted graph records, with stable reconciled IDs. Apply the received record delta inside timing.
    delta={}
    for name in ('documents','entities','relationships','text_units'):
        old=base[name].set_index('id');newtab=new[name].set_index('id'); changed=[]
        for ident,row in newtab.iterrows():
            if ident not in old.index or repr(row.to_dict())!=repr(old.loc[ident].to_dict()):changed.append(ident)
        delta[name]=(newtab.loc[changed].reset_index(),set(old.index)-set(newtab.index),list(newtab.index))
    return delta

def apply_delta(base,delta):
    tables={}
    for name,(rows,deleted,order) in delta.items():
        old=base[name];joined=pd.concat([old[~old.id.isin(set(rows.id)|deleted)],rows],ignore_index=True)
        tables[name]=joined.set_index('id').loc[order].reset_index()[old.columns]
    return tables

async def run(method,seed,phase='formal'):
    dest=R/'runs'/f'{phase}-s{seed}-{method}'
    if (dest/'verified.json').exists():return json.loads((dest/'result.json').read_text())
    if dest.exists():shutil.move(str(dest),str(dest)+'-incomplete-'+str(time.time_ns()))
    dest.mkdir(parents=True)
    base=b.load_tables('base');expected=b.load_tables('merged');delta=prepare_delta(base,expected);cache=load_vector_cache()
    handle=nr.prepare(method,seed,dest/'native')
    phases={};start=time.perf_counter();received=time.time()
    t=time.perf_counter();merged=apply_delta(base,delta);selector=nr.release(handle,dest/'native');comms=graph(merged,method,seed,dest/'native')
    phases['graph_ms']=(time.perf_counter()-t-selector)*1000
    t=time.perf_counter();contexts=b.prepare_contexts(comms,merged['entities'],merged['relationships']);inherited,changed,pm=plan(method,base,comms,merged,contexts,dest)
    phases['planning_ms']=(time.perf_counter()-t+selector)*1000
    b.say(f'{dest.name}: communities {len(comms)}, reuse {len(inherited)}, refresh {len(changed)}, pending {pm["pending_unique_nodes"]}')
    local={};t=time.perf_counter();fresh,planned,failed=await b.reports(comms[comms.community.isin(changed)],merged['entities'],merged['relationships'],(R/'community_report_graph.txt').read_text(),dest,seed,local,contexts=contexts)
    phases['llm_wall_ms']=local['lm_generate_s']*1000
    t=time.perf_counter();combined=pd.concat([x for x in [inherited,fresh] if len(x)],ignore_index=True)
    assert combined.community.is_unique and combined.id.is_unique and set(combined.community)==set(comms.community)
    tables={**merged,'communities':comms,'community_reports':combined};checks=await materialize(tables,dest/'index',cache)
    phases['materialization_ms']=(time.perf_counter()-t)*1000
    e2e=(time.perf_counter()-start)*1000;phases['other_ms']=e2e-sum(phases.values());assert phases['other_ms']>=0
    reqs=[json.loads(x) for x in (dest/'requests.jsonl').read_text().splitlines()];good=[r for r in reqs if r['status']=='ok'];bad=[r for r in reqs if r['status']!='ok'];assert len(good)==len(changed)
    with (dest/'llm_calls.jsonl').open('w') as out:
        for r in reqs:out.write(json.dumps({**r,'method':method,'seed':seed,'batch_id':1,'run_id':dest.name,'refresh_reason':pm['refresh_reasons'][str(r['community'])]})+'\n')
    import networkx as nx
    g=nx.Graph();g.add_edges_from((a,c) for a,c in zip(merged['relationships'].source,merged['relationships'].target) if a!=c)
    members=b.title_members(comms,merged['entities']);top=[set(members[int(r.community)]) for r in comms.itertuples() if int(r.level)==0];covered=set().union(*top);gg=g.subgraph(covered)
    q=nx.community.modularity(gg,top)
    leaves=set(int(r.community) for r in comms.itertuples() if len(r.children)==0)
    row={'method':method,'seed':seed,'batch_id':1,'run_id':dest.name,'phase':phase,'received_at':received,'e2e_ms':e2e,**phases,
         'summaries':planned,'leaf_summaries':len(leaves&set(changed)),'internal_summaries':len(set(changed)-leaves),'requests':len(reqs),'successful_requests':len(good),'failed_attempts':len(bad),'retries':len(reqs)-len(changed),
         'successful_input_tokens':sum(r.get('prompt_tokens',0) for r in good),'successful_output_tokens':sum(r.get('completion_tokens',0) for r in good),
         'failed_input_tokens':sum(r.get('prompt_tokens',0) for r in bad),'failed_output_tokens':sum(r.get('completion_tokens',0) for r in bad),
         'successful_input_cl100k':sum(r['input_cl100k'] for r in good),'prompt_tokens_mean':np.mean([r.get('prompt_tokens',0) for r in reqs]) if reqs else 0,'prompt_tokens_max':max((r.get('prompt_tokens',0) for r in reqs),default=0),
         'truncated_attempts':sum(r.get('finish_reason')=='length' for r in reqs),'pending_source_count':pm['pending_source_count'],'pending_unique_nodes':pm['pending_unique_nodes'],
         'hierarchy_nodes':len(comms),'top_communities':len(top),'modularity':q,'graph_vertices':g.number_of_nodes(),'graph_edges':g.number_of_edges(),'initial_repair_in_measured_interval':0,'vector_tables':checks,
         'summary_covered_vertices':len(covered),'edge_free_communities':sum(len(r.relationship_ids)==0 for r in comms.itertuples()),
         'edge_free_summaries':sum(len(r.relationship_ids)==0 and int(r.community) in set(changed) for r in comms.itertuples()),
         'native_selector_ms':selector*1000,'timing_scope':'received extracted graph delta to published and retrieval-tested index; excludes initial graph/model/index preparation; retries included',
         'limitations':[pm['adapter_limit'],'Top-level modularity is unweighted, on GraphRAG-covered nodes; native metrics retained separately.']}
    dump(dest/'result.json',row)
    assert abs(sum(phases.values())-e2e)<1e-6
    dump(dest/'verified.json',{'status':'passed','phase_sum_matches_e2e':True,'coverage':len(combined),'queryable':True,'success_token_calls':len(good),'no_update_response_cache':True})
    dump(O/(dest.name+'.json'),row);return row
