"""Actual parallel summary timing; each method runs alone on the same graph."""
import asyncio,json,time,statistics,random
from pathlib import Path
import pandas as pd
import full_update as f
from isolation import Guard
R=Path(__file__).resolve().parent;O=f.OUT;SRC=R.parent/'stable-update-20261016/runs/formal-filtered-s17-dylo'
def load_plan(method):
    dest=R/'plans'/method
    comms=pd.read_parquet(dest/'communities.parquet')
    contexts=[(int(level),pd.DataFrame(records)) for level,records in json.loads((dest/'contexts.json').read_text())]
    assert f.b.context_hashes(contexts)==json.loads((dest/'context_hashes.json').read_text())
    plan=json.loads((dest/'refresh_plan.json').read_text())
    return comms,contexts,plan
def requests(path):
    result=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:pass
    return result
def totals(rows):
    return {status:{'calls':len(z:=[r for r in rows if (r['status']=='ok')==(status=='successful')]),
        'input_tokens':sum(r.get('prompt_tokens',0) for r in z),'output_tokens':sum(r.get('completion_tokens',0) for r in z),
        'total_tokens':sum(r.get('prompt_tokens',0)+r.get('completion_tokens',0) for r in z)} for status in ['successful','failed']}
def source_tables():return {n:pd.read_parquet(SRC/'merged_graph'/f'{n}.parquet') for n in ['documents','entities','relationships','text_units']}
async def tune():
    selected=O/'selected_concurrency.json'
    if selected.exists():return json.loads(selected.read_text())
    merged=source_tables();comms,contexts,plan=load_plan('dylo');target=set(plan['changed'])
    records=[r for _,frame in contexts for r in frame.itertuples() if int(r.community) in target]
    records=sorted(records,key=lambda r:(len(r.context_string),int(r.community)))
    rng=random.Random(17);sample=[]
    for k in range(4):
        bucket=records[len(records)*k//4:len(records)*(k+1)//4]
        sample.extend(int(r.community) for r in rng.sample(bucket,min(4,len(bucket))))
    f.e.dump(O/'tuning_sample.json',{'seed':17,'communities':sample,'selection':'4 actual tasks from each prompt-length quartile, same16 tasks for each concurrency; setup only'})
    results=[]
    for concurrency in [1,4,8]:
        dest=R/'tuning'/f'concurrency-{concurrency}';dest.mkdir(parents=True,exist_ok=True)
        f.b.CONCURRENCY=concurrency;local={};guard=Guard();guard.start();started=time.perf_counter()
        try:
            fresh,planned,failed=await f.b.reports(comms[comms.community.isin(sample)],merged['entities'],merged['relationships'],
                (R/'community_report_graph.txt').read_text(),dest,17,local,contexts=contexts)
            wall=time.perf_counter()-started;isolated=guard.finish(dest/'resource_isolation.json')
            t=totals(requests(dest/'requests.jsonl'))
            row={'concurrency':concurrency,'status':'passed' if isolated and planned==len(sample) and len(fresh)==len(sample) else 'failed',
                 'wall_s':wall,'completed_per_second':len(fresh)/wall,'tokens':t,'sample':len(sample)}
        except BaseException as exc:
            guard.finish(dest/'resource_isolation.json');row={'concurrency':concurrency,'status':'failed','error':str(exc),'tokens':totals(requests(dest/'requests.jsonl'))}
        results.append(row);f.e.dump(O/'tuning.json',results);print('TUNING',row,flush=True)
    good=[r for r in results if r['status']=='passed'];assert good,'No concurrency passed actual validation'
    best=max(good,key=lambda r:r['completed_per_second']);serial=next((r for r in good if r['concurrency']==1),None)
    result={'concurrency':best['concurrency'],'completed_per_second':best['completed_per_second'],
            'measured_speedup_vs_1':best['completed_per_second']/serial['completed_per_second'] if serial else None,
            'basis':'All16 real summaries validated; English schema, source IDs,2048 cap and retries unchanged; no response cache',
            'projected_all7_summary_s':sum(v['refresh'] for v in json.loads((O/'cpu_validation.json').read_text())['checks'].values())/best['completed_per_second']}
    f.e.dump(selected,result);return result
async def run(method,concurrency):
    dest=R/'runs'/method
    if (dest/'verified.json').exists():
        row=json.loads((dest/'result.json').read_text());assert row['concurrency']==concurrency;return row
    if dest.exists():raise RuntimeError('Do not resume a partial wall clock: '+str(dest))
    dest.mkdir(parents=True)
    merged=source_tables();assert f.semantic_hash(merged)==json.loads((O/'cpu_validation.json').read_text())['graph_semantic_sha256']
    comms,contexts,plan=load_plan(method);target=set(plan['changed']);inherited=pd.read_parquet(R/'plans'/method/'inherited.parquet')
    cache=f.e.load_vector_cache();f.b.CONCURRENCY=concurrency;guard=Guard();guard.start();local={}
    f.e.dump(dest/'timing_start.json',{'started_at':time.time(),'planned':len(target),'concurrency':concurrency,'scope':'community resummarization only; extraction and CPU plan excluded'})
    start=time.perf_counter()
    try:
        fresh,planned,failed=await f.b.reports(comms[comms.community.isin(target)],merged['entities'],merged['relationships'],
            (R/'community_report_graph.txt').read_text(),dest,17,local,contexts=contexts)
        summary_wall=time.perf_counter()-start
        calls=requests(dest/'requests.jsonl');t=totals(calls)
        assert len(fresh)==len(target)==t['successful']['calls']
        combined=pd.concat([x for x in [inherited,fresh] if len(x)],ignore_index=True)
        assert combined.community.is_unique and combined.id.is_unique and set(combined.community)==set(comms.community)
        start=time.perf_counter();vectors=await f.e.materialize({**merged,'communities':comms,'community_reports':combined},dest/'index',cache)
        index_wall=time.perf_counter()-start
        isolated=guard.finish(dest/'resource_isolation.json');assert isolated,'GPU contention'
        row={'status':'completed','method':method,'seed':17,'rounds':1,'concurrency':concurrency,'summary_wall_s':summary_wall,
             'model_generation_wall_s':local['lm_generate_s'],'report_finalize_s':local['report_finalize_s'],
             'index_publication_s':index_wall,'queryable':True,'summaries':len(target),'reused':len(inherited),'hierarchy_nodes':len(comms),
             'tokens':t,'refresh_policy':plan['refresh_policy'],'full_reset_control':method=='full-reset-rebuild',
             'graph_semantic_sha256':f.semantic_hash(merged),'timing_scope':'Actual community re-summary wall clock, including retries; excludes extraction, initial index setup, CPU planning and separate index publication',
             'source_extraction_tokens_reference':str(SRC/'extraction_calls.jsonl'),'extraction_calls_in_this_trial':0,
             'no_response_cache':True,'no_prefix_cache':True,'vector_tables':vectors}
        f.e.dump(dest/'result.json',row);f.e.dump(dest/'verified.json',{'status':'passed','complete_summary_coverage':True,'queryable':True,'resource_isolation':'passed'})
        f.e.dump(O/f'{method}.json',row);return row
    except BaseException:
        if not guard.stop.is_set():guard.finish(dest/'resource_isolation.json')
        raise
if __name__=='__main__':
    import sys
    if sys.argv[1]=='tune':asyncio.run(tune())
    else:asyncio.run(run(sys.argv[1],int(sys.argv[2])))
