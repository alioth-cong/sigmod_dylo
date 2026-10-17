"""Recompute measured maintenance and answer preferences without model calls."""
from pathlib import Path
import csv,json,math,collections
from unpack import unpack
ROOT=Path(__file__).resolve().parent
ARMS=['dylo','batch','rebuild','hit-leiden','dynamo','qca','df-louvain','ds-louvain']
DIMS=['comprehensiveness','diversity','empowerment','directness']
def read(p):return json.loads(p.read_text())
def ledger(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def audit():
    unpack()
    table={r['method']:r for r in csv.DictReader((ROOT/'results/all_results.csv').open())}
    assert set(table)==set(ARMS)
    graphs=set();summary=[]
    for arm in ARMS:
        group='table5-std4-20261016' if arm in ARMS[:3] else 'table5-all-baselines-20261016'
        dest=ROOT/'observed/work'/group/'runs'/arm
        row=read(dest/'result.json');assert row['status']=='completed' and row['queryable']
        assert read(dest/'verified.json')['resource_isolation']=='passed'
        graphs.add(row['graph_semantic_sha256'])
        records=ledger(dest/'delta_summaries/requests.jsonl')+ledger(dest/'merged_refresh/requests.jsonl')
        good=[r for r in records if r['status']=='ok'];failed=[r for r in records if r['status']!='ok']
        assert len({(r['community'],r['level']) for r in ledger(dest/'merged_refresh/requests.jsonl') if r['status']=='ok'})==len([r for r in ledger(dest/'merged_refresh/requests.jsonl') if r['status']=='ok'])
        values={'net_update_tasks':len(good),'successful_input_tokens':sum(r['prompt_tokens'] for r in good),
                'successful_output_tokens':sum(r['completion_tokens'] for r in good),
                'successful_total_tokens':sum(r['prompt_tokens']+r['completion_tokens'] for r in good),
                'failed_calls':len(failed),'failed_tokens':sum(r.get('prompt_tokens',0)+r.get('completion_tokens',0) for r in failed)}
        for name,value in values.items():assert value==int(table[arm][name]),(arm,name,value,table[arm][name])
        t=row['net_update_maintenance_wall_s'];parts=row['phase_times_s']
        assert math.isclose(sum(parts.values()),t,abs_tol=.002)
        phase_summary=parts['delta_summary_s']+parts['merged_refresh_s']
        ts=row['net_update_summary_wall_s']
        assert 0 <= ts-phase_summary < .1, (arm, 'summary wrapper overhead', ts-phase_summary)
        assert math.isclose(ts,float(table[arm]['net_summary_s']),abs_tol=.002)
        assert math.isclose(t,float(table[arm]['net_graph_to_queryable_s']),abs_tol=.002)
        assert math.isclose(100*phase_summary/t,float(table[arm]['summary_percent']),abs_tol=.00001)
        assert math.isclose(row['maintenance_wall_including_initial_repair_s']-row['initial_missing_repair_wall_s'],t,abs_tol=.002)
        summary.append(dict(method=arm,total_s=t,summary_s=ts,**values))
    assert len(graphs)==1
    q=ROOT/'observed/outputs/table5-all-baselines-20261016/quality'
    for arm in ARMS:
        answers=[read(q/'answers'/arm/f'q{i:03d}.json') for i in range(60)]
        assert all(a['status']=='ok' and a['arm']==arm and a['qid']==i and a['answer'].strip() for i,a in enumerate(answers))
    import numpy as np
    published={(x['baseline'],x['dimension']):x for x in read(ROOT/'results/quality/quality-full.json')['rows']}
    checked=0;oriented=0
    for arm in ARMS[1:]:
        verdicts=read(q/f'verdicts-full-{arm}.json');assert len(verdicts)==480
        assert len({(x['qid'],x['dimension'],x['orientation']) for x in verdicts})==480
        for dim in DIMS:
            values=[]
            for qid in range(60):
                pair=[x for x in verdicts if x['dimension']==dim and x['qid']==qid]
                assert len(pair)==2 and {x['orientation'] for x in pair}=={0,1}
                scores=[]
                for x in pair:
                    v=x['verdict'];assert v['status']=='ok' and v['winner'] in [0,1,2]
                    score=.5 if v['winner']==0 else float(v['winner']==(1 if x['dylo_first'] else 2))
                    assert score==x['dylo_score'];scores.append(score);oriented+=1
                values.append(sum(scores)/2)
            v=np.array(values);rng=np.random.default_rng(17)
            means=v[rng.integers(0,60,size=(10000,60))].mean(axis=1)
            ci=100*np.quantile(means,[.025,.975]);p=published[(arm,dim)]
            assert abs(100*float(v.mean())-p['dylo_preference_score'])<1e-9
            assert np.allclose(ci,p['bootstrap95_percent'],atol=1e-8,rtol=0)
            checked+=1
    result={'status':'passed','maintenance_methods':8,'shared_graphs':len(graphs),
            'new_answers_checked':480,'oriented_judgments_recomputed':oriented,
            'preference_and_bootstrap_rows':checked,'new_model_calls':0,
            'timing_scope':'graph-ready maintenance; common extraction and initial repairs excluded',
            'raw_payload_scope':'all attempt ledgers; representative and failed maintenance API payloads; all final answers and oriented verdicts'}
    target=ROOT/'verification';target.mkdir(exist_ok=True)
    (target/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (target/'maintenance-recomputed.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':audit()
