"""Recompute paper maintenance totals and quality CIs from saved observations; no model calls."""
from pathlib import Path
import json,tarfile,math,re
import numpy as np
from bundle_io import open_bundle
R=Path(__file__).resolve().parent

def members(name,accept):
 out={}
 with open_bundle(name) as t:
  for m in t:
   if m.isfile() and accept(m.name):out[m.name]=t.extractfile(m).read()
 return out

def audit():
 result=json.loads((R/'results/qwen32-all-baselines-20261017/all_results.json').read_text())
 paper=(R.parents[1]/'paper/exp8-mhrag-e2e.tex').read_text()
 quality=json.loads((R/'results/qwen32-all-baselines-20261017/quality_results.json').read_text())['rows']
 graph=set();initial=set();checks=[]
 for row in result['rows']:
  arm=row['method'];prefix=f'observed/runs/{arm}/'
  files=members('maintenance-'+arm+'.tar.gz',lambda s:s in [prefix+'result.json',prefix+'verified.json',prefix+'delta_summaries/requests.jsonl',prefix+'merged_refresh/requests.jsonl'])
  d=json.loads(files[prefix+'result.json']);assert json.loads(files[prefix+'verified.json'])['status']=='passed'
  calls=[json.loads(line) for s,b in files.items() if s.endswith('requests.jsonl') for line in b.splitlines() if line.strip()]
  ok=[c for c in calls if c['status']=='ok'];bad=[c for c in calls if c['status']!='ok']
  tokens=sum(c['prompt_tokens']+c['completion_tokens'] for c in ok)
  assert tokens==row['successful_total_tokens'] and len(ok)==row['reports']
  assert len(bad)==row['failed_calls']
  assert math.isclose(d['net_update_maintenance_wall_s'],row['maintenance_s'],abs_tol=1e-6)
  assert math.isclose(sum(d['phase_times_s'].values()),row['maintenance_s'],abs_tol=.001)
  for x in [f"{row['maintenance_s']:.1f}",f"{row['summary_s']:.1f}",f"{tokens/1000:,.1f}"]:assert x in paper,(arm,x,'missing in paper')
  graph.add(d['graph_semantic_sha256']);initial.add(d['initial_reports_sha256'])
  checks.append(dict(method=arm,calls=len(ok),successful_tokens=tokens,maintenance_s=row['maintenance_s']))
 assert len(graph)==len(initial)==1
 files=members('quality.tar.gz',lambda s: '/answers/' in s or '/verdicts-full-' in s or s.endswith('/judge_calibration.json'))
 assert json.loads(files['observed/quality/judge_calibration.json'])['status']=='passed'
 methods={r['method'] for r in result['rows']}
 for arm in methods:
  answers=[json.loads(b) for s,b in files.items() if f'/answers/{arm}/' in s]
  assert len(answers)==60 and {a['qid'] for a in answers}==set(range(60)) and all(a['status']=='ok' and a['answer'].strip() for a in answers)
 judged=0
 for arm in methods-{'dylo'}:
  verdicts=json.loads(files[f'observed/quality/verdicts-full-{arm}.json']);assert len(verdicts)==480
  assert len({(r['qid'],r['dimension'],r['orientation']) for r in verdicts})==480
  judged+=len(verdicts)
  for dim in ['comprehensiveness','diversity','empowerment','directness']:
   vals=[]
   for qid in range(60):
    pair=[v for v in verdicts if v['qid']==qid and v['dimension']==dim];assert len(pair)==2
    scores=[]
    for v in pair:
     assert v['verdict']['status']=='ok';w=v['verdict']['winner'];s=.5 if w==0 else float(w==(1 if v['dylo_first'] else 2));assert s==v['dylo_score'];scores.append(s)
    vals.append(sum(scores)/2)
   v=np.array(vals);rng=np.random.default_rng(17);ci=100*np.quantile(v[rng.integers(0,60,size=(10000,60))].mean(axis=1),[.025,.975]);score=100*v.mean()
   target=next(x for x in quality if x['baseline']==arm and x['dimension']==dim)
   assert math.isclose(score,target['dylo_preference_score'],abs_tol=1e-8)
   assert np.allclose(ci,target['bootstrap95_percent'],atol=1e-8)
   assert f'{score:.2f}' in paper,(arm,dim,'paper score mismatch')
 report=dict(status='passed',methods=8,answers=480,oriented_judgments=judged,quality_dimensions=28,maintenance=checks,paper_values_match=True,quality_scores_and_CIs_recomputed=True)
 out=R/'verification';out.mkdir(exist_ok=True);(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':audit()
