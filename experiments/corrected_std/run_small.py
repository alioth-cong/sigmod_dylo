import os,subprocess,json,csv
from pathlib import Path
root=Path(__file__).resolve().parent
binary=root/'bin/dylo_std_corrected'
if not binary.exists():binary=root/'dylo_std_corrected'
results=[]
for case in range(12):
 p=root/'small'/f'case{case}'
 for inst in ['louvain','slm','leiden']:
  out=p/inst;out.mkdir(exist_ok=True)
  env={k:v for k,v in os.environ.items() if not k.startswith('DYLO_')}
  env.update(DYLO_EMIT_HIER='1',DYLO_LAMBDA='0.1',DYLO_DEFER='1',DYLO_DEFER_EPS='0.05',DYLO_DEFER_MODE='quality',DYLO_TREE_DP='1',DYLO_HIER_GAMMA='0.1',DYLO_TOPDOWN='0',DYLO_CORRECTED_VERIFY='1',DYLO_INIT_PART=str(p/'init_part.txt'),DYLO_SKIP_INIT_DETECT='1',DYLO_RECLAIM='1',DYLO_FAST_RECLAIM='1',DYLO_SIGMA_ONLY='1',DYLO_SIGMA='.2',DYLO_AUDIT_K='2',DYLO_FINAL_AUDIT='1')
  if case>=8:env['DYLO_COST_UNITS']='normalized'
  with (out/'stderr.log').open('w') as f:
   r=subprocess.run([str(binary),inst,str(p/'initial.txt'),str(p/'inc.txt'),'5',str(out)],env=env,stdout=f,stderr=f)
  if r.returncode:
   print('FAIL',case,inst,(out/'stderr.log').read_text()[-3000:]);raise SystemExit(1)
  ledger=list(csv.DictReader((out/'ledger.tsv').open(),delimiter='\t'))
  assert len(ledger)==5
  assert all(abs(float(x['conservation_error']))<1e-7 for x in ledger)
  results.append({'case':case,'instance':inst,'batches':5,'status':'pass','units':env.get('DYLO_COST_UNITS','std')})
  print('PASS',case,inst,flush=True)
(root/'small_results.json').write_text(json.dumps(results,indent=2))
print('PASS streamed integration',len(results))
