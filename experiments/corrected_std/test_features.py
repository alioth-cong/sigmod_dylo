import csv,hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BINARY=ROOT/'bin/dylo_std_corrected'
BASE=dict(DYLO_EMIT_HIER='1',DYLO_TOPDOWN='0',DYLO_DEFER='1',DYLO_DEFER_MODE='quality',DYLO_DEFER_EPS='.05',DYLO_TREE_DP='1',DYLO_LAMBDA='0',DYLO_HIER_GAMMA='.1',DYLO_CORRECTED_VERIFY='1',DYLO_VALIDATE_TREE='1',DYLO_SIGMA='1')
def rows(p):
 with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
results=[]
with tempfile.TemporaryDirectory() as d:
 base=Path(d);fixture=ROOT/'small/case0'
 variants=[('dp',{}),('cut',{'DYLO_TREE_DP':'0'}),('cut_explicit',{'DYLO_SEL_MODE':'cut'}),('greedy',{'DYLO_SEL_MODE':'greedy'}),('indep',{'DYLO_SEL_MODE':'indep'}),('greedy_legacy',{'DYLO_GREEDY':'1'}),('defer_off',{'DYLO_DEFER':'0'}),('greedy_defer_off',{'DYLO_DEFER':'0','DYLO_SEL_MODE':'greedy'}),('indep_defer_off',{'DYLO_DEFER':'0','DYLO_SEL_MODE':'indep'}),('cert_off',{'DYLO_LAZY_CERT':'0'}),('tree_off',{'DYLO_INC_TREE':'0'}),('prune_off',{'DYLO_PRUNE':'0'}),('source_off',{'DYLO_SRC_CHARGE':'0'}),('gain_uncached',{'DYLO_GAIN_CACHE':'0'}),('shadow',{'DYLO_SEL_SHADOW':'1'}),('b1',{'DYLO_WMODEL':'b1'}),('subtree',{'DYLO_WMODEL':'subtree'}),('rint4',{'DYLO_R_INT':'4'}),('alpha2',{'DYLO_ALPHA':'2'})]
 digests={}
 for inst in ['louvain','slm','leiden']:
  for name,extra in variants:
   out=base/f'{inst}-{name}';out.mkdir()
   env={k:v for k,v in os.environ.items() if not k.startswith('DYLO_')};env.update(BASE,**extra,DYLO_INIT_PART=str(fixture/'init_part.txt'),DYLO_SKIP_INIT_DETECT='1')
   with (out/'log').open('w') as log:p=subprocess.run([str(BINARY),inst,str(fixture/'initial.txt'),str(fixture/'inc.txt'),'5',str(out)],env=env,stdout=log,stderr=log)
   if p.returncode:raise RuntimeError(name+' '+(out/'log').read_text()[-3000:])
   config={r['key']:r['value'] for r in rows(out/'runtime_config.tsv')}
   ledger=rows(out/'ledger.tsv');assert len(ledger)==5 and all(abs(float(r['conservation_error']))<1e-6 for r in ledger)
   if extra.get('DYLO_DEFER')=='0' or extra.get('DYLO_SRC_CHARGE')=='0':assert all(float(r['pending_end'])==0 for r in ledger)
   calls=rows(out/'selector_stats.tsv');assert calls,(inst,name,'no actual selector test')
   expected='cut' if name in ['cut','cut_explicit','defer_off'] else ('greedy' if name in ['greedy','greedy_legacy','greedy_defer_off'] else ('indep' if name in ['indep','indep_defer_off'] else 'dp'))
   assert all(r['solver']==expected for r in calls),(name,calls)
   assert any(int(r['eligible'])>0 for r in calls),(name,'solver must receive actual eligible actions')
   digest=hashlib.sha256((out/'com_5.txt').read_bytes()).hexdigest();digests[inst,name]=digest
   if name in ['cut','cut_explicit','cert_off','tree_off','prune_off','gain_uncached','shadow']:assert digest==digests[inst,'dp'],(inst,name,'optimization changed final partition')
   results.append({'instance':inst,'variant':name,'status':'pass','solver':expected,'calls':len(calls)})
 # Distinct actual seed parameters, cold detection, and exact replay.
 for seed in [7,42,123,7]:
  out=base/f'seed-{seed}-{len(results)}';out.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('DYLO_')};env.update(BASE,DYLO_SEED=str(seed))
  with (out/'log').open('w') as log:subprocess.run([str(BINARY),'leiden',str(fixture/'initial.txt'),str(fixture/'inc.txt'),'5',str(out)],env=env,stdout=log,stderr=log,check=True)
  c={r['key']:r['value'] for r in rows(out/'runtime_config.tsv')};assert int(c['detector_seed'])==int(c['hierarchy_seed'])==seed
  digest=hashlib.sha256((out/'com_0.txt').read_bytes()+(out/'com_5.txt').read_bytes()).hexdigest()
  results.append({'seed':seed,'status':'pass','partition_hash':digest})
 assert results[-1]['partition_hash']==results[-4]['partition_hash']
(ROOT/'feature_results.json').write_text(json.dumps(results,indent=2));print('PASS compatibility paths',len(results))
