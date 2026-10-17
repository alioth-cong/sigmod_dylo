"""Repeat the deferred balance and externally invalidated report lifecycle."""
import csv,json,os,subprocess,tempfile
from pathlib import Path
from graphrag_adapter import Report,plan,apply,read_native
ROOT=Path(__file__).resolve().parent
binary=ROOT/'bin/dylo_std_corrected'
if not binary.exists():binary=ROOT/'dylo_std_corrected'
fixture=ROOT/'small/deferred'
records=[]
for external in [False,True]:
 with tempfile.TemporaryDirectory() as d:
  out=Path(d)/'native';out.mkdir()
  env={k:v for k,v in os.environ.items() if not k.startswith('DYLO_')}
  env.update(DYLO_EMIT_HIER='1',DYLO_INIT_PART=str(fixture/'init_part.txt'),
             DYLO_SKIP_INIT_DETECT='1',DYLO_LAMBDA='.1',DYLO_DEFER='1',
             DYLO_DEFER_MODE='quality',DYLO_DEFER_EPS='.05',DYLO_HIER_GAMMA='.1',
             DYLO_TOPDOWN='0',DYLO_CORRECTED_VERIFY='1',DYLO_SIGMA='1',
             DYLO_TREE_DP='1',DYLO_NO_COM='1')
  if external:env['DYLO_CONTENT_DIR']=str(fixture/'content_events')
  with (out/'log').open('w') as log:
   subprocess.run([str(binary),'leiden',str(fixture/'initial.txt'),
                   str(fixture/'inc.txt'),'6',str(out)],env=env,stdout=log,stderr=log,check=True)
  ledger=list(csv.DictReader((out/'ledger.tsv').open(),delimiter='\t'))
  balances=[float(row['pending_end']) for row in ledger]
  assert balances==[51,51,51,51,51,0],balances
  assert all(float(row['booked'])==51 for row in ledger)
  assert float(ledger[-1]['retired'])==51
  assert all(float(row['conservation_error'])==0 for row in ledger)
  assert float(ledger[-1]['cumulative_executed'])==float(ledger[-1]['cumulative_accrual'])==102
  node=(0,0);old={node:Report('old source report',frozenset(str(i) for i in range(51)),
       {str(i):'evidence' for i in range(51)},node,1)}
  members={node:frozenset(str(i) for i in range(1,51))}
  evidence={node:{str(i):'evidence' for i in range(1,51)}}
  mapping={node:node};parents={node:None}
  refresh,deferred=read_native(out,1)
  tasks=plan(old,members,evidence,mapping,refresh,deferred,parents)
  assert len(tasks)==1 and not tasks[0].regenerate
  settled=6
  refresh,deferred=read_native(out,settled)
  tasks=plan(old,members,evidence,mapping,refresh,deferred,parents)
  assert tasks[0].regenerate and node in refresh
  updated=apply(old,tasks,members,evidence,mapping,refresh,
                lambda *args:'new source report',Path(d)/'index.json')
  assert updated[node].native_version==refresh[node]
  records.append({'external_invalidation':external,'status':'pass','balances':balances,
                  'booked':51,'retired':51,'cumulative_accrual':102,'ledger':ledger})
(ROOT/'deferred_validation.json').write_text(json.dumps(records,indent=2))
print('PASS structural deferred lifecycle: book once; epsilon settlement; evidence-only request does not retire pending or open reports')
