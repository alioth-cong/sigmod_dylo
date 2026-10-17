"""Distinguish structural refresh from an unchanged community receiving edges."""
import os,csv,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BINARY=ROOT/'bin/dylo_std_corrected'
with tempfile.TemporaryDirectory() as d:
 base=Path(d);fixture=base/'data';fixture.mkdir()
 edges=[(u,v) for start in [0,4] for u in range(start,start+4) for v in range(u+1,start+4)]+[(0,4)]
 (fixture/'initial.txt').write_text(''.join(f'{u}\t{v}\n' for u,v in edges))
 (fixture/'init_part.txt').write_text('0\t1\t2\t3\n4\t5\t6\t7\n')
 (fixture/'inc_1.txt').write_text('0\t1\t1\n4\t5\t1\n0\t4\t1\n')
 (fixture/'inc_2.txt').write_text('8\t2\t2\n')
 content=fixture/'content';content.mkdir();(content/'content_1.txt').write_text('0 0\n')
 env={k:v for k,v in os.environ.items() if not k.startswith('DYLO_')}
 env.update(DYLO_EMIT_HIER='1',DYLO_DEFER='1',DYLO_DEFER_EPS='.05',DYLO_HIER_GAMMA='.1',DYLO_TOPDOWN='0',DYLO_LAMBDA='1000000',DYLO_INIT_PART=str(fixture/'init_part.txt'),DYLO_SKIP_INIT_DETECT='1',DYLO_CORRECTED_VERIFY='1',DYLO_SIGMA='1',DYLO_CONTENT_DIR=str(content))
 def run(binary,label):
  out=base/label;out.mkdir()
  with (out/'log').open('w') as log:
   subprocess.run([str(binary),'leiden',str(fixture/'initial.txt'),str(fixture/'inc.txt'),'2',str(out)],env=env,stdout=log,stderr=log,check=True)
  with (out/'ledger.tsv').open() as f:ledger=list(csv.DictReader(f,delimiter='\t'))
  def partition(k):return {frozenset(map(int,line.split())) for line in (out/f'com_{k}.txt').read_text().splitlines() if line.strip()}
  assert partition(0)==partition(1),'test must actually keep community membership unchanged'
  return out,ledger
 out,ledger=run(BINARY,'structural')
 assert float(ledger[0]['executed_raw'])==float(ledger[0]['pending_end'])==float(ledger[0]['accrual_raw'])==0,ledger[0]
 assert float(ledger[1]['executed_raw'])>0,'new report membership must be priced'
 with (out/'refresh_plan.tsv').open() as f:plan=list(csv.DictReader(f,delimiter='\t'))
 assert not any(int(r['batch'])==1 for r in plan)
 assert any(int(r['batch'])==2 and r['reason']=='structural_arrival' for r in plan)
 legacy=ROOT.parent/'std_corrected/dylo_std_corrected'
 if legacy.exists():
  _,previous=run(legacy,'previous_evidence_scope')
  assert float(previous[0]['executed_raw'])>0
 print('PASS cost scope: unchanged membership/edge-only/evidence-only => zero; newly created structural report => positive')
