"""Authoritative cost scorer for the corrected std experiment suite."""
import argparse,csv,json,statistics,math
from pathlib import Path

def structural_rows(directory,n_batches=None):
 directory=Path(directory)
 with (directory/'runtime_config.tsv').open() as f:cfg={r['key']:r['value'] for r in csv.DictReader(f,delimiter='\t')}
 if cfg.get('structural_cost_scope')!='1':raise ValueError('explicit structural scope required')
 with (directory/'ledger.tsv').open() as f:ledger=list(csv.DictReader(f,delimiter='\t'))
 if n_batches is not None and len(ledger)!=n_batches:raise ValueError('incomplete stream')
 rows=[]
 for expected,r in enumerate(ledger,1):
  if int(r['batch'])!=expected:raise ValueError('missing or duplicate batch')
  values={k:float(r[k]) for k in ['executed_raw','pending_start','pending_end','accrual_raw','snapshot_scale','conservation_error']}
  if any(not math.isfinite(x) for x in values.values()):raise ValueError('nonfinite ledger')
  if abs(values['conservation_error'])>1e-5:raise ValueError('unbalanced structural ledger')
  a=values['executed_raw']+values['pending_end']-values['pending_start']
  if abs(a-values['accrual_raw'])>1e-7*max(1,abs(a)):raise ValueError('incorrect accrual')
  scale=values['snapshot_scale'];normalized=a/scale if scale>0 else 0
  rows.append((int(r['batch']),a,normalized))
 return rows

def stream_hier_cost(out_dir,n_batches):
 """Legacy tuple interface, now based on structural accrual, never partition churn."""
 return structural_rows(out_dir,n_batches)

def summary(directory):
 rs=structural_rows(directory)
 return {'scope':'structural_accrual','batches':len(rs),'cost_mean':statistics.mean(r[2] for r in rs),'cost_raw_sum':sum(r[1] for r in rs),'normalization':'current snapshot report-weight sum; initial report creation excluded'}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('directory',type=Path);args=a.parse_args();print(json.dumps(summary(args.directory),ensure_ascii=False,indent=2))
