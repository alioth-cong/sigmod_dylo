"""Run the corrected std with the explicit paper main-table preset.
Caller DYLO_* overrides are preserved and recorded by the native driver.
"""
import os,sys,subprocess
from pathlib import Path
from validate_release import BASE
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 if len(sys.argv)<6:raise SystemExit('usage: launch_std.py {louvain|slm|leiden} initial.txt inc.txt n_batches output_dir [threads_ignored]')
 env=dict(os.environ)
 for k,v in {**BASE,'DYLO_NO_COM':'0','DYLO_COST_SCOPE':'structural'}.items():env.setdefault(k,v)
 Path(sys.argv[5]).mkdir(parents=True,exist_ok=True)
 raise SystemExit(subprocess.call([str(ROOT/'bin/dylo_std_corrected')]+sys.argv[1:],env=env))
