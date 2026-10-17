import os,subprocess,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
binary=ROOT/'bin/dylo_std_corrected'
if not binary.exists():binary=ROOT/'dylo_std_corrected'
fixture=ROOT/'small/case0'
for fail in [False,True]:
 with tempfile.TemporaryDirectory(dir=ROOT) as d:
  base=Path(d);ack=base/'ack';ack.mkdir();out=base/'native';out.mkdir()
  env={k:v for k,v in os.environ.items() if not k.startswith('DYLO_')}
  env.update(DYLO_EMIT_HIER='1',DYLO_INIT_PART=str(fixture/'init_part.txt'),DYLO_SKIP_INIT_DETECT='1',DYLO_LAMBDA='.1',DYLO_DEFER='1',DYLO_DEFER_EPS='.05',DYLO_HIER_GAMMA='.1',DYLO_CORRECTED_VERIFY='1',DYLO_SIGMA='1',DYLO_ACK_DIR=str(ack))
  with (base/'log').open('w') as log:
   proc=subprocess.Popen([str(binary),'leiden',str(fixture/'initial.txt'),str(fixture/'inc.txt'),'2',str(out)],env=env,stdout=log,stderr=log)
   for batch in [1,2]:
    deadline=time.monotonic()+15
    while True:
     f=out/'batch_ready.tsv'
     rows=f.read_text().splitlines() if f.exists() else []
     if any(row.startswith(str(batch)+'\t') for row in rows):break
     if proc.poll() is not None:raise RuntimeError((base/'log').read_text())
     if time.monotonic()>deadline:proc.kill();raise TimeoutError('native plan not flushed')
     time.sleep(.01)
    # Native graph stream cannot expose the next batch before materialization.
    assert (out/f'nodes_{batch}_L0.tsv').exists()
    assert (out/f'com_{batch}.txt').exists()
    time.sleep(.03)
    assert not any(row.startswith(str(batch+1)+'\t') for row in (out/'batch_ready.tsv').read_text().splitlines())
    (ack/f'{batch}.{"fail" if fail else "ok"}').write_text('test acknowledgement')
    if fail:break
   rc=proc.wait(timeout=15)
   if fail:
    assert rc!=0 and not any(row.startswith('2\t') for row in (out/'batch_ready.tsv').read_text().splitlines())
   else:assert rc==0
print('PASS native publication gate: wait for report success; failure prevents next batch')
