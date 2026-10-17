"""Monitor the explicitly declared model GPU; never stop foreign processes."""
import json,os,subprocess,threading,time
from pathlib import Path
class Guard:
    def __init__(self):
        self.gpu=os.environ.get('E1_GPU_UUID')
        self.owned={int(x) for x in os.environ.get('E1_OWNED_GPU_PIDS','').split(',') if x}
        self.stop=threading.Event();self.samples=[];self.violations=[]
    def sample(self):
        if not self.gpu or not self.owned:raise RuntimeError('Declare E1_GPU_UUID and E1_OWNED_GPU_PIDS before a measured replay')
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        active={int(line.split(',')[0].strip()) for line in p.stdout.splitlines() if line.split(',')[1].strip()==self.gpu}
        foreign=sorted(active-self.owned);row={'at':time.time(),'gpu_uuid':self.gpu,'compute_pids':sorted(active),'external_gpu_pids':foreign}
        self.samples.append(row)
        if foreign:self.violations.append(row)
    def loop(self):
        while not self.stop.wait(5):
            try:self.sample()
            except Exception as exc:self.violations.append({'monitor_error':str(exc)});return
    def start(self):
        self.sample()
        if self.violations:raise RuntimeError('Foreign GPU process detected')
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def finish(self,path):
        self.stop.set();self.thread.join(timeout=10);self.sample()
        Path(path).write_text(json.dumps({'status':'passed' if not self.violations else 'contaminated','samples':self.samples,'violations':self.violations},indent=2))
        return not self.violations
