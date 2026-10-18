def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import json, os, subprocess, threading, time
from pathlib import Path
GPU = os.environ.get('E1_GPU_UUID', 'GPU-REDACTED')

class Guard:

    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.violations = []

    def sample(self):
        p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        unknown = []
        own = []
        if p.returncode:
            unknown = ['nvidia-smi unavailable']
        for line in p.stdout.splitlines():
            (pid, gpu) = [x.strip() for x in line.split(',')]
            if gpu != GPU:
                continue
            try:
                group = Path('/proc/' + pid + '/cgroup').read_text()
            except FileNotFoundError:
                continue
            if any((s in group for s in ['dylo-e123-qwen.service', 'dylo-e123-embedding.service', 'dylo-fast-qwen.service'])):
                own.append(pid)
            else:
                unknown.append(pid)
        row = {'at': time.time(), 'own_gpu_pids': own, 'external_gpu_pids': unknown}
        self.samples.append(row)
        if unknown:
            self.violations.append(row)

    def loop(self):
        while not self.stop.wait(5):
            self.sample()

    def start(self):
        self.sample()
        if self.violations:
            raise RuntimeError('External GPU process present before formal trial')
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def finish(self, path):
        self.stop.set()
        self.thread.join(timeout=10)
        self.sample()
        path.write_text(json.dumps({'status': 'passed' if not self.violations else 'contaminated', 'samples': self.samples, 'violations': self.violations}, indent=2))
        return not self.violations
