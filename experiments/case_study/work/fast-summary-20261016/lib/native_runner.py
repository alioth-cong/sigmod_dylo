def _artifact_path(p):
    from pathlib import Path
    return str(next((r for r in Path(__file__).resolve().parents if (r / 'ARTIFACT_SNAPSHOT.json').exists())) / p)

def _artifact_env(k, v):
    import os
    return os.environ.get(k, v)
import json, os, shutil, subprocess, time
from pathlib import Path
R = Path(__file__).resolve().parent
S = R / 'data/stream'
N = R / 'native'
H = Path(_artifact_path('harness'))

def prepare(method, seed, dest):
    dest.mkdir(parents=True, exist_ok=True)
    env = {k: v for (k, v) in os.environ.items() if not k.startswith('DYLO_')}
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', E1_SEED=str(seed), E1_GATE=str(dest / 'gate'))
    if method == 'dylo':
        raise ValueError('DyLo uses the shared standard driver through all_planning.build')
    elif method in ('batch', 'qca'):
        cp = str(N / 'java') + ':' + (H / 'java/build/.classpath').read_text().strip()
        cmd = [_artifact_env('JAVA', 'java'), '-Xmx16g', '-cp', cp, 'BaselineDriver', method, str(S / 'initial_padded.txt'), str(S / 'baseline_init_part.txt'), str(S / ('incq.txt' if method == 'qca' else 'inc.txt')), '1', str(dest)]
    elif method == 'hit-leiden':
        env['DYLO_P'] = '3'
        cmd = [str(N / 'hit_driver'), str(S / 'initial_padded.txt'), str(S / 'inc.txt'), '1', str(dest), '5.0']
    elif method == 'df-louvain':
        cmd = [str(N / 'df_driver'), 'df', str(S / 'initial_padded.txt'), str(S / 'inc.txt'), '1', str(dest), '1']
    else:
        return None
    log = (dest / 'native.log').open('w')
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=dest)
    start = time.perf_counter()
    while not (dest / 'gate.ready').exists():
        if p.poll() is not None:
            raise RuntimeError(f'{method} initialization failed; inspect {dest}/native.log')
        if time.perf_counter() - start > 300:
            p.terminate()
            raise RuntimeError('Native initialization timeout')
        time.sleep(0.01)
    shutil.copy2(S / 'id2name.json', dest / 'id2name.json')
    (dest / 'command.json').write_text(json.dumps({'command': cmd, 'environment': {k: v for (k, v) in env.items() if k.startswith(('DYLO_', 'E1_'))}, 'initialization_s_excluded': time.perf_counter() - start}, indent=2))
    return (p, log)

def release(handle, dest):
    if handle is None:
        return 0.0
    (p, log) = handle
    (dest / 'gate.go').write_text('go')
    p.wait(timeout=600)
    log.close()
    if p.returncode:
        raise RuntimeError(f'Native update failed: {dest}')
    f = dest / 'phase_times.json'
    return json.loads(f.read_text())['selector_ms'] / 1000 if f.exists() else 0.0
