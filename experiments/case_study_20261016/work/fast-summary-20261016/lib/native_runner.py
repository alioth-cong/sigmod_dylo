# Path-portable copy; original and published SHA-256 in provenance.json.
def _artifact_path(relative):
    from pathlib import Path as _P
    import os as _O
    root = next((p for p in _P(__file__).resolve().parents if (p / 'ARTIFACT_SNAPSHOT.json').exists()))
    return str(root / relative)

def _artifact_env(name, default):
    import os as _O
    return _O.environ.get(name, default)
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
        env.update(DYLO_EMIT_HIER='1', DYLO_LAMBDA='0.1', DYLO_AUDIT_K='8', DYLO_REBUILD_R='1e18', DYLO_SIGMA_ONLY='1', DYLO_SIGMA='0.2', DYLO_FINAL_AUDIT='1', DYLO_TOPDOWN='0', DYLO_HIER_GAMMA='5.0', DYLO_PRUNE='1', DYLO_PRUNE_MODE='leaf', DYLO_DEFER='1', DYLO_DEFER_EPS='0.05', DYLO_DEFER_MODE='quality', DYLO_TREE_DP='1', DYLO_LAZY_CERT='1', DYLO_INC_TREE='1', DYLO_LAZY_RECLAIM='1', DYLO_INIT_PART=str(S / 'init_part.txt'), DYLO_BENCHMARK_SOURCE_STATE=str(dest / 'source_state.tsv'))
        cmd = [str(N / 'dylo_driver'), 'leiden', str(S / 'initial.txt'), str(S / 'inc.txt'), '1', str(dest), '1']
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
