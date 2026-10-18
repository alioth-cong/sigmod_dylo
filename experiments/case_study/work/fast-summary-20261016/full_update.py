"""Shared graph-input helpers from the measured case study."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/fast-summary-20261016'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP','True')
sys.path.insert(0,str(ROOT/'lib'))
import benchmark as b
import engine as e
import native_runner as nr
e.R=b.ROOT=nr.R=ROOT
nr.N=ROOT/'native'
def edges(tables):
    return {tuple(sorted((a, c))) for (a, c) in zip(tables['relationships'].source, tables['relationships'].target) if a != c}

def stream_for(base, merged, dest):
    s = dest / 'data/stream'
    s.mkdir(parents=True)
    (old, new) = (edges(base), edges(merged))
    assert old <= new
    oldnames = sorted({x for pair in old for x in pair})
    names = oldnames + sorted({x for pair in new for x in pair} - set(oldnames))
    ids = {n: i for (i, n) in enumerate(names)}
    text = ''.join((f'{ids[a]}\t{ids[c]}\t1.0\n' for (a, c) in sorted(old)))
    (s / 'initial.txt').write_text(text)
    (s / 'initial_padded.txt').write_text(text + f'{len(ids) - 1}\t{len(ids) - 1}\t0.0\n')
    inc = ''.join((f'{ids[a]}\t{ids[c]}\t1.0\n' for (a, c) in sorted(new - old)))
    (s / 'inc.txt').write_text('')
    (s / 'incq.txt').write_text('')
    (s / 'inc_1.txt').write_text(inc)
    (s / 'incq_1.txt').write_text(inc.replace('\t1.0\n', '\t1\n'))
    e.dump(s / 'id2name.json', {str(i): n for (n, i) in ids.items()})
    members = b.title_members(base['communities'], base['entities'])
    (assigned, groups) = (set(), [])
    for row in sorted(base['communities'].itertuples(), key=lambda x: -x.level):
        group = members[int(row.community)] - assigned & set(ids)
        if group:
            groups.append(sorted((ids[n] for n in group)))
            assigned.update(group)
    groups.extend(([ids[n]] for n in names if n not in assigned))
    (s / 'init_part.txt').write_text(''.join(('\t'.join(map(str, g)) + '\n' for g in groups)))
    present = {int(x) for line in (s / 'initial_padded.txt').read_text().splitlines() for x in line.split()[:2]}
    jgroups = [[x for x in g if x in present] for g in groups]
    covered = [x for g in jgroups for x in g]
    assert len(covered) == len(set(covered)) and set(covered) == present
    (s / 'baseline_init_part.txt').write_text(''.join(('\t'.join(map(str, g)) + '\n' for g in jgroups if g)))
    nr.S = s
    return {'initial_edges': len(old), 'added_edges': len(new - old), 'vertices': len(names)}

def semantic_hash(tables):
    value = {'entities': sorted(((str(r.title), str(r.type), str(r.description)) for r in tables['entities'].itertuples())), 'relationships': sorted(((str(r.source), str(r.target), str(r.description), float(r.weight)) for r in tables['relationships'].itertuples()))}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
