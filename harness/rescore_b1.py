#!/usr/bin/env python3
"""Re-score every saved per-batch baseline partition under both the old and the b1 weights (pure CPU post-processing)."""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
H = Path(DYLO_ROOT + "/harness"); sys.path.insert(0, str(H))
RUNS = Path(DYLO_ROOT + "/data/runs")
HIER = {"dylo-louvain", "dylo-slm", "dylo-leiden", "leiden-hier", "louvain-hier", "hit-leiden"}

def nb_of(out):
    k = 0
    while (out / f"com_{k}.txt").exists() or (out / f"com_{k}_L0.txt").exists(): k += 1
    return k - 1

def one(ds, m):
    from hier_cost_w import stream, summarize
    out = RUNS / ds / f"out_{m}"; t0 = time.time(); nb = nb_of(out)
    hier = m in HIER and (out / "com_1_L0.txt").exists()
    try:
        s = summarize(stream(out, nb, hier))
    except Exception as e:
        return dict(dataset=ds, method=m, status=f"err {type(e).__name__} {e}"[:120])
    return dict(dataset=ds, method=m, kind="hier" if hier else "flat", status="ok" if s else "empty",
                sec=round(time.time()-t0, 1), **s)

if __name__ == "__main__":
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    jobs = [(d.name, o.name[4:]) for d in sorted(RUNS.iterdir()) if d.is_dir() and (not only or d.name in only)
            for o in sorted(d.glob("out_*"))]
    res = []
    with ProcessPoolExecutor(64) as ex:
        fs = {ex.submit(one, *j): j for j in jobs}
        for f in as_completed(fs):
            r = f.result(); res.append(r)
            print(r.get("dataset"), r.get("method"), r.get("status"), round(r.get("cost_old", -1), 4), round(r.get("cost_b1", -1), 4), r.get("sec"), flush=True)
    cols = ["dataset","method","kind","status","batches","cost_old","cost_b1","calls","calls_norm","nodes","sec"]
    with open(sys.argv[1], "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in sorted(res, key=lambda r: (r["dataset"], r["method"])): w.writerow(r)
