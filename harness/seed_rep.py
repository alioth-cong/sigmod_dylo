#!/usr/bin/env python3
"""Repeated runs: dispersion of Q / latency / Cost (SIGMOD asks for the number of runs and the variation).

The original implementation hard-coded the detector seed to 123 (driver_std4.cxx:389), so DyLo is
bit-deterministic for a given input -- repeating the same seed always gives zero variance. What must be measured is the effect of **the detector's own randomness**
on the conclusions, hence repetitions with different seeds. driver_seedrep.cxx adds DYLO_SEED; the default 123
is bit-identical to std4 (verified on 42/42 batches).

Reports min-max and IQR for the dispersion columns of Tables 4/5.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import os, shutil, statistics, subprocess, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
R=Path(DYLO_ROOT + ""); D=R/"harness/cpp/dylo"
S=Path(DYLO_SCRATCH)
sys.path.insert(0,str(R/"harness"))
BASE=dict(DYLO_EMIT_HIER="1",DYLO_LAMBDA="1e-1",DYLO_AUDIT_K="8",DYLO_REBUILD_R="1e18",
  DYLO_SIGMA_ONLY="1",DYLO_SIGMA="0.2",DYLO_FINAL_AUDIT="1",DYLO_TOPDOWN="0",
  DYLO_HIER_GAMMA="0.1",DYLO_PRUNE="1",DYLO_PRUNE_MODE="leaf",DYLO_DEFER="0",
  DYLO_LAZY_CERT="1",DYLO_INC_TREE="1",DYLO_LAZY_RECLAIM="1",
  DYLO_RECLAIM="1",DYLO_FAST_RECLAIM="1")
SEEDS=[123,7,42,2024,31337]
def one(a):
    ds,nb,inst,seed=a
    from hier_cost import stream_hier_cost
    d=S/"seedrep"/f"{ds}_{inst}_{seed}"; shutil.rmtree(d,ignore_errors=True)
    d.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ); e.update(BASE); e["DYLO_SEED"]=str(seed)
    rec=dict(ds=ds,inst=inst,seed=seed)
    try:
        subprocess.run([str(D/"dylo_driver_seedrep"),inst,
            str(R/f"data/runs/{ds}/initial.txt"),str(R/f"data/runs/{ds}/inc.txt"),
            str(nb),str(d)],env=e,stdout=subprocess.DEVNULL,
            stderr=open(d/"err","w"),check=True,timeout=21600)
        m=[l.rstrip().split("\t") for l in open(d/"metrics.tsv")][1:]
        hc=stream_hier_cost(d,nb)
        rec.update(status="ok",Q=statistics.mean(float(x[2]) for x in m),
            Qe=float(m[-1][2]),lat=statistics.mean(float(x[1]) for x in m[1:]),
            cost=statistics.mean(y[2] for y in hc) if hc else 0.0)
    except Exception as ex: rec.update(status=type(ex).__name__)
    shutil.rmtree(d,ignore_errors=True)
    return rec
if __name__=="__main__":
    jobs=[]
    for ds in ("facebook","cit-hepth","cit-hepph","epinions","twitter","dblp"):
        nb=json.loads((R/f"data/runs/{ds}/meta.json").read_text())["n_batches"]
        for inst in ("leiden","louvain","slm"):
            for s in SEEDS: jobs.append((ds,nb,inst,s))
    print(f"{len(jobs)} jobs (6 datasets x 3 instances x {len(SEEDS)} seeds), 12 in parallel",flush=True)
    res=[]
    with ProcessPoolExecutor(12) as ex:
        for n,f in enumerate(as_completed([ex.submit(one,j) for j in jobs]),1):
            r=f.result(); res.append(r)
            json.dump(res,open(R/"results/seed_rep.json","w"),indent=1)
            if n%15==0: print(f"  [{n}/{len(jobs)}]",flush=True)
    print("\n-> results/seed_rep.json")
