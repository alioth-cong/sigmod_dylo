#!/usr/bin/env python3
"""Controlled ablation of DEFER on/off: what does the asymmetric rule actually buy?

The paper's argument (4c-lazy.tex:25 / driver_std4.cxx:1218):
  deferred sources -> each action has one anchor -> the exact selector collapses from max-weight-closure/min-cut
  to a linear tree DP -> faster.
But a first measurement showed DEFER=1 to be slower on Epinions/Flickr, with consistently higher cost.
Latency is noisy (milliseconds per batch), so runs are repeated over 5 seeds here, together with the attributable intermediate quantities
measured alongside:
  dp_frac   share of batches using the tree DP (the rest use min-cut)
  rounds    mean rounds per batch
  peak_rss  peak memory
  K_T       unsettled source-side redundancy (0 by construction when DEFER=0)
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import os, shutil, statistics, subprocess, sys, json, time, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
R=Path(DYLO_ROOT + ""); D=R/"harness/cpp/dylo"
S=Path(DYLO_ROOT + "/tmpruns/scratch")
sys.path.insert(0,str(R/"harness"))
BASE=dict(DYLO_EMIT_HIER="1",DYLO_LAMBDA="1e-1",DYLO_AUDIT_K="8",DYLO_REBUILD_R="1e18",
  DYLO_SIGMA_ONLY="1",DYLO_SIGMA="0.2",DYLO_FINAL_AUDIT="1",DYLO_TOPDOWN="0",
  DYLO_HIER_GAMMA="0.1",DYLO_PRUNE="1",DYLO_PRUNE_MODE="leaf",DYLO_LAZY_CERT="1",
  DYLO_INC_TREE="1",DYLO_LAZY_RECLAIM="1",DYLO_RECLAIM="1",DYLO_FAST_RECLAIM="1",
  DYLO_SRC_CHARGE="1")
def one(a):
    ds,nb,defer,seed=a
    from hier_cost import stream_hier_cost
    d=S/"deferab"/f"{ds}_d{defer}_s{seed}"; shutil.rmtree(d,ignore_errors=True)
    d.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ); e.update(BASE)
    e.update(DYLO_DEFER=str(defer),DYLO_SEED=str(seed))
    rec=dict(ds=ds,defer=defer,seed=seed)
    try:
        t0=time.time()
        subprocess.run(["/usr/bin/time","-v","-o",str(d/"mem"),
            str(D/"dylo_driver_seedrep"),"leiden",
            str(R/f"data/runs/{ds}/initial.txt"),str(R/f"data/runs/{ds}/inc.txt"),
            str(nb),str(d)],env=e,stdout=subprocess.DEVNULL,
            stderr=open(d/"err","w"),check=True,timeout=21600)
        wall=time.time()-t0
        m=[l.rstrip().split("\t") for l in open(d/"metrics.tsv")][1:]
        hc=stream_hier_cost(d,nb)
        dp=[x.rstrip().split("\t") for x in open(d/"dp.tsv")][1:]
        dp=[x for x in dp if len(x)>=5]
        dpn=sum(1 for x in dp if float(x[3])>0)
        rss=None
        for l in open(d/"mem"):
            if "Maximum resident" in l: rss=int(l.split()[-1])/1048576
        err=open(d/"err").read()
        g=lambda k:(int(re.search(k+r"=(\d+)",err).group(1)) if re.search(k+r"=(\d+)",err) else 0)
        rec.update(status="ok",
            lat=statistics.mean(float(x[1]) for x in m[1:]),
            lat_p95=sorted(float(x[1]) for x in m[1:])[int(len(m[1:])*0.95)],
            wall=round(wall,1), Q=statistics.mean(float(x[2]) for x in m),
            cost=statistics.mean(y[2] for y in hc) if hc else 0.0,
            rounds=statistics.mean(float(x[4]) for x in m if len(x)>4),
            dp_frac=dpn/max(len(dp),1), rss=rss,
            deferred=g("deferred"), flushes=g("flushes"))
    except Exception as ex: rec.update(status=type(ex).__name__)
    shutil.rmtree(d,ignore_errors=True)
    return rec
if __name__=="__main__":
    jobs=[]
    for ds in ("epinions","flickr","youtube"):
        nb=json.loads((R/f"data/runs/{ds}/meta.json").read_text())["n_batches"]
        for defer in (0,1):
            for s in (123,7,42,2024,31337): jobs.append((ds,nb,defer,s))
    print(f"{len(jobs)} jobs (3ds x 2 x 5seed), 6 in parallel",flush=True)
    res=[]
    with ProcessPoolExecutor(6) as ex:
        for n,f in enumerate(as_completed([ex.submit(one,j) for j in jobs]),1):
            r=f.result(); res.append(r)
            json.dump(res,open(R/"results/defer_ab.json","w"),indent=1)
            if n%6==0: print(f"  [{n}/{len(jobs)}]",flush=True)
    print("\n-> results/defer_ab.json")
