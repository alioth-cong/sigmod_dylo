#!/usr/bin/env python3
"""Three-path epsilon sweep: turn eps from "a symbol without a value" into a curve with a knee.

Preliminary diagnosis (distribution of |dq_A|/|q_A| measured in drift.tsv):
  Epinions 60 surviving samples, median 5.8e-3, max 1.98e-2
  DBLP     171 surviving samples, median 1.8e-3, max 4.66e-3
The default eps=0.05 exceeds the maximum observed drift of both (2.5x / 10.7x), so the predicate never fires,
and every flush is driven by "community vanished" (vanished=3648/4870, quality=0).
The operative range of eps is actually [1e-4, 1e-2]; the default is one to two orders of magnitude higher.

Three paths:
  P1 low-range eps sweep        -- directly gives the cost/freshness trade-off curve
  P2 singleton reclamation off  -- deferred sources are mostly small communities that reclamation removes before drift accumulates;
                                   with it off more samples survive and eps has more to act on
  P3 large batches              -- drift is checked once per batch, so larger batches accumulate more per check,
                                   shifting the drift distribution right, possibly into the range where the default eps fires

Each arm reports the closing P_T / K_T (last line of defer.tsv):
  P_T = communities that still have residual members   K_T = sum_c sigma(c) = total residual members
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import os, shutil, statistics, subprocess, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
R=Path(DYLO_ROOT + ""); D=R/"harness/cpp/dylo"
S=Path(DYLO_ROOT + "/tmpruns/epsfull")
sys.path.insert(0,str(R/"harness"))
BASE=dict(DYLO_EMIT_HIER="1",DYLO_LAMBDA="1e-1",DYLO_AUDIT_K="8",DYLO_REBUILD_R="1e18",
  DYLO_SIGMA_ONLY="1",DYLO_SIGMA="0.2",DYLO_FINAL_AUDIT="1",DYLO_TOPDOWN="0",
  DYLO_HIER_GAMMA="0.1",DYLO_PRUNE="1",DYLO_PRUNE_MODE="leaf",DYLO_LAZY_CERT="1",
  DYLO_INC_TREE="1",DYLO_LAZY_RECLAIM="1")
EPS=["1e-5","1e-4","3e-4","1e-3","3e-3","1e-2","0.05"]
def one(a):
    path,ds,base,nb,eps,extra=a
    from hier_cost import stream_hier_cost
    d=S/"epsf"/f"{path}_{ds}_{eps}"; shutil.rmtree(d,ignore_errors=True)
    d.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ); e.update(BASE); e.update(extra)
    e.update(DYLO_DEFER="1",DYLO_DEFER_MODE="quality",DYLO_DEFER_EPS=eps,DYLO_DRIFT_LOG="1")
    t0=time.time(); rec=dict(path=path,ds=ds,eps=eps)
    try:
        subprocess.run([str(D/"dylo_driver_eps"),"leiden",str(base/"initial.txt"),
            str(base/"inc.txt"),str(nb),str(d)],env=e,stdout=subprocess.DEVNULL,
            stderr=open(d/"err","w"),check=True,timeout=21600)
        m=[l.rstrip().split("\t") for l in open(d/"metrics.tsv")][1:]
        hc=stream_hier_cost(d,nb)                      # re-derived from partitions (kept for comparison)
        rsm=[x.rstrip().split("\t") for x in open(d/"resum.tsv")][1:]
        rsm=[x for x in rsm if len(x)>=6]
        df=[x.rstrip().split("\t") for x in open(d/"defer.tsv")][1:]
        df=[x for x in df if len(x)>=7]
        dr=[]
        if (d/"drift.tsv").exists():
            dr=[float(x.rstrip().split("\t")[4]) for x in open(d/"drift.tsv").readlines()[1:]
                if len(x.rstrip().split("\t"))>=6]
        rec.update(status="ok",
            Q=statistics.mean(float(x[2]) for x in m),
            lat=statistics.mean(float(x[1]) for x in m[1:]),
            # deferral changes only the refresh **plan**, not the partition; re-deriving from partitions with hier_cost would
            # charge every source community (driver_std4.cxx:243 warns about this), so the primary accounting uses
            # the driver's own resum.tsv, with the partition-based accounting kept alongside to show the difference.
            cost=statistics.mean(float(x[5]) for x in rsm) if rsm else 0.0,
            cost_part=statistics.mean(y[2] for y in hc) if hc else 0.0,
            PT_end=int(df[-1][1]) if df else 0, KT_end=int(df[-1][2]) if df else 0,
            KT_max=max(int(x[2]) for x in df) if df else 0,
            KT_mean=round(statistics.mean(int(x[2]) for x in df),2) if df else 0.0,
            deferred=int(df[-1][3]) if df else 0,
            fq=int(df[-1][5]) if df else 0, fv=int(df[-1][6]) if df else 0,
            drift_n=len(dr), drift_med=round(statistics.median(dr),8) if dr else 0.0,
            drift_max=round(max(dr),8) if dr else 0.0, sec=round(time.time()-t0,1))
    except Exception as ex:
        rec.update(status=f"{type(ex).__name__}")
    shutil.rmtree(d,ignore_errors=True)
    return rec
if __name__=="__main__":
    jobs=[]
    for ds in ("epinions","dblp"):
        nb=json.loads((R/f"data/runs/{ds}/meta.json").read_text())["n_batches"]
        b=R/f"data/runs/{ds}"
        for eps in EPS:
            jobs.append(("P1",ds,b,nb,eps,dict(DYLO_RECLAIM="1",DYLO_FAST_RECLAIM="1")))
            jobs.append(("P2-norecl",ds,b,nb,eps,{}))     # RECLAIM unset = reclamation off
    for ds,nb in (("epinions_x4",10),("flickr_x32",10)):
        b=S/"bigbatch"/ds
        for eps in EPS:
            jobs.append(("P3-bigbatch",ds,b,nb,eps,dict(DYLO_RECLAIM="1",DYLO_FAST_RECLAIM="1")))
    print(f"{len(jobs)} jobs, 10 in parallel\n",flush=True)
    res=[]
    with ProcessPoolExecutor(10) as ex:
        for n,f in enumerate(as_completed([ex.submit(one,j) for j in jobs]),1):
            r=f.result(); res.append(r)
            json.dump(res,open(R/"results/eps_full_resum.json","w"),indent=1,ensure_ascii=False)
            print(f"  [{n}/{len(jobs)}] {r['path']:<12}{r['ds']:<12}ε={r['eps']:<6}{r.get('status')}",flush=True)
    print("\n-> results/eps_full.json")
