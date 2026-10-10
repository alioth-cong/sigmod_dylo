#!/usr/bin/env python3
"""Exp-2, all six arms x triplets, with uniform accounting (full stream with native batches, hierarchical cost, lambda=1e-1).

V0  full system (DEFER=1, so the accrual ledger is observable)
V1  certificates off      (DYLO_LAZY_CERT=0)      -- Section 6 implementation optimization
V2  incremental cost tree off (DYLO_INC_TREE=0)   -- Section 6 implementation optimization
V3  candidate pruning off (DYLO_PRUNE=0)          -- Section 6 implementation optimization (lossy)
V4a greedy instead of OpenSet (DYLO_GREEDY=1)     -- contribution-level (review R2)
V4b source price off      (DYLO_SRC_CHARGE=0)     -- contribution-level (accrual ledger)
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import hashlib, os, shutil, statistics, subprocess, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
R=Path(DYLO_ROOT + ""); D=R/"harness/cpp/dylo"
S=Path(DYLO_ROOT + "/tmpruns/scratch")
sys.path.insert(0,str(R/"harness"))
BASE=dict(DYLO_EMIT_HIER="1",DYLO_LAMBDA="1e-1",DYLO_AUDIT_K="8",DYLO_REBUILD_R="1e18",
  DYLO_SIGMA_ONLY="1",DYLO_SIGMA="0.2",DYLO_FINAL_AUDIT="1",DYLO_TOPDOWN="0",
  DYLO_HIER_GAMMA="0.1",DYLO_PRUNE_MODE="leaf",DYLO_LAZY_RECLAIM="1",
  DYLO_RECLAIM="1",DYLO_FAST_RECLAIM="1",DYLO_DEFER="1")
ARMS={
 "V0": dict(DYLO_PRUNE="1",DYLO_LAZY_CERT="1",DYLO_INC_TREE="1",DYLO_GREEDY="0",DYLO_SRC_CHARGE="1"),
 "V1": dict(DYLO_PRUNE="1",DYLO_LAZY_CERT="0",DYLO_INC_TREE="1",DYLO_GREEDY="0",DYLO_SRC_CHARGE="1"),
 "V2": dict(DYLO_PRUNE="1",DYLO_LAZY_CERT="1",DYLO_INC_TREE="0",DYLO_GREEDY="0",DYLO_SRC_CHARGE="1"),
 "V3": dict(DYLO_PRUNE="0",DYLO_LAZY_CERT="1",DYLO_INC_TREE="1",DYLO_GREEDY="0",DYLO_SRC_CHARGE="1"),
 "V4a":dict(DYLO_PRUNE="1",DYLO_LAZY_CERT="1",DYLO_INC_TREE="1",DYLO_GREEDY="1",DYLO_SRC_CHARGE="1"),
 "V4b":dict(DYLO_PRUNE="1",DYLO_LAZY_CERT="1",DYLO_INC_TREE="1",DYLO_GREEDY="0",DYLO_SRC_CHARGE="0"),
}
def one(a):
    ds,nb,arm,extra=a
    from hier_cost import stream_hier_cost
    d=S/"exp2f"/f"{ds}_{arm}"; shutil.rmtree(d,ignore_errors=True); d.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ); e.update(BASE); e.update(extra)
    rec=dict(ds=ds,arm=arm)
    try:
        subprocess.run([str(D/"dylo_driver_r2greedy"),"leiden",
            str(R/f"data/runs/{ds}/initial.txt"),str(R/f"data/runs/{ds}/inc.txt"),
            str(nb),str(d)],env=e,stdout=subprocess.DEVNULL,
            stderr=open(d/"err","w"),check=True,timeout=21600)
        m=[l.rstrip().split("\t") for l in open(d/"metrics.tsv")][1:]
        hc=stream_hier_cost(d,nb)
        sig=[]
        for k in range(nb+1):
            f=d/f"com_{k}.txt"
            cs=sorted(tuple(sorted(int(x) for x in l.split())) for l in f.open() if l.strip()) if f.exists() else []
            sig.append(hashlib.blake2b(repr(cs).encode(),digest_size=8).hexdigest())
        rec.update(status="ok",lat=statistics.mean(float(x[1]) for x in m[1:]),
            Q=statistics.mean(float(x[2]) for x in m),Qe=float(m[-1][2]),
            cost=statistics.mean(y[2] for y in hc) if hc else 0.0,sig=sig)
    except Exception as ex: rec.update(status=type(ex).__name__)
    shutil.rmtree(d,ignore_errors=True)
    return rec
if __name__=="__main__":
    jobs=[]
    for ds in ("epinions","flickr","youtube"):
        nb=json.loads((R/f"data/runs/{ds}/meta.json").read_text())["n_batches"]
        for arm,extra in ARMS.items(): jobs.append((ds,nb,arm,extra))
    print(f"{len(jobs)} jobs, 6 in parallel",flush=True)
    res=[]
    with ProcessPoolExecutor(6) as ex:
        for n,f in enumerate(as_completed([ex.submit(one,j) for j in jobs]),1):
            r=f.result(); res.append(r)
            print(f"  [{n}/{len(jobs)}] {r['ds']:<9}{r['arm']:<5}{r.get('status')}",flush=True)
    out=[]
    for x in res:
        b=next((y for y in res if y["ds"]==x["ds"] and y["arm"]=="V0"),None)
        if x["status"]=="ok" and b and b["status"]=="ok":
            x["dcost"]=(x["cost"]/b["cost"]-1)*100 if b["cost"] else 0.0
            x["dlat"]=x["lat"]/b["lat"]
            x["ndiff"]=sum(1 for p,q in zip(b["sig"],x["sig"]) if p!=q)
        out.append({k:v for k,v in x.items() if k!="sig"})
    json.dump(out,open(R/"results/exp2_full.json","w"),indent=1)
    print("\n-> results/exp2_full.json")
