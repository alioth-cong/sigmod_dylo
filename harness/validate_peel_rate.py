#!/usr/bin/env python3
"""Validate rejection rate / Peel trigger rate -- direct evidence of surrogate sufficiency.

Review: the surrogate qualifiers in 4-incAlgo.tex:261 and 2-problem.tex:95 are accurate, but between C3's
"solve that surrogate exactly" and Algorithm 1's \\lIf{$\\neg$Validate}{Peel}
one sentence is missing. If Validate's rejection rate is low, that is exactly evidence of surrogate sufficiency and belongs in the experiments.

Code semantics (driver_std4.cxx:1335):
    accepted = (nSelected>0 && actualDeltaF > 1e-12)
i.e. whether the set chosen by the surrogate still has positive gain when re-evaluated on the **true** objective F.
Line 1360 `if(!accepted && !lamInf && PEEL)` triggers the Peel local-optimality repair.

Statistics (selection.tsv, excluding the PEEL rows themselves):
  proposed rounds = rounds with nSelected>0 (the surrogate actually proposed a non-empty set)
  rejected        = those with accepted==0
  rejection rate  = rejected / proposed rounds        <- the share in which the surrogate is insufficient
  Peel trigger rate = PEEL rows / proposed rounds

Uses DYLO_NO_COM=1 to skip writing partition files: this metric needs only selection.tsv, not com_*.txt,
avoiding tens of GB of intermediate files on large graphs (an earlier oversight filled the root partition).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import os, shutil, statistics, subprocess, sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
R=Path(DYLO_ROOT + ""); D=R/"harness/cpp/dylo"
OUT=R/"tmpruns/vprate"          # kept on the data disk, not /tmp (root partition)
BASE=dict(DYLO_EMIT_HIER="1",DYLO_LAMBDA="1e-1",DYLO_AUDIT_K="8",DYLO_REBUILD_R="1e18",
  DYLO_SIGMA_ONLY="1",DYLO_SIGMA="0.2",DYLO_FINAL_AUDIT="1",DYLO_TOPDOWN="0",
  DYLO_HIER_GAMMA="0.1",DYLO_PRUNE="1",DYLO_PRUNE_MODE="leaf",DYLO_DEFER="0",
  DYLO_LAZY_CERT="1",DYLO_INC_TREE="1",DYLO_LAZY_RECLAIM="1",
  DYLO_RECLAIM="1",DYLO_FAST_RECLAIM="1",DYLO_NO_COM="1")
def one(a):
    ds,nb,inst=a
    d=OUT/f"{ds}_{inst}"; shutil.rmtree(d,ignore_errors=True); d.mkdir(parents=True,exist_ok=True)
    e=dict(os.environ); e.update(BASE)
    rec=dict(ds=ds,inst=inst)
    try:
        subprocess.run([str(D/"dylo_driver_std4"),inst,
            str(R/f"data/runs/{ds}/initial.txt"),str(R/f"data/runs/{ds}/inc.txt"),
            str(nb),str(d)],env=e,stdout=subprocess.DEVNULL,
            stderr=open(d/"err","w"),check=True,timeout=21600)
        rows=[l.rstrip("\n").split("\t") for l in open(d/"selection.tsv")][1:]
        sel=[x for x in rows if len(x)>=12 and x[2]!="PEEL"]
        peel=[x for x in rows if len(x)>=3 and x[2]=="PEEL"]
        prop=[x for x in sel if int(x[3])>0]          # rounds in which the surrogate proposed a non-empty set
        rej =[x for x in prop if x[11]=="0"]
        rec.update(status="ok",rounds=len(sel),proposed=len(prop),
            rejected=len(rej), peel=len(peel),
            rej_rate=(len(rej)/len(prop) if prop else 0.0),
            peel_rate=(len(peel)/len(prop) if prop else 0.0))
    except Exception as ex: rec.update(status=type(ex).__name__)
    shutil.rmtree(d,ignore_errors=True)
    return rec
if __name__=="__main__":
    jobs=[]
    for ds in ("facebook","cit-hepth","cit-hepph","epinions","twitter","dblp","flickr","youtube"):
        nb=json.loads((R/f"data/runs/{ds}/meta.json").read_text())["n_batches"]
        for inst in ("leiden","louvain","slm"): jobs.append((ds,nb,inst))
    print(f"{len(jobs)} jobs, 8 in parallel",flush=True)
    res=[]
    with ProcessPoolExecutor(8) as ex:
        for f in as_completed([ex.submit(one,j) for j in jobs]):
            r=f.result(); res.append(r)
            if r["status"]=="ok":
                print(f"  {r['ds']:<11}{r['inst']:<8} proposed={r['proposed']:<5} "
                      f"rejected={r['rejected']:<4} ({r['rej_rate']*100:.2f}%)  "
                      f"Peel={r['peel']} ({r['peel_rate']*100:.2f}%)",flush=True)
            else: print(f"  {r['ds']:<11}{r['inst']:<8}{r['status']}",flush=True)
    json.dump(res,open(R/"results/validate_peel_rate.json","w"),indent=1)
    ok=[x for x in res if x["status"]=="ok"]
    tp=sum(x["proposed"] for x in ok); tr=sum(x["rejected"] for x in ok); tk=sum(x["peel"] for x in ok)
    print(f"\ntotal: {tp} rounds proposed, {tr} rejected ({tr/tp*100:.3f}%), Peel {tk} times ({tk/tp*100:.3f}%)")
    print("-> results/validate_peel_rate.json")
