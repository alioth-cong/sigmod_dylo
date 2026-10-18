#!/usr/bin/env python3
"""DEFER on/off: selection-phase timing + cost from resum.tsv.

Two corrections of earlier measurement errors:

(1) Timing. In end-to-end latency the selection phase is buried under graph building/certificates/output, hiding
    the gain of "single anchor -> tree DP replaces min-cut". driver_selection_timing.cxx accumulates the time of
    the DP path and the min-cut path separately (verified bit-identical to std4).

(2) Cost accounting. hier_cost.py re-derives cost from **partitions**: once a source community loses a vertex its member set changes,
    and it is charged whether or not the driver deferred its refresh -- so the effect of deferral is invisible to it.
    The comment at driver_std4.cxx:243 warns about exactly this; the correct source is the refresh plan
    the driver itself emits, resum.tsv (column cost_norm). The earlier report that "defer raises cost by 13-45%"
    was entirely this artifact and has been withdrawn.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import os, re, shutil, statistics as st, subprocess, sys, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

R = Path(DYLO_ROOT + ""); D = R / "harness/cpp/dylo"
OUT = R / "tmpruns/seltime"
sys.path.insert(0, str(R / "harness"))
BASE = dict(DYLO_EMIT_HIER="1", DYLO_LAMBDA="1e-1", DYLO_AUDIT_K="8",
            DYLO_REBUILD_R="1e18", DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2",
            DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0", DYLO_HIER_GAMMA="0.1",
            DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_LAZY_CERT="1",
            DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1", DYLO_RECLAIM="1",
            DYLO_FAST_RECLAIM="1", DYLO_SRC_CHARGE="1")


def one(a):
    ds, nb, defer, seed = a
    from hier_cost import stream_hier_cost
    d = OUT / f"{ds}_d{defer}_s{seed}"
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ); e.update(BASE)
    e.update(DYLO_DEFER=str(defer), DYLO_SEED=str(seed))
    rec = dict(ds=ds, defer=defer, seed=seed)
    try:
        subprocess.run([str(D / "dylo_driver_selection_timing"), "leiden",
                        str(R / f"data/runs/{ds}/initial.txt"),
                        str(R / f"data/runs/{ds}/inc.txt"), str(nb), str(d)],
                       env=e, stdout=subprocess.DEVNULL,
                       stderr=open(d / "err", "w"), check=True, timeout=21600)
        err = open(d / "err").read()
        m = re.search(r"dp_rounds=(\d+) dp_ms=([\d.]+) cut_rounds=(\d+) cut_ms=([\d.]+)", err)
        mm = [l.rstrip().split("\t") for l in open(d / "metrics.tsv")][1:]
        rs = [l.rstrip().split("\t") for l in open(d / "resum.tsv")][1:]
        rs = [x for x in rs if len(x) >= 6]
        hc = stream_hier_cost(d, nb)
        rec.update(status="ok",
                   lat=st.mean(float(x[1]) for x in mm[1:]),
                   Q=st.mean(float(x[2]) for x in mm),
                   cost_resum=st.mean(float(x[5]) for x in rs) if rs else 0.0,
                   cost_part=st.mean(y[2] for y in hc) if hc else 0.0,
                   dp_rounds=int(m[1]), dp_ms=float(m[2]),
                   cut_rounds=int(m[3]), cut_ms=float(m[4]))
    except Exception as ex:  # noqa: BLE001
        rec.update(status=f"{type(ex).__name__}")
    shutil.rmtree(d, ignore_errors=True)
    return rec


if __name__ == "__main__":
    jobs = []
    for ds in ("epinions", "flickr", "youtube", "dblp"):
        nb = json.loads((R / f"data/runs/{ds}/meta.json").read_text())["n_batches"]
        for defer in (0, 1):
            for s in (123, 7, 42):
                jobs.append((ds, nb, defer, s))
    print(f"{len(jobs)} jobs (4 datasets x DEFER{{0,1}} x 3 seeds), 8 in parallel", flush=True)
    res = []
    with ProcessPoolExecutor(8) as ex:
        for n, f in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            r = f.result(); res.append(r)
            json.dump(res, open(R / "results/defer_seltime.json", "w"), indent=1)
            if n % 8 == 0:
                print(f"  [{n}/{len(jobs)}]", flush=True)
    print("\n-> results/defer_seltime.json")
