#!/usr/bin/env python3
"""LiveJournal rerun in main-table accounting: rebuild + all baselines, with controlled parallelism.

Why a separate script: LiveJournal was the only one of the ten datasets never run through. The native stream has 4,286 batches
x 4.85M nodes, 37MB of com files per batch, about 100GB per method; twelve methods need ~900GB,
more than the disk holds, and leiden-static reruns the whole graph from scratch per batch, so the full stream is infeasible.

It is therefore cut to the first 200 batches (200k stream edges). The batch count is of the same order as Flickr (320) / YouTube (299),
and cost_norm is a per-batch average, consistent with the other nine datasets.

Parallelism: Java methods use -Xmx128g, and on 4.85M nodes their resident memory reaches tens of GB, so they are throttled
separately from C++ methods to avoid four JVMs exhausting memory at once.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
H = ROOT / "harness"
RUNS = ROOT / "data/runs"

# methods in the main-table ROWS produced by run.py. hit-leiden goes through exp5 lambda=0 and DyLo through
# recost_reclaim_ab.py (std4); neither is handled here.
CPP = ["df-louvain", "df-leiden", "ds-louvain"]
JAVA = ["gremod", "qca", "batch", "dynamo"]
PY = ["leiden-static"]          # i.e. the main table's From-scratch Leiden (rebuild)


def free_gb(p=DYLO_ROOT) -> float:
    s = os.statvfs(p)
    return s.f_bavail * s.f_frsize / 1e9


def one(method: str, ds: str, nb: int, timeout: int) -> dict:
    t0 = time.time()
    log = ROOT / f"logs/lj_run_{method}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(H / "run.py"), "--method", method,
           "--dataset", ds, "--max-batches", str(nb), "--timeout", str(timeout)]
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout + 600)
            st = "ok" if r.returncode == 0 else f"exit={r.returncode}"
        except subprocess.TimeoutExpired:
            st = "TLE"
    out = RUNS / ds / f"out_{method}"
    done = 0
    while (out / f"com_{done}.txt").exists() or (out / f"com_{done}_L0.txt").exists():
        done += 1
    return dict(method=method, status=st, batches_done=done - 1,
                sec=round(time.time() - t0, 1), free_gb=round(free_gb()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="livejournal")
    ap.add_argument("--batches", type=int, default=200)
    ap.add_argument("--groups", default="cpp,java",
                    help="comma-separated cpp / java / py; py is leiden-static (rebuild)")
    ap.add_argument("--cpp-par", type=int, default=3)
    ap.add_argument("--java-par", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=20 * 3600)
    a = ap.parse_args()

    groups = a.groups.split(",")
    plan = [(m, a.cpp_par) for m in (CPP if "cpp" in groups else [])] \
        + [(m, a.java_par) for m in (JAVA if "java" in groups else [])] \
        + [(m, 1) for m in (PY if "py" in groups else [])]
    print(f"{a.dataset} first {a.batches} batches, {len(plan)} methods, "
          f"{free_gb():.0f}G disk free", flush=True)

    res = []
    for pool, ms in (("cpp", CPP), ("java", JAVA), ("py", PY)):
        if pool not in groups:
            continue
        par = {"cpp": a.cpp_par, "java": a.java_par, "py": 1}[pool]
        print(f"\n== {pool} group: {ms}, parallelism {par} ==", flush=True)
        with ThreadPoolExecutor(par) as ex:
            futs = {ex.submit(one, m, a.dataset, a.batches, a.timeout): m for m in ms}
            for f in as_completed(futs):
                r = f.result()
                res.append(r)
                print(f"  {r['method']:<14} {r['status']:<8} "
                      f"batches={r['batches_done']:<4} {r['sec']}s disk free {r['free_gb']}G",
                      flush=True)

    out = ROOT / f"results/lj_run_status.json"
    prev = json.loads(out.read_text()) if out.exists() else []
    json.dump(prev + res, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\nstatus -> {out}")


if __name__ == "__main__":
    main()
