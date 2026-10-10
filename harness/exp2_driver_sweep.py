#!/usr/bin/env python3
"""Cross-version check of the driver: does the cost difference between V0 and PRUNE=0 depend on which binary is used?

Origin: the Exp-2 decision experiment used only dylo_driver_std4. If Table 5 had been produced with another version,
the conclusion might not transfer. Here the same pair of configurations is rerun for every candidate driver.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import hashlib, os, shutil, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
D = ROOT / "harness/cpp/dylo"
RUNS, OUT = ROOT / "data/runs", ROOT / "tmpruns/exp2sweep"
DS, NB = "epinions", 20

BASE = dict(DYLO_EMIT_HIER="1", DYLO_LAMBDA="1e-1", DYLO_AUDIT_K="8",
            DYLO_REBUILD_R="1e18", DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2",
            DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0", DYLO_HIER_GAMMA="0.1",
            DYLO_DEFER="0", DYLO_LAZY_CERT="1", DYLO_INC_TREE="1",
            DYLO_LAZY_RECLAIM="1")

DRIVERS = ["dylo_driver_std", "dylo_driver_std2", "dylo_driver_std3",
           "dylo_driver_std4", "dylo_driver_bucket", "dylo_driver_bucket2",
           "dylo_sel", "dylo_driver_stale", "dylo_driver_attach"]

def go(drv, tag, cfg):
    d = OUT / drv / tag
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.update(BASE); env.update(cfg)
    r = RUNS / DS
    try:
        subprocess.run([str(D/drv), "leiden", str(r/"initial.txt"), str(r/"inc.txt"),
                        str(NB), str(d)], env=env, stdout=subprocess.DEVNULL,
                       stderr=open(d/"err","w"), check=True, timeout=7200)
        m = [l.rstrip().split("\t") for l in (d/"metrics.tsv").open()][1:]
        lat = statistics.mean(float(x[1]) for x in m[1:])
        Q = float(m[-1][2])
        rows = [l.rstrip().split("\t") for l in (d/"resum.tsv").open()][1:]
        cost = statistics.mean(float(x[3]) for x in rows) if rows else 0.0
        sig = []
        for k in range(NB+1):
            f = d/f"com_{k}.txt"
            cs = sorted(tuple(sorted(int(x) for x in l.split())) for l in f.open() if l.strip()) if f.exists() else []
            sig.append(hashlib.blake2b(repr(cs).encode(), digest_size=8).hexdigest())
        return dict(ok=1, lat=lat, Q=Q, cost=cost, sig=sig)
    except Exception as e:
        return dict(ok=0, why=f"{type(e).__name__}")

print(f"{'driver':<22}{'V0 lat':>9}{'V0 Q':>10}{'V0 cost':>10}"
      f"{'P0 lat':>9}{'P0 Q':>10}{'P0 cost':>10}{'dcost%':>9}  partition")
for drv in DRIVERS:
    if not (D/drv).exists():
        print(f"{drv:<22} (missing)"); continue
    a = go(drv, "v0", dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf"))
    b = go(drv, "p0", dict(DYLO_PRUNE="0"))
    if not a["ok"] or not b["ok"]:
        print(f"{drv:<22} failed V0={a.get('why','ok')} P0={b.get('why','ok')}"); continue
    dc = (b["cost"]/a["cost"]-1)*100 if a["cost"] else 0.0
    diff = [i for i,(x,y) in enumerate(zip(a["sig"],b["sig"])) if x!=y]
    print(f"{drv:<22}{a['lat']:>9.3f}{a['Q']:>10.6f}{a['cost']:>10.1f}"
          f"{b['lat']:>9.3f}{b['Q']:>10.6f}{b['cost']:>10.1f}{dc:>+9.1f}  "
          f"{'same' if not diff else f'{len(diff)} batches differ from {diff[0]}'}")
