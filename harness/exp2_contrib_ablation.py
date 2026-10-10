#!/usr/bin/env python3
"""Exp-2 extra arms: ablate the contributions themselves, not the speed-up techniques.

The existing V1/V2/V3 turn off certificates, the incremental cost tree, and candidate pruning -- all engineering optimizations of the form
"same decisions, a bit faster". Measurements (exp2_prune_probe.py) confirm it:
V1/V2 give partitions bit-identical to V0; only latency changes. Such arms cannot answer "are the paper's contributions worth it".

This script adds two contribution-level arms:

  R2  per-destination greedy instead of the OpenSet DP (Alg 2)
      OpenSet gives the exact optimum over ancestor-closed dirty sets and does two things greedy cannot:
        (a) the cost of a shared ancestor is amortized across siblings -- k destinations under one parent pay w(parent) once;
        (b) it can accept a parent with g-lambda*w<0 as long as the subtree surplus pays it back.
      greedy decides each destination independently, each paying its whole ancestor chain, so (a)(b) are lost.
      Switch DYLO_GREEDY=1 (driver_r2greedy.cxx; GREEDY=0 verified bit-identical to std4).

  R3  flush removal-only sources immediately, i.e. DYLO_DEFER=0
      The default transient overlap does not rewrite A's summary right away when v leaves A; it refreshes once the residual share
      reaches THETA. The source comments state that partition, Q, and community counts are unchanged by construction;
      only the refresh SCHEDULE changes. So this arm's effect **can only be seen in cost**,
      and is necessarily zero in the partition -- which is exactly what is to be verified.

Partitions are compared by per-batch member-set hashes (ignoring community ids); cost uses the hierarchical accounting.
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import hashlib
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
D = ROOT / "harness/cpp/dylo"
RUNS = ROOT / "data/runs"
OUT = ROOT / "tmpruns/exp2contrib"
sys.path.insert(0, str(ROOT / "harness"))

# the main-table operating point, kept verbatim; DEFER is overridden per arm
BASE = dict(DYLO_EMIT_HIER="1", DYLO_LAMBDA="1e-1", DYLO_AUDIT_K="8",
            DYLO_REBUILD_R="1e18", DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2",
            DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0", DYLO_HIER_GAMMA="0.1",
            DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf", DYLO_LAZY_CERT="1",
            DYLO_INC_TREE="1", DYLO_LAZY_RECLAIM="1",
            DYLO_RECLAIM="1", DYLO_FAST_RECLAIM="1")

ARMS = {
    # name                     binary                  extra environment
    "V0 full":               ("dylo_driver_r2greedy", dict(DYLO_GREEDY="0", DYLO_DEFER="1")),
    "R2 greedy (no OpenSet)": ("dylo_driver_r2greedy", dict(DYLO_GREEDY="1", DYLO_DEFER="1")),
    "R3 flush (no defer)":   ("dylo_driver_r2greedy", dict(DYLO_GREEDY="0", DYLO_DEFER="0")),
    "R2+R3":                 ("dylo_driver_r2greedy", dict(DYLO_GREEDY="1", DYLO_DEFER="0")),
}


def part_sig(d: Path, nb: int) -> list:
    sig = []
    for k in range(nb + 1):
        f = d / f"com_{k}.txt"
        if not f.exists():
            sig.append("")
            continue
        cs = sorted(tuple(sorted(int(x) for x in l.split()))
                    for l in f.open() if l.strip())
        sig.append(hashlib.blake2b(repr(cs).encode(), digest_size=8).hexdigest())
    return sig


def run(name, binname, extra, ds, nb) -> dict:
    from hier_cost import stream_hier_cost
    d = OUT / ds / name.split()[0]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.update(BASE); env.update(extra)
    r = RUNS / ds
    try:
        subprocess.run([str(D / binname), "leiden", str(r / "initial.txt"),
                        str(r / "inc.txt"), str(nb), str(d)],
                       env=env, stdout=subprocess.DEVNULL,
                       stderr=open(d / "err", "w"), check=True, timeout=14400)
    except Exception as e:  # noqa: BLE001
        return dict(name=name, status=type(e).__name__)
    m = [l.rstrip("\n").split("\t") for l in (d / "metrics.tsv").open()][1:]
    hc = stream_hier_cost(d, nb)
    return dict(name=name, status="ok",
                lat=statistics.mean(float(x[1]) for x in m[1:]),
                Q=statistics.mean(float(x[2]) for x in m),
                Q_end=float(m[-1][2]), ncomm=m[-1][3],
                cost=statistics.mean(r[2] for r in hc) if hc else 0.0,
                sig=part_sig(d, nb), dir=str(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="epinions,flickr,youtube")
    ap.add_argument("--batches", type=int, default=0, help="0 = native number of batches")
    a = ap.parse_args()
    import json

    allres = {}
    for ds in a.datasets.split(","):
        nb = json.loads((RUNS / ds / "meta.json").read_text())["n_batches"]
        if a.batches:
            nb = min(nb, a.batches)
        print(f"\n===== {ds} ({nb} batches) =====", flush=True)
        res = []
        for name, (b, extra) in ARMS.items():
            r = run(name, b, extra, ds, nb)
            res.append(r)
            if r["status"] == "ok":
                print(f"  {name:<24} lat {r['lat']:>8.2f}ms  Q {r['Q']:.6f}  "
                      f"C(D) {r['cost']:.6f}  communities {r['ncomm']}", flush=True)
            else:
                print(f"  {name:<24} {r['status']}", flush=True)

        base = res[0]
        print(f"\n  {'arm':<24}{'same as V0?':<12}{'dQ':>12}{'dlat%':>9}{'dC(D)%':>10}")
        for r in res:
            if r["status"] != "ok":
                continue
            diff = [i for i, (x, y) in enumerate(zip(base["sig"], r["sig"])) if x != y]
            dc = (r["cost"] / base["cost"] - 1) * 100 if base["cost"] else 0.0
            dl = (r["lat"] / base["lat"] - 1) * 100 if base["lat"] else 0.0
            print(f"  {r['name']:<24}{('yes' if not diff else f'no({len(diff)} batches)'):<12}"
                  f"{r['Q']-base['Q']:>+12.6f}{dl:>+9.1f}{dc:>+10.1f}")
        allres[ds] = [{k: v for k, v in r.items() if k != "sig"} for r in res]

    p = ROOT / "results/exp2_contrib_ablation.json"
    json.dump(allres, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"\nresults -> {p}")


if __name__ == "__main__":
    main()
