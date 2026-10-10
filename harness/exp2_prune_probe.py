#!/usr/bin/env python3
"""Exp-2 decision experiment: does pruning change the committed set at all?

The conflict: the paper's safe-reduction claim says non-positive removal + bucket pruning
preserve the optimum of the frozen objective, while the Exp-2 table (V3) says turning off candidate pruning changes the
committed set and raises C(D) by 51-119%. Both cannot be true.

The comment at lines 192-228 of driver_bucket.cxx gives the answer: the admissibility of DYLO_PRUNE
depends on DYLO_PRUNE_MODE --
  MODE=comm : uses the destination **community** size as the cost, an overestimate (Facebook 224 vs 22),
              stricter than the model it approximates, and it "can discard a move the full selector
              would have accepted" -- unsafe;
  MODE=leaf : uses the destination **fine leaf** size; closure(D) is that weight plus ancestors, so it is a true lower bound,
              "the filter can no longer discard a move the full selector
              and it never discards a move the full selector would have accepted -- safe. leaf is the default.

So this script compares the committed sets of four configurations bit by bit (the per-batch partition files) to decide:
  1) whether PRUNE=0 and the default (leaf) give the same partition  -> whether the theorem holds
  2) whether MODE=comm gives a different partition            -> whether V3 actually measured comm
  3) the effect of DISABLE_BUCKET

Partitions are compared by the normalized member sets of the per-batch com_k.txt (hashed after sorting), ignoring community ids --
ids can change with enumeration order; the member sets are what the committed set really is.
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
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver_std4"
RUNS = ROOT / "data/runs"
OUT = ROOT / "tmpruns/exp2probe"

# same operating point as end-to-end / main table
BASE = dict(DYLO_EMIT_HIER="1", DYLO_LAMBDA="1e-1", DYLO_AUDIT_K="8",
            DYLO_REBUILD_R="1e18", DYLO_SIGMA_ONLY="1", DYLO_SIGMA="0.2",
            DYLO_FINAL_AUDIT="1", DYLO_TOPDOWN="0", DYLO_HIER_GAMMA="0.1",
            DYLO_DEFER="0", DYLO_LAZY_CERT="1", DYLO_INC_TREE="1",
            DYLO_LAZY_RECLAIM="1")

CONFIGS = {
    "V0 default (PRUNE=1, leaf)": dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf"),
    "PRUNE=0 (off)":              dict(DYLO_PRUNE="0"),
    "PRUNE=1, MODE=comm":         dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="comm"),
    "DISABLE_BUCKET":             dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
                                       DYLO_DISABLE_BUCKET="1"),
    "V1 no certificates":         dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
                                       DYLO_LAZY_CERT="0"),
    "V2 no inc cost-tree":        dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
                                       DYLO_INC_TREE="0"),
    # hypothesis: Table 5's V3 actually turned off cost awareness (lambda=0), not pruning.
    # The evidence is the direction -- turning off lambda raises cost (by 7.3x on DBLP in ablation.csv),
    # whereas turning off pruning measurably lowers it by 12.9%. A few lambdas are swept here to see which matches +52/+50/+119%.
    "lambda=0 (no cost term)":    dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
                                       DYLO_LAMBDA="0"),
    "lambda=1e-3":                dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
                                       DYLO_LAMBDA="1e-3"),
    "lambda=1e-2":                dict(DYLO_PRUNE="1", DYLO_PRUNE_MODE="leaf",
                                       DYLO_LAMBDA="1e-2"),
    # combination: turn off both pruning and the cost term. Pruning alone "lowers cost by 12.9%" at lambda=1e-1,
    # but its filter criterion is gain < lambda*|D| -- the smaller lambda, the looser the criterion, so the two are not orthogonal,
    # and the combined effect need not equal the sum of the separate effects.
    "PRUNE=0 + lambda=0":         dict(DYLO_PRUNE="0", DYLO_LAMBDA="0"),
    "PRUNE=0 + lambda=1e-2":      dict(DYLO_PRUNE="0", DYLO_LAMBDA="1e-2"),
    "PRUNE=0 + lambda=3e-2":      dict(DYLO_PRUNE="0", DYLO_LAMBDA="3e-2"),
}


def part_sig(d: Path, nb: int) -> list[str]:
    """Normalized fingerprint of each batch's partition: hash of sorted member sets, ignoring community ids."""
    sig = []
    for k in range(nb + 1):
        f = d / f"com_{k}.txt"
        if not f.exists():
            sig.append("")
            continue
        cs = sorted(tuple(sorted(int(x) for x in l.split())) for l in f.open() if l.strip())
        h = hashlib.blake2b(repr(cs).encode(), digest_size=8).hexdigest()
        sig.append(h)
    return sig


def run(name: str, cfg: dict, ds: str, nb: int) -> dict:
    d = OUT / ds / name.split()[0].replace("=", "").replace(",", "")
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ); env.update(BASE); env.update(cfg)
    r = RUNS / ds
    try:
        subprocess.run([str(DRIVER), "leiden", str(r / "initial.txt"),
                        str(r / "inc.txt"), str(nb), str(d)],
                       env=env, stdout=subprocess.DEVNULL,
                       stderr=open(d / "err", "w"), check=True, timeout=7200)
    except Exception as e:  # noqa: BLE001
        return dict(name=name, status=f"{type(e).__name__}")
    m = [l.rstrip("\n").split("\t") for l in (d / "metrics.tsv").open()][1:]
    lat = statistics.mean(float(x[1]) for x in m[1:]) if len(m) > 1 else 0.0
    Q = float(m[-1][2])
    cost = norm = 0.0
    rs = d / "resum.tsv"
    if rs.exists():
        rows = [l.rstrip("\n").split("\t") for l in rs.open()][1:]
        if rows:
            cost = statistics.mean(float(x[3]) for x in rows)
            norm = statistics.mean(float(x[5]) for x in rows)
    return dict(name=name, status="ok", lat=lat, Q=Q, cost=cost, norm=norm,
                sig=part_sig(d, nb), dir=str(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="epinions")
    ap.add_argument("--batches", type=int, default=20)
    a = ap.parse_args()
    import json
    nb = a.batches
    meta = json.loads((RUNS / a.ds / "meta.json").read_text())
    nb = min(nb, meta["n_batches"])
    print(f"dataset {a.ds}, running {nb} batches (full stream {meta['n_batches']} batches)\n")

    res = []
    for name, cfg in CONFIGS.items():
        r = run(name, cfg, a.ds, nb)
        res.append(r)
        if r["status"] == "ok":
            print(f"  {name:<30} lat {r['lat']:>8.3f}ms  Q {r['Q']:.6f}  "
                  f"cost {r['cost']:>9.1f}  share {r['norm']:.5f}")
        else:
            print(f"  {name:<30} {r['status']}")

    base = next(r for r in res if r["name"].startswith("V0"))
    print(f"\n{'configuration':<30}{'same partition as V0?':<18}{'differing batches':<12}{'dQ':>12}{'dcost%':>10}")
    for r in res:
        if r["status"] != "ok":
            continue
        diff = [i for i, (x, y) in enumerate(zip(base["sig"], r["sig"])) if x != y]
        same = "yes (bit-identical)" if not diff else f"no"
        dq = r["Q"] - base["Q"]
        dc = (r["cost"] / base["cost"] - 1) * 100 if base["cost"] else 0.0
        print(f"{r['name']:<30}{same:<18}{str(diff[:6]) if diff else '--':<12}"
              f"{dq:>+12.6f}{dc:>+10.1f}")

    json.dump([{k: v for k, v in r.items() if k != "sig"} for r in res],
              open(ROOT / f"results/exp2_probe_{a.ds}.json", "w"), indent=1)
    print(f"\nresults: {ROOT}/results/exp2_probe_{a.ds}.json")


if __name__ == "__main__":
    main()
