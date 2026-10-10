#!/usr/bin/env python3
"""
Emit results/localization.csv: per dataset, DyLo's affected-region localization
vol(A)/|E| (mean, max over batches) and |A|, demonstrating the a-priori margin-gated
affected-region bound (vol(A) << |E| on every graph, incl. dense gplus). Parses the
DYLO_DEBUG stderr of the fast incremental driver.

Usage: localization.py --datasets facebook,cit-hepph,... [--instance leiden] [--batches N]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse, csv, os, re, statistics, subprocess
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver"
RUNS = ROOT / "data" / "runs"
RESULTS = ROOT / "results"
DBG = re.compile(r"\|A\|=(\d+) vol\(A\)=([0-9.]+) vol/\|E\|=([0-9.]+)")


def run(inst, ds, batches):
    rd = RUNS / ds
    env = dict(os.environ, DYLO_DEBUG="1")
    p = subprocess.run([str(DRIVER), inst, str(rd/"initial.txt"), str(rd/"inc.txt"),
                        str(batches), f"/tmp/loc_{inst}_{ds}"],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    A, vol = [], []
    for line in p.stderr.splitlines():
        m = DBG.search(line)
        if m:
            A.append(int(m.group(1))); vol.append(float(m.group(3)))
    return A, vol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--instance", default="leiden")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--out", default=str(RESULTS / "localization.csv"))
    args = ap.parse_args()
    rows = []
    for ds in args.datasets.split(","):
        A, vol = run(args.instance, ds, args.batches)
        if not vol:
            print(f"[loc] {ds}: no debug output"); continue
        rows.append(dict(method=f"dylo-{args.instance}", dataset=ds,
                         mean_volA_over_E=f"{statistics.mean(vol):.5f}",
                         max_volA_over_E=f"{max(vol):.5f}",
                         mean_A=f"{statistics.mean(A):.0f}",
                         batches=len(vol),
                         note="margin-gated affected region; insertion-only"))
        print(f"[loc] {ds:10s} mean vol(A)/|E|={statistics.mean(vol):.4f} "
              f"max={max(vol):.4f} mean|A|={statistics.mean(A):.0f}")
    out = Path(args.out)
    hdr = ["method", "dataset", "mean_volA_over_E", "max_volA_over_E", "mean_A", "batches", "note"]
    existing = []
    if out.exists():
        existing = [r for r in csv.DictReader(open(out))
                    if (r["method"], r["dataset"]) not in {(x["method"], x["dataset"]) for x in rows}]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in existing + rows:
            w.writerow({k: r.get(k, "") for k in hdr})
    print(f"[loc] wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
