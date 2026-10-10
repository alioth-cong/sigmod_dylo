#!/usr/bin/env python3
"""
ISOLATED solo latency benchmark for the fixed DyLo fast incremental path. Runs each
(instance, dataset) ONE AT A TIME (no parallel drivers — call only when the machine is
quiet) with the installed binary, single-thread. Collects mean ms/batch + Q + vol(A)/|E|.

Updates results/latency.csv dylo-* rows in place (preserving hit-leiden / baseline rows),
emits results/latency_compare.csv (speedup vs DF-Louvain single-thread) and
results/localization.csv. NOTE: run nothing else CPU-heavy concurrently.

Usage: latency_solo.py [--datasets a,b,...] [--instances louvain,slm,leiden] [--batches N]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse, csv, os, re, statistics, subprocess, time
from pathlib import Path

ROOT = Path(DYLO_ROOT + "")
DRIVER = ROOT / "harness/cpp/dylo/dylo_driver"
RUNS = ROOT / "data" / "runs"
RESULTS = ROOT / "results"
DBG = re.compile(r"vol/\|E\|=([0-9.]+)")
COMMIT = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()


def run(inst, ds, batches):
    rd = RUNS / ds
    env = dict(os.environ, DYLO_DEBUG="1")
    p = subprocess.run([str(DRIVER), inst, str(rd/"initial.txt"), str(rd/"inc.txt"),
                        str(batches), f"/tmp/latsolo_{inst}_{ds}"],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    rows = [l.split("\t") for l in open(Path(f"/tmp/latsolo_{inst}_{ds}")/"metrics.tsv")][1:]
    lat = statistics.mean(float(r[1]) for r in rows[1:])   # skip batch 0 (no update)
    Q = statistics.mean(float(r[2]) for r in rows)
    vols = [float(m.group(1)) for line in p.stderr.splitlines() if (m := DBG.search(line))]
    return lat, Q, (statistics.mean(vols) if vols else float("nan")), (max(vols) if vols else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="facebook,cit-hepth,cit-hepph,epinions,dblp,twitter,flickr,youtube,gplus")
    ap.add_argument("--instances", default="louvain,slm,leiden")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()
    dsets = args.datasets.split(","); insts = args.instances.split(",")

    df_lat = {r["dataset"]: float(r["value"]) for r in csv.DictReader(open(RESULTS/"latency.csv"))
              if r["method"] == "df-louvain"}
    lat = {}; loc = []
    for inst in insts:
        for ds in dsets:
            t0 = time.time()
            ms, Q, mvol, xvol = run(inst, ds, args.batches)
            lat[(inst, ds)] = (ms, Q)
            if inst == "leiden":
                loc.append(dict(method="dylo-leiden", dataset=ds, mean_volA_over_E=f"{mvol:.5f}",
                                max_volA_over_E=f"{xvol:.5f}", note="margin-gated affected region; single-thread"))
            print(f"[lat] {inst:7s} {ds:10s} {ms:7.3f}ms Q={Q:.4f} "
                  f"vol(A)/|E|={mvol:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # update latency.csv dylo-* rows in place
    rows = list(csv.DictReader(open(RESULTS/"latency.csv")))
    hdr = list(rows[0].keys())
    for r in rows:
        m = r["method"]
        if m.startswith("dylo-") and (m[5:], r["dataset"]) in lat:
            ms, _ = lat[(m[5:], r["dataset"])]
            r["value"] = f"{ms:.3f}"
            r["notes"] = re.sub(r"; re-measured solo.*$", "", r["notes"]) + f"; re-measured solo single-thread ({COMMIT}): margin-gate+flat-adj"
    with open(RESULTS/"latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for r in rows: w.writerow(r)

    # latency_compare.csv (speedup vs DF-Louvain single-thread)
    with open(RESULTS/"latency_compare.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "dylo_louvain_ms", "dylo_leiden_ms", "df_louvain_ms_1T",
                    "speedup_louvain_vs_df", "speedup_leiden_vs_df", "threads", "note"])
        for ds in dsets:
            lo = lat.get(("louvain", ds), (float("nan"),))[0]
            le = lat.get(("leiden", ds), (float("nan"),))[0]
            d = df_lat.get(ds, float("nan"))
            w.writerow([ds, f"{lo:.3f}", f"{le:.3f}", f"{d:.3f}",
                        f"{d/lo:.2f}" if lo == lo else "", f"{d/le:.2f}" if le == le else "",
                        1, "DyLo fast incremental path; DF-Louvain committed single-thread"])

    # localization.csv
    if loc:
        with open(RESULTS/"localization.csv", "w", newline="") as f:
            hdr2 = ["method", "dataset", "mean_volA_over_E", "max_volA_over_E", "note"]
            w = csv.DictWriter(f, fieldnames=hdr2); w.writeheader()
            for r in loc: w.writerow(r)
    print(f"[lat] updated latency.csv, latency_compare.csv, localization.csv ({COMMIT})")


if __name__ == "__main__":
    main()
