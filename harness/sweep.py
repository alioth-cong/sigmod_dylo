#!/usr/bin/env python3
"""
Sweep orchestrator: schedule (method, dataset) cells with a concurrency limit.

Usage:
  sweep.py prep --datasets all|small|medium|big|<name,name,...>
  sweep.py run  --datasets <selection> --methods <name,name,...> --max-parallel N --timeout SECS

Reads manifest, dispatches subprocesses calling harness/run.py, streams a one-line
status per cell to stdout. Skip cells already complete (per_batch tsv present and
metrics.tsv populated to last batch).
"""
from __future__ import annotations
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

ROOT     = Path(DYLO_ROOT + "")
HARNESS  = ROOT / "harness"
RESULTS  = ROOT / "results"
PER_BATCH= RESULTS / "per_batch"
RUNS     = ROOT / "data" / "runs"
EDGES    = ROOT / "data" / "edges"
MANIFEST = ROOT / "data" / "manifest.tsv"

DATASET_GROUPS = {
    "tiny":   ["facebook"],
    "small":  ["facebook", "cit-hepth", "cit-hepph"],
    "medium": ["epinions", "twitter", "dblp", "flickr"],
    "big":    ["gplus", "youtube"],
    "huge":   ["livejournal"],
    "all":    ["facebook","cit-hepth","cit-hepph","epinions","twitter","gplus","dblp","youtube","flickr","livejournal"],
}

ALL_METHODS = ["louvain-static","qca","gremod","batch","leiden-static","slm-static","df-louvain","ds-louvain","dynamo","df-leiden"]


def resolve_datasets(sel: str) -> list[str]:
    out: list[str] = []
    for tok in sel.split(","):
        tok = tok.strip()
        if not tok: continue
        if tok in DATASET_GROUPS:
            out.extend(DATASET_GROUPS[tok])
        else:
            out.append(tok)
    # dedupe preserving order
    seen = set(); uniq = []
    for d in out:
        if d not in seen: seen.add(d); uniq.append(d)
    return uniq


def prep(datasets: list[str]) -> None:
    for d in datasets:
        meta_p = RUNS / d / "meta.json"
        if meta_p.exists():
            print(f"[prep:{d}] cached")
            continue
        edges = EDGES / f"{d}.tsv"
        if not edges.exists():
            print(f"[prep:{d}] MISSING EDGES {edges}")
            continue
        print(f"[prep:{d}] running")
        subprocess.run([sys.executable, str(HARNESS/"prep_data.py"), "--name", d], check=True)

def ensure_initcoms(datasets: list[str]) -> None:
    """Pre-compute initial_com.txt for each dataset SERIALLY to avoid race
    between parallel cells that all call ensure_initial_community()."""
    # Use run.py's helper directly by importing it
    sys.path.insert(0, str(HARNESS))
    from run import ensure_initial_community  # noqa
    for d in datasets:
        ic = RUNS / d / "initial_com.txt"
        if ic.exists():
            print(f"[initcom:{d}] cached")
            continue
        print(f"[initcom:{d}] running Louvain on initial graph")
        ensure_initial_community(d)


def cell_already_done(method: str, dataset: str) -> bool:
    pb = PER_BATCH / f"{method}__{dataset}.tsv"
    meta_p = RUNS / dataset / "meta.json"
    if not pb.exists() or not meta_p.exists(): return False
    meta = json.loads(meta_p.read_text())
    # check last batch present in pb
    try:
        lines = pb.read_text().splitlines()
        if len(lines) < 2: return False
        last_b = int(lines[-1].split("\t")[0])
        return last_b >= meta["n_batches"]
    except Exception:
        return False


def run_cell(method: str, dataset: str, timeout: int) -> tuple[str, str, float, str]:
    t0 = time.time()
    if cell_already_done(method, dataset):
        return method, dataset, time.time()-t0, "cached"
    cmd = [sys.executable, str(HARNESS/"run.py"), "--method", method, "--dataset", dataset, "--timeout", str(timeout)]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout+30)
        # last status line
        out = r.stdout.decode(errors="replace")
        status = "ok" if r.returncode == 0 else f"exit={r.returncode}"
        # try to parse out the status from the run.py print
        for line in out.splitlines():
            if line.startswith(f"[{method}/{dataset}]"):
                # contains "ok in X.Ys" or "TLE in ..."
                parts = line.split()
                if len(parts) >= 2:
                    status = parts[1]
        return method, dataset, time.time()-t0, status
    except subprocess.TimeoutExpired:
        return method, dataset, time.time()-t0, "outer-TLE"
    except Exception as e:
        return method, dataset, time.time()-t0, f"err:{type(e).__name__}"


def cmd_run(args):
    datasets = resolve_datasets(args.datasets)
    methods  = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    cells = [(m, d) for d in datasets for m in methods]
    # Prep all datasets first (sequential, fast)
    prep(datasets)
    # Pre-compute initial_com.txt only if a selected method actually needs it
    # (qca/gremod/batch). Static refs + dynamo + C++ drivers build their own
    # initial partition, so skip the (potentially very expensive) Louvain initcom.
    sys.path.insert(0, str(HARNESS))
    from run import METHODS as _M  # noqa
    if any(_M.get(m, {}).get("needs_initcom") for m in methods):
        ensure_initcoms(datasets)
    else:
        print("[sweep] no method needs initcom; skipping initcom precompute")
    print(f"[sweep] {len(cells)} cells, max_parallel={args.max_parallel}, timeout={args.timeout}s")
    done = 0
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_cell, m, d, args.timeout): (m, d) for m, d in cells}
        for f in as_completed(futs):
            m, d, dt, st = f.result()
            done += 1
            print(f"[{done}/{len(cells)}] {m}/{d}  {st}  in {dt:.1f}s", flush=True)


def cmd_prep(args):
    datasets = resolve_datasets(args.datasets)
    prep(datasets)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prep = sub.add_parser("prep")
    ap_prep.add_argument("--datasets", default="all")
    ap_prep.set_defaults(func=cmd_prep)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--datasets", default="small")
    ap_run.add_argument("--methods",  default="all")
    ap_run.add_argument("--max-parallel", type=int, default=8)
    ap_run.add_argument("--timeout", type=int, default=4*3600)
    ap_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
