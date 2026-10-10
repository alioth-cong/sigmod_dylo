#!/usr/bin/env python3
"""
Uniform harness — drive a (method, dataset) pair end-to-end and append rows to per-metric CSVs.

Methods registered below. Each is invoked through a launcher that writes:
  metrics.tsv (batch, latency_ms, modularity, num_communities)
  com_0.txt..com_K.txt (one community per line, tab-separated node IDs)

Outputs (appended, long-format):
  results/latency.csv   method,dataset,value,unit,language,repo,commit,notes
  results/quality.csv   method,dataset,value,unit,language,repo,commit,notes
  results/resumm.csv    method,dataset,value,unit,language,repo,commit,notes
  results/per_batch/<method>__<dataset>.tsv   batch,latency_ms,modularity,num_communities,cost_abs,cost_norm
"""
from __future__ import annotations
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(DYLO_ROOT + "")
DATA      = ROOT / "data"
RUNS      = DATA / "runs"
LOGS      = ROOT / "logs" / "runs"
RESULTS   = ROOT / "results"
PER_BATCH = RESULTS / "per_batch"

HARNESS   = ROOT / "harness"
JAVA_BIN  = Path(os.environ.get("JAVA_HOME", os.path.expanduser("~/.local/opt/jdk-21.0.5+11"))) / "bin" / "java"
DRIVER_CLASS = "BaselineDriver"

METHODS = {
    "louvain-static": dict(language="Java",   repo="nogrady/dynamo",                   commit="4a94453c", notes="static reference; re-run per batch; flat",        launcher="java-driver", driver_arg="louvain-static",  needs_initcom=False),
    "qca":            dict(language="Java",   repo="nogrady/dynamo",                   commit="4a94453c", notes="incremental; flat",                                launcher="java-driver", driver_arg="qca",             needs_initcom=True),
    "gremod":         dict(language="Java",   repo="nogrady/dynamo",                   commit="4a94453c", notes="incremental insertion-only; flat",                 launcher="java-driver", driver_arg="gremod",          needs_initcom=True),
    "batch":          dict(language="Java",   repo="nogrady/dynamo",                   commit="4a94453c", notes="batch incremental; flat",                          launcher="java-driver", driver_arg="batch",           needs_initcom=True),
    "leiden-static":  dict(language="Python", repo="vtraag/leidenalg",                 commit="c08884a4", notes="static reference; re-run per batch; flat",        launcher="py-script",   script="leiden_static.py"),
    "slm-static":     dict(language="Java",   repo="deepminder/SLM4J",                 commit="eed958f6", notes="genuine SLM (Smart Local Moving, Waltman & van Eck 2013) via SLM4J Modularity Optimizer algo=3; from-scratch per batch; flat", launcher="py-script", script="slm_static.py"),
    "df-louvain":     dict(language="C++",    repo="puzzlef/louvain-communities-openmp-dynamic", commit="a0469c8b", notes="dynamic frontier Louvain; OpenMP; flat",     launcher="binary",      binary=str(ROOT/"harness/cpp/dylo_driver"), binary_args=["df"]),
    "ds-louvain":     dict(language="C++",    repo="puzzlef/louvain-communities-openmp-dynamic", commit="a0469c8b", notes="dynamic delta-screening Louvain; OpenMP; flat", launcher="binary",      binary=str(ROOT/"harness/cpp/dylo_driver"), binary_args=["ds"]),
    "dynamo":         dict(language="Java",   repo="nogrady/dynamo",                   commit="4a94453c", notes="DynaMo (Zhuang et al 2021); incremental modularity; flat", launcher="java-driver", driver_arg="dynamo", needs_initcom=False),
    "df-leiden":      dict(language="C++",    repo="puzzlef/leiden-communities-openmp-dynamic", commit="28c304fe", notes="Dynamic Frontier Leiden (arXiv:2405.11658); OpenMP; flat", launcher="binary", binary=str(ROOT/"harness/cpp/dylo_leiden_driver"), binary_args=["df"]),
    "louvain-hier":   dict(language="Python", repo="igraph/python-igraph",                commit="v1.0.0",   notes="from-scratch Louvain with full dendrogram; HIERARCHICAL re-summarization cost (P1)", launcher="py-script", script="louvain_hier.py"),
    "leiden-hier":    dict(language="Python", repo="vtraag/leidenalg",                    commit="c08884a4", notes="from-scratch Leiden with full dendrogram (aggregation passes); HIERARCHICAL re-summarization cost (P1)", launcher="py-script", script="leiden_hier.py"),
    "dylo-louvain":   dict(language="C++",    repo="this-artifact (harness/cpp/dylo)",      commit="0",        notes="DyLo cost-aware incremental Louvain (Alg 1 affected-region + open-set DP, Alg 2). Refine=identity. Tracks louvain-cpp-static (same C++ base, from-scratch per batch) within eps; facebook ~25x same-impl speedup (1.9 vs 47.3 ms/batch); ~157x vs Java louvain-static. lambda via DYLO_LAMBDA env (default 0).", launcher="binary", binary=str(ROOT/"harness/cpp/dylo/dylo_driver"), binary_args=["louvain"]),
    # Fair from-scratch references: DyLo's OWN C++ base detector re-run per batch (no
    # incremental state). Same code path as dylo-* minus the incremental driver, so the
    # quality+speedup comparison is same-implementation (not C++ vs Java louvain-static).
    "louvain-cpp-static": dict(language="C++", repo="this-artifact (harness/cpp/dylo)",     commit="0",        notes="from-scratch Louvain via DyLo's C++ base detector (detect_iter), re-run per batch; fair same-impl baseline for dylo-louvain", launcher="binary", binary=str(ROOT/"harness/cpp/dylo/dylo_driver"), binary_args=["louvain-fs"]),
    "slm-cpp-static":     dict(language="C++", repo="this-artifact (harness/cpp/dylo)",     commit="0",        notes="from-scratch SLM via DyLo's C++ base detector (detect_iter), re-run per batch; fair same-impl baseline for dylo-slm", launcher="binary", binary=str(ROOT/"harness/cpp/dylo/dylo_driver"), binary_args=["slm-fs"]),
    "leiden-cpp-static":  dict(language="C++", repo="this-artifact (harness/cpp/dylo)",     commit="0",        notes="from-scratch Leiden via DyLo's C++ base detector (detect_iter), re-run per batch; fair same-impl baseline for dylo-leiden", launcher="binary", binary=str(ROOT/"harness/cpp/dylo/dylo_driver"), binary_args=["leiden-fs"]),
    "dylo-slm":       dict(language="C++",    repo="this-artifact (harness/cpp/dylo)",      commit="0",        notes="DyLo cost-aware incremental SLM (Alg 1 + open-set DP). Refine=full within-community local-moving (Smart Local Moving, Waltman & van Eck 2013) + aggregate-init lift. Detect Q matches slm-static within eps<=0.005 on 6 datasets. lambda via DYLO_LAMBDA env (default 0).", launcher="binary", binary=str(ROOT/"harness/cpp/dylo/dylo_driver"), binary_args=["slm"]),
    "dylo-leiden":    dict(language="C++",    repo="this-artifact (harness/cpp/dylo)",      commit="0",        notes="DyLo cost-aware incremental Leiden (Alg 1 + open-set DP). Refine=connectedness-gated single-pass merge (Traag/van Eck/Waltman 2019) + aggregate-init lift -> well-connected sub-communities. Detect Q matches leiden-static within eps<=0.005 on 6 datasets. lambda via DYLO_LAMBDA env (default 0).", launcher="binary", binary=str(ROOT/"harness/cpp/dylo/dylo_driver"), binary_args=["leiden"]),
}


def ensure_initial_community(dataset: str) -> Path:
    """Produce data/runs/<dataset>/initial_com.txt by running Louvain on initial.txt.
       Cached if already present."""
    run_dir = RUNS / dataset
    out = run_dir / "initial_com.txt"
    if out.exists():
        return out

    harness_build = HARNESS / "java" / "build"
    cp = (harness_build / ".classpath").read_text().strip()
    classpath = f"{harness_build}:{cp}"

    tmp_dir = run_dir / "_initcom"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    args = [str(JAVA_BIN), "-Xmx128g", "-cp", classpath, DRIVER_CLASS,
            "louvain-static", str(run_dir/"initial.txt"), str(run_dir/"inc.txt"), "0", str(tmp_dir)]
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"init_com__{dataset}.log"
    print(f"[init_com:{dataset}] running Louvain on initial.txt")
    with open(log_path, "w") as logf:
        subprocess.run(args, stdout=logf, stderr=subprocess.STDOUT, check=True)
    shutil.copy(tmp_dir / "com_0.txt", out)
    shutil.rmtree(tmp_dir)
    return out


def launch(method: str, dataset: str, timeout_s: int, max_batches: int = 0) -> tuple[Path, dict]:
    run_dir = RUNS / dataset
    out_dir = run_dir / f"out_{method}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    meta = json.loads((run_dir/"meta.json").read_text())
    n_batches = meta["n_batches"]
    # LiveJournal's native stream is 4286 batches x 4.85M nodes, 37MB of com files per batch --
    # running one method fully takes ~100GB, twelve methods do not fit, and leiden-static reruns the whole graph per batch.
    # It is cut to the first N batches, of the same order as Flickr (320) / YouTube (299), with unchanged cost accounting.
    if max_batches and max_batches < n_batches:
        n_batches = max_batches
    initial = run_dir / "initial.txt"
    inc_base = run_dir / "inc.txt"

    info = METHODS[method]
    subprocess_cwd: Path | None = None
    if info["launcher"] == "java-driver":
        harness_build = HARNESS / "java" / "build"
        cp = (harness_build / ".classpath").read_text().strip()
        classpath = f"{harness_build}:{cp}"
        cmd = [str(JAVA_BIN), "-Xmx128g", "-cp", classpath, DRIVER_CLASS, info["driver_arg"], str(initial)]
        if info.get("needs_initcom"):
            initial_com = ensure_initial_community(dataset)
            cmd.append(str(initial_com))
        cmd += [str(inc_base), str(n_batches), str(out_dir)]
        if info["driver_arg"] == "dynamo":
            # DynaMo uses cwd-relative data/<tag>/... ; isolate by running each cell in its own dir.
            subprocess_cwd = out_dir
    elif info["launcher"] == "py-script":
        script = HARNESS / info["script"]
        cmd = [sys.executable, str(script)] + info.get("script_args", []) + [
            str(initial), str(inc_base), str(n_batches), str(out_dir)
        ]
    elif info["launcher"] == "binary":
        # binary <args...> <initial> <inc-base> <numBatches> <outDir> <numThreads>
        threads = os.environ.get("DYLO_THREADS", "1")
        cmd = [info["binary"]] + info.get("binary_args", []) + [
            str(initial), str(inc_base), str(n_batches), str(out_dir), threads
        ]
    else:
        raise SystemExit(f"unknown launcher for {method}")

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{method}__{dataset}.log"
    mem_path = LOGS / f"{method}__{dataset}.mem.txt"
    cmd = ["/usr/bin/time", "-v", "-o", str(mem_path)] + cmd
    status = "ok"
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        try:
            cp_res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=timeout_s, cwd=subprocess_cwd)
            if cp_res.returncode != 0:
                status = f"exit={cp_res.returncode}"
        except subprocess.TimeoutExpired:
            status = "TLE"
        except Exception as e:
            status = f"err:{type(e).__name__}"
    elapsed = time.time() - t0
    peak_rss_kb = None
    if mem_path.exists():
        for line in mem_path.read_text().splitlines():
            if "Maximum resident set size" in line:
                try:
                    peak_rss_kb = int(line.strip().split()[-1])
                except ValueError:
                    pass
    print(f"[{method}/{dataset}] {status} in {elapsed:.1f}s peak_rss={peak_rss_kb}KB log={log_path}")
    return out_dir, dict(status=status, log=str(log_path), n_batches=n_batches, elapsed_s=elapsed,
                          peak_rss_kb=peak_rss_kb)


def aggregate(method: str, dataset: str, out_dir: Path, run_meta: dict) -> dict:
    sys.path.insert(0, str(HARNESS))
    from dirty_cost import stream_cost  # noqa
    metrics_tsv = out_dir / "metrics.tsv"
    if not metrics_tsv.exists():
        return dict(method=method, dataset=dataset, status="missing-metrics")
    rows = []
    with open(metrics_tsv) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4: continue
            rows.append((int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))
    n_batches = run_meta["n_batches"]
    cost_rows = stream_cost(out_dir, n_batches)
    cost_by_b = {r[0]: (r[1], r[2]) for r in cost_rows}

    PER_BATCH.mkdir(parents=True, exist_ok=True)
    pb_path = PER_BATCH / f"{method}__{dataset}.tsv"
    with open(pb_path, "w") as g:
        g.write("batch\tlatency_ms\tmodularity\tnum_communities\tcost_abs\tcost_norm\n")
        for b, ms, mod, nc in rows:
            ca, cn = cost_by_b.get(b, (float("nan"), float("nan")))
            g.write(f"{b}\t{ms}\t{mod}\t{nc}\t{ca}\t{cn}\n")

    # P1: if the method emitted a dendrogram (com_<k>_L0.txt), also compute the
    # hierarchical re-summarization cost and write a per_batch_hier tsv.
    if (out_dir / "com_0_L0.txt").exists():
        from hier_cost import stream_hier_cost  # noqa
        hrows = stream_hier_cost(out_dir, n_batches)
        if hrows:
            PB_HIER = RESULTS / "per_batch_hier"
            PB_HIER.mkdir(parents=True, exist_ok=True)
            with open(PB_HIER / f"{method}__{dataset}.tsv", "w") as g:
                g.write("batch\tcost_abs_hier\tcost_norm_hier\n")
                for b, ca, cn in hrows:
                    g.write(f"{b}\t{ca}\t{cn}\n")

    batch_rows = [r for r in rows if r[0] >= 1]
    avg_lat = sum(r[1] for r in batch_rows) / max(1, len(batch_rows))
    finite_mods = [r[2] for r in batch_rows if r[2] == r[2]]  # not NaN
    avg_mod = (sum(finite_mods) / max(1, len(finite_mods))) if finite_mods else float("nan")
    avg_cost_abs  = sum(cost_by_b.get(b,(0,0))[0] for b,*_ in batch_rows) / max(1, len(batch_rows))
    avg_cost_norm = sum(cost_by_b.get(b,(0,0))[1] for b,*_ in batch_rows) / max(1, len(batch_rows))

    return dict(
        method=method, dataset=dataset,
        avg_latency_ms=avg_lat, avg_modularity=avg_mod,
        avg_cost_abs=avg_cost_abs, avg_cost_norm=avg_cost_norm,
        batches=len(batch_rows),
    )


def append_csv(csv_path: Path, header: list[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a") as g:
        w = csv.writer(g)
        if new: w.writerow(header)
        w.writerow([row.get(h, "") for h in header])


def emit_long_csvs(method: str, dataset: str, summary: dict, status: str):
    meta = METHODS[method]
    base_notes = meta["notes"] + f"; status={status}"
    common = dict(method=method, dataset=dataset, language=meta["language"], repo=meta["repo"], commit=meta["commit"])
    header = ["method", "dataset", "value", "unit", "language", "repo", "commit", "notes"]
    append_csv(RESULTS/"latency.csv",  header, {**common, "value": summary.get("avg_latency_ms",""), "unit": "ms/batch",                      "notes": base_notes})
    append_csv(RESULTS/"quality.csv",  header, {**common, "value": summary.get("avg_modularity",""), "unit": "Q (modularity)",                "notes": base_notes})
    append_csv(RESULTS/"resumm.csv",   header, {**common, "value": summary.get("avg_cost_norm",""),  "unit": "fraction of partition (dirty)", "notes": base_notes + f"; abs={summary.get('avg_cost_abs','')}"})
    peak_kb = summary.get("peak_rss_kb", "")
    peak_mb = f"{peak_kb/1024:.1f}" if isinstance(peak_kb, (int, float)) else ""
    append_csv(RESULTS/"memory.csv",   header, {**common, "value": peak_mb, "unit": "peak RSS (MB)",                                          "notes": base_notes})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS.keys()))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--timeout", type=int, default=4*3600)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = native batch count from meta")
    args = ap.parse_args()
    out_dir, run_meta = launch(args.method, args.dataset, args.timeout, args.max_batches)
    if run_meta["status"] != "ok":
        emit_long_csvs(args.method, args.dataset, {"peak_rss_kb": run_meta.get("peak_rss_kb")}, run_meta["status"])
        sys.exit(0)
    summary = aggregate(args.method, args.dataset, out_dir, run_meta)
    summary["status"] = "ok"
    summary["peak_rss_kb"] = run_meta.get("peak_rss_kb")
    print(json.dumps(summary, indent=2, default=str))
    emit_long_csvs(args.method, args.dataset, summary, "ok")


if __name__ == "__main__":
    main()
