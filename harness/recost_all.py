#!/usr/bin/env python3
"""Recompute in parallel the re-summarization cost of every (dataset, method) under the current configuration.

Background: the Cost column of tab:main matched the on-disk resumm_hier.csv only half of the time
(on Facebook DynaMo/DF-Leiden/Delta-screen/DF-Louvain/GreMod/DyLo-Louvain agreed,
but From-scratch Leiden, Batch, QCA, HIT-Leiden, DyLo-SLM, DyLo-Leiden did not),
showing that the table mixed two generations of measurements. This script recomputes the Cost of all ten datasets with one accounting.

The cost is pure CPU post-processing: every method's per-batch partitions are already stored in
data/runs/<dataset>/out_<method>/com_<k>[_L<i>].txt and only need re-aggregation per def:cost,
without rerunning any detection algorithm. Hence it parallelizes over (dataset, method).

The accounting is identical to run.py / emit_resumm_hier.py:
  - methods with a real hierarchy (dylo-*, *-hier, hit-leiden) use the hierarchical cost stream_hier_cost;
  - flat methods use the leaf cost stream_cost, a lower bound on their hierarchical cost
    (a flat partition gets no amortization of shared ancestors, so mapping it onto a tree can only cost more).

Usage: recost_all.py [--workers N] [--out path] [--only dataset,...]
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

H = Path(DYLO_ROOT + "/harness")
RUNS = Path(DYLO_ROOT + "/data/runs")
sys.path.insert(0, str(H))

# methods with a real multi-level dendrogram use the hierarchical cost; the rest use the flat leaf cost
HIER = {"dylo-louvain", "dylo-slm", "dylo-leiden",
        "leiden-hier", "louvain-hier", "hit-leiden"}


def n_batches(out_dir: Path) -> int:
    k = 0
    while (out_dir / f"com_{k}.txt").exists() or (out_dir / f"com_{k}_L0.txt").exists():
        k += 1
    return k - 1


def one(ds: str, method: str) -> dict:
    """Average cost of one (dataset, method). Imports inside the subprocess so the main process holds no large objects."""
    from dirty_cost import stream_cost
    from hier_cost import stream_hier_cost

    out = RUNS / ds / f"out_{method}"
    t0 = time.time()
    nb = n_batches(out)
    if nb < 1:
        return dict(dataset=ds, method=method, status="no-batches", batches=0)

    rec = dict(dataset=ds, method=method, batches=nb, status="ok")
    try:
        if method in HIER and (out / "com_1_L0.txt").exists():
            rows = stream_hier_cost(out, nb)
            rec["kind"] = "hier"
        else:
            rows = stream_cost(out, nb)
            rec["kind"] = "flat"
        if not rows:
            return dict(rec, status="empty")
        rec["cost_norm"] = sum(r[2] for r in rows) / len(rows)
        rec["cost_abs"] = sum(r[1] for r in rows) / len(rows)
        rec["n_used"] = len(rows)
    except Exception as e:  # noqa: BLE001
        return dict(rec, status=f"error: {type(e).__name__}: {str(e)[:80]}")
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=DYLO_ROOT + "/results/resumm_recost.csv")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    jobs = []
    only = set(args.only.split(",")) if args.only else None
    for dsdir in sorted(RUNS.iterdir()):
        if not dsdir.is_dir():
            continue
        if only and dsdir.name not in only:
            continue
        for od in sorted(dsdir.glob("out_*")):
            jobs.append((dsdir.name, od.name[len("out_"):]))
    print(f"{len(jobs)} (dataset, method) jobs, {args.workers} in parallel", flush=True)

    res = []
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, d, m): (d, m) for d, m in jobs}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            res.append(r)
            done += 1
            if r["status"] == "ok":
                print(f"  [{done}/{len(jobs)}] {r['dataset']:<12} {r['method']:<18} "
                      f"{r['kind']:<5} batches={r['batches']:<5} "
                      f"cost={r['cost_norm']:.6f} ({r.get('sec',0)}s)", flush=True)
            else:
                print(f"  [{done}/{len(jobs)}] {r['dataset']:<12} {r['method']:<18} "
                      f"{r['status']}", flush=True)

    res.sort(key=lambda r: (r["dataset"], r["method"]))
    with open(args.out, "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=["dataset", "method", "kind", "batches",
                                          "n_used", "cost_norm", "cost_abs", "status", "sec"])
        w.writeheader()
        for r in res:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    ok = sum(1 for r in res if r["status"] == "ok")
    print(f"\ndone {ok}/{len(res)} in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
