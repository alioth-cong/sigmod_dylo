#!/usr/bin/env python3
"""Weight-model sensitivity: same partitions, same dirty sets, only the node-weight function changes; one pass computes every accounting.

Accountings (n = subtree entity count; for a leaf, its member count):
  count     the paper's current model: leaf = n, internal = number of children
  subtree   leaf = n, internal = n (only internal nodes switched to counting subtree entities)
  alpha     A + B*n (fixed overhead + linear, no cap)
  cap4k     min(CAP(4000), A + B*n)   <- B1, the measured configuration max_input_length=4000
  cap8k     min(CAP(8000), A + B*n)   <- GraphRAG's default max_input_length=8000
  cap16k    min(CAP(16000), A + B·n)
CAP(M) = 1.0713*(1928 + M) - 73.2 (template + context cap, converted to Qwen tokens;
6278 at 4000, consistent with the fitted 6239).
DyLo runs are scored with two structures: whole (the whole tree) and leaf (leaf level only, the same structure as flat baselines).
"""
import os as _os  # artifact: paths are relative to the repository root
DYLO_ROOT = _os.environ.get('DYLO_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DYLO_MHRAG = _os.environ.get('DYLO_MHRAG', _os.path.join(DYLO_ROOT, 'mhrag'))
DYLO_SCRATCH = _os.environ.get('DYLO_SCRATCH', '/tmp/dylo')
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hier_cost import _parents, _self_dirty  # noqa: E402
from hier_cost_w import read_any  # noqa: E402

A, B = 2026.0, 89.54
CAP = lambda M: 1.0713 * (1928 + M) - 73.2  # noqa: E731
W = {
    "count": None,
    "subtree": lambda n: float(n),
    "alpha": lambda n: A + B * n,
    "cap4k": lambda n: min(CAP(4000), A + B * n),
    "cap8k": lambda n: min(CAP(8000), A + B * n),
    "cap16k": lambda n: min(CAP(16000), A + B * n),
}
RUNS = Path(DYLO_ROOT + "/data/runs")
L4 = Path(DYLO_ROOT + "/tmpruns/layer4")
HIER = {"hit-leiden"}


def batch(prev, curr):
    nlev = len(curr)
    sd = [_self_dirty(prev[i] if (prev and i < len(prev)) else [], curr[i]) for i in range(nlev)]
    par = _parents(curr)
    nch = [[0] * len(curr[i]) for i in range(nlev)]
    for i in range(nlev - 1):
        for c, p in enumerate(par[i]):
            if p >= 0:
                nch[i + 1][p] += 1
    dirty = [list(x) for x in sd]
    for i in range(nlev - 1):
        for c in range(len(curr[i])):
            if dirty[i][c] and par[i][c] >= 0:
                dirty[i + 1][par[i][c]] = True
    cost = {k: 0.0 for k in W}
    tot = {k: 0.0 for k in W}
    for i in range(nlev):
        for c in range(len(curr[i])):
            n = len(curr[i][c])
            for k, f in W.items():
                w = (n if i == 0 else nch[i][c]) if f is None else f(n)
                tot[k] += w
                if dirty[i][c]:
                    cost[k] += w
    return {k: cost[k] / tot[k] if tot[k] else 0.0 for k in W}


def score(out: Path, nb: int, hier: bool) -> dict:
    acc = {k: 0.0 for k in W}
    n, prev = 0, None
    for k in range(nb + 1):
        curr = read_any(out, k, hier)
        if not curr:
            break
        if prev is not None:
            for kk, v in batch(prev, curr).items():
                acc[kk] += v
            n += 1
        prev = curr
    return {k: v / n for k, v in acc.items()} if n else {}


def job(a):
    label, ds, out, hier = a
    nb = json.loads((RUNS / ds / "meta.json").read_text())["n_batches"]
    return dict(dataset=ds, method=label, **score(Path(out), nb, hier))


def main():
    dss = sys.argv[2].split(",")
    jobs = []
    for ds in dss:
        for m in ["dynamo", "batch", "qca", "df-leiden", "ds-louvain", "df-louvain", "gremod", "hit-leiden"]:
            jobs.append((m, ds, str(RUNS / ds / f"out_{m}"), m in HIER))
        for inst in ["leiden", "louvain", "slm"]:
            d = L4 / f"{ds}__{inst}__old__lam1e-1"
            if d.exists():
                jobs.append((f"dylo-{inst}-whole", ds, str(d), True))
                jobs.append((f"dylo-{inst}-leaf", ds, str(d), False))
    with ProcessPoolExecutor(64) as ex:
        res = list(ex.map(job, jobs))
    with open(sys.argv[1], "w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=["dataset", "method", *W])
        w.writeheader()
        for r in res:
            w.writerow(r)
    print(f"{len(res)} rows -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
