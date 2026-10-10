#!/usr/bin/env bash
# Smoke test: Facebook stream, DyLo-Leiden, paper default configuration.
# Checks mean Q and mean normalized re-summarization cost against results/sel_ablation.csv.
set -euo pipefail
ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
make -s harness/cpp/dylo/dylo_driver_std4
mkdir -p data/raw data/edges
if [[ ! -f data/edges/facebook.tsv ]]; then
  curl -fsSL -o data/raw/facebook_combined.txt.gz https://snap.stanford.edu/data/facebook_combined.txt.gz
  python3 - <<'PY'
import gzip
E = set()
for line in gzip.open("data/raw/facebook_combined.txt.gz", "rt"):
    p = line.split()
    if len(p) >= 2 and not line.startswith("#"):
        u, v = int(p[0]), int(p[1])
        if u != v:
            E.add((min(u, v), max(u, v)))
with open("data/edges/facebook.tsv", "w") as g:
    for u, v in E:
        g.write(f"{u}\t{v}\n")
PY
fi
[[ -f data/runs/facebook/meta.json ]] || python3 harness/prep_data.py --name facebook
OUT="${DYLO_SCRATCH:-/tmp/dylo}/smoke_facebook"; mkdir -p "$OUT"
NB=$(python3 -c "import json;print(json.load(open('data/runs/facebook/meta.json'))['n_batches'])")
DYLO_EMIT_HIER=1 DYLO_LAMBDA=1e-1 DYLO_AUDIT_K=8 DYLO_REBUILD_R=1e18 DYLO_SIGMA_ONLY=1 \
DYLO_SIGMA=0.2 DYLO_FINAL_AUDIT=1 DYLO_TOPDOWN=0 DYLO_HIER_GAMMA=0.1 DYLO_PRUNE=1 \
DYLO_PRUNE_MODE=leaf DYLO_DEFER=1 DYLO_DEFER_EPS=0.05 DYLO_LAZY_CERT=1 DYLO_INC_TREE=1 \
DYLO_LAZY_RECLAIM=1 DYLO_RECLAIM=1 DYLO_FAST_RECLAIM=1 DYLO_NO_COM=1 \
  harness/cpp/dylo/dylo_driver_std4 leiden data/runs/facebook/initial.txt data/runs/facebook/inc.txt "$NB" "$OUT" 2>/dev/null >/dev/null
python3 - "$OUT" <<'PY'
import csv, statistics, sys
o = sys.argv[1]
q = statistics.mean(float(r.split("\t")[2]) for r in open(f"{o}/metrics.tsv").readlines()[1:])
c = statistics.mean(float(r.split("\t")[5]) for r in open(f"{o}/resum.tsv").readlines()[1:])
ref = next(r for r in csv.DictReader(open("results/sel_ablation.csv"))
           if r["dataset"] == "facebook" and r["mode"] == "dp" and r["lam"] == "1e-1")
rq, rc = float(ref["Q"]), float(ref["cost"])
print(f"mean Q    {q:.5f}  (paper run {rq:.5f})")
print(f"mean cost {c:.5f}  (paper run {rc:.5f})")
ok = abs(q - rq) < 1e-4 and abs(c - rc) < 1e-3
print("SMOKE TEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
