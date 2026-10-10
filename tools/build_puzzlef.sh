#!/usr/bin/env bash
set -euo pipefail
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
cd ${DYLO_ROOT}/external/puzzlef-dynamic
echo "[puzzlef] g++ $(g++ --version | head -1)"
# Defaults from main.sh, but we mostly drive via our harness.
# Compile with insertion-only friendly defaults; we'll override via -D in harness if needed.
g++ -std=c++17 -O3 -fopenmp \
    -DTYPE=float \
    -DMAX_THREADS=64 \
    -DREPEAT_BATCH=1 -DREPEAT_METHOD=1 \
    -DBATCH_UNIT='"%"' -DBATCH_LENGTH=1 \
    -DBATCH_DELETIONS_BEGIN=0 -DBATCH_DELETIONS_END=0 -DBATCH_DELETIONS_STEP='*=2' \
    -DBATCH_INSERTIONS_BEGIN=0.001 -DBATCH_INSERTIONS_END=0.001 -DBATCH_INSERTIONS_STEP='*=2' \
    -DNUM_THREADS_MODE='"all"' -DNUM_THREADS_BEGIN=1 -DNUM_THREADS_END=1 -DNUM_THREADS_STEP='*=2' \
    -o louvain-dyn main.cxx
ls -la louvain-dyn
touch ${DYLO_ROOT}/.puzzlef_built
echo "DONE"
