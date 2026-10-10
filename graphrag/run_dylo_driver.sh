#!/bin/bash
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
# Run DyLo's C++ driver on the edge stream extracted from the official snapshots, producing the hierarchies of two variants.
#
#   norecl -- no singleton reclamation (source-side merging off)
#   recl   -- the paper's std configuration, with structure-triggered source-side merging
#
# Note that DYLO_RECLAIM / DYLO_FAST_RECLAIM are both tested as getenv(...)!=nullptr,
# so writing =0 turns them on; to turn them off, leave both variables unset.
set -u
cd ${DYLO_ROOT}/graphrag
D=${DYLO_ROOT}/harness/cpp/dylo/dylo_driver_std4
S=${DYLO_ROOT}/graphrag/dylo_arm
NB=20

[ -f "$S/initial.txt" ] || { echo "edge stream $S/initial.txt missing, run dylo_stream.py first"; exit 1; }

# warm start: DyLo takes over GraphRAG's already-built leaf partition instead of cold-starting its own.
# Otherwise the two initial partitions are unrelated (measured 783 vs 6,564), summary inheritance never hits,
# and the measured cost is an artifact of the wiring.
BASE=${DYLO_ROOT}/grag_index_hotpot_base
PART=$S/init_part.txt
if [ ! -f "$PART" ]; then
  python export_leaf_part.py \
    $BASE/output/communities.parquet $BASE/output/entities.parquet \
    $S/id2name.json $PART || exit 1
fi

COMMON="DYLO_EMIT_HIER=1 DYLO_LAMBDA=1e-1 DYLO_AUDIT_K=8 DYLO_REBUILD_R=1e18 \
DYLO_SIGMA_ONLY=1 DYLO_SIGMA=0.2 DYLO_FINAL_AUDIT=1 DYLO_TOPDOWN=0 \
DYLO_HIER_GAMMA=0.1 DYLO_PRUNE=1 DYLO_PRUNE_MODE=leaf DYLO_DEFER=0 \
DYLO_LAZY_CERT=1 DYLO_INC_TREE=1 DYLO_LAZY_RECLAIM=1"

for v in norecl recl attach; do
  o=$S/$v; rm -rf $o; mkdir -p $o
  # attach uses a different binary: it only places new vertices and never moves existing ones,
  # so existing communities keep their member sets, 100% of reports can be inherited, and the cost is minimal.
  BIN=$D
  case "$v" in
    recl)   EXTRA="DYLO_RECLAIM=1 DYLO_FAST_RECLAIM=1" ;;
    attach) EXTRA="DYLO_ATTACH=1"
            BIN=${DYLO_ROOT}/harness/cpp/dylo/dylo_driver_attach ;;
    *)      EXTRA="" ;;
  esac
  t0=$(date +%s)
  env $COMMON $EXTRA DYLO_INIT_PART=$PART $BIN leiden $S/initial.txt $S/inc.txt $NB $o 2>$o/err >/dev/null
  t1=$(date +%s)
  # dylo_cluster.py reads both com_* and id2name.json from DYLO_DIR
  cp $S/id2name.json $o/
  echo "[$v] $((t1-t0))s  Q=$(tail -1 $o/metrics.tsv | cut -f3)  last-batch communities=$(tail -1 $o/metrics.tsv | cut -f4)  $(grep -o 'reclaims=[0-9]*' $o/err)"
done
echo DONE_DRIVER
