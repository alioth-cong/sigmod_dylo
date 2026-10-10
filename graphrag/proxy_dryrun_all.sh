#!/bin/bash
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
# Tier-A dry run for every index under clean_chain: full indexes pass all communities; incremental indexes pass only the rewrite set by the patch rule.
C=${DYLO_MHRAG}/clean_chain
OUT=${DYLO_ROOT}/results/proxy_v2
PY=python
G=${DYLO_ROOT}/graphrag
one() {
  d=$1; name=$(basename $d)
  [ -f $d/output/communities.parquet ] || return 0
  [ -d $d/cache/community_reporting ] || return 0
  prev=$(ls -d $d/update_output/*/previous 2>/dev/null | sort | tail -1)
  if [ -n "$prev" ]; then extra="--changed-from $prev"; else extra=""; fi
  $PY $G/proxy_dryrun.py $d $extra --out $OUT/$name.csv > $OUT/$name.log 2>&1
  echo "$name: $(grep -o 'cache-key hits.*' $OUT/$name.log | head -1) $(grep -o 'rewrite set recomputed by the patch rule: [0-9]*' $OUT/$name.log)"
}
export -f one; export OUT PY G
ls -d $C/*/ | sed 's#/$##' | grep -vE 'mc[0-9]|^m-|arena|pack|judge|new60|am_|parts|gamma|config|logs|snapshots|/gs$|/m-' | xargs -P 24 -I{} bash -c 'one {}'
