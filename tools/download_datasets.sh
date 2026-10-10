#!/usr/bin/env bash
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
# Download all datasets to baselines/data/raw/, normalize to baselines/data/edges/<name>.tsv
# (undirected simple edge list, tab-separated, 0-or-1-indexed nodes preserved as-is).
# Records |V|, |E|, avg-degree, sha256 in baselines/data/manifest.tsv.
set -euo pipefail

ROOT="${DYLO_ROOT}"
DATA="$ROOT/data"
RAW="$DATA/raw"
EDGES="$DATA/edges"
MAN="$DATA/manifest.tsv"
mkdir -p "$RAW" "$EDGES"

if [[ ! -f "$MAN" ]]; then
  printf "name\tnodes\tedges\tavg_degree\tsha256_raw\tsource_url\n" > "$MAN"
fi

declare -A URL=(
  [facebook]='https://snap.stanford.edu/data/facebook_combined.txt.gz'
  [gplus]='https://snap.stanford.edu/data/gplus_combined.txt.gz'
  [twitter]='https://snap.stanford.edu/data/twitter_combined.txt.gz'
  [epinions]='https://snap.stanford.edu/data/soc-Epinions1.txt.gz'
  [livejournal]='https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz'
  [cit-hepph]='https://snap.stanford.edu/data/cit-HepPh.txt.gz'
  [cit-hepth]='https://snap.stanford.edu/data/cit-HepTh.txt.gz'
  [dblp]='https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz'
  [youtube]='https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz'
  [flickr]='https://nrvis.com/download/data/soc/soc-flickr.zip'
)

normalize() {
  # Read whitespace-separated edges, skip comment/header lines, drop self-loops & dupes
  # Output u<TAB>v with u<v.
  local in="$1" out="$2"
  python3 - "$in" "$out" <<'PY'
import sys, gzip, io, os
src, dst = sys.argv[1], sys.argv[2]
opener = gzip.open if src.endswith('.gz') else open
edges = set()
with opener(src, 'rt', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(('#', '%', '/')):
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) < 2:
            continue
        try:
            u, v = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if u == v: continue
        if u > v: u, v = v, u
        edges.add((u, v))
nodes = set()
for u,v in edges:
    nodes.add(u); nodes.add(v)
with open(dst, 'w') as g:
    for u,v in edges:
        g.write(f"{u}\t{v}\n")
n, m = len(nodes), len(edges)
sys.stderr.write(f"NODES={n} EDGES={m} AVG_DEG={(2*m/max(n,1)):.3f}\n")
PY
}

fetch_one() {
  local name="$1" url="$2"
  local fn raw_path
  fn=$(basename "$url")
  raw_path="$RAW/$fn"
  echo "============================================================"
  echo "[$name] $url"
  if [[ -f "$raw_path" ]]; then
    echo "[$name] already downloaded"
  else
    curl -fL --retry 3 -o "$raw_path.partial" "$url"
    mv "$raw_path.partial" "$raw_path"
  fi
  local sha
  sha=$(sha256sum "$raw_path" | awk '{print $1}')

  local out="$EDGES/$name.tsv"
  if [[ ! -f "$out" ]]; then
    if [[ "$fn" == *.zip ]]; then
      unzip -o "$raw_path" -d "$RAW/$name.zipdir" >/dev/null
      # pick the first .edges or .mtx or .txt file
      local inner
      inner=$(find "$RAW/$name.zipdir" -type f \( -name '*.edges' -o -name '*.mtx' -o -name '*.txt' \) | head -1)
      echo "[$name] inner: $inner"
      normalize "$inner" "$out" 2> "$EDGES/$name.stat"
    else
      normalize "$raw_path" "$out" 2> "$EDGES/$name.stat"
    fi
  fi
  # parse stats line
  local n m d
  read -r n m d < <(awk -F'[= ]' '{print $2, $4, $6}' "$EDGES/$name.stat")
  # append manifest only if not present
  if ! grep -q "^$name	" "$MAN"; then
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$n" "$m" "$d" "$sha" "$url" >> "$MAN"
  fi
  echo "[$name] OK n=$n m=$m avg_deg=$d sha256=${sha:0:12}"
}

# Order small->large so quick wins come first
for name in facebook cit-hepth cit-hepph epinions twitter gplus dblp youtube flickr livejournal; do
  fetch_one "$name" "${URL[$name]}"
done

echo
echo "=== MANIFEST ==="
column -t -s $'\t' "$MAN"
touch "$DATA/.datasets_ready"
echo "DONE"
