#!/usr/bin/env bash
# Clone the third-party baselines at the exact commits used in the paper into external/.
set -euo pipefail
ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$ROOT/external"
fetch() {  # name url commit
  local d="$ROOT/external/$1"
  [[ -d "$d/.git" ]] || git clone --quiet "$2" "$d"
  git -C "$d" checkout --quiet "$3"
  echo "$1 @ $(git -C "$d" rev-parse --short HEAD)"
}
fetch dynamo          https://github.com/nogrady/dynamo.git                                4a94453
fetch puzzlef-dynamic https://github.com/puzzlef/louvain-communities-openmp-dynamic.git    a0469c8
fetch puzzlef-leiden  https://github.com/puzzlef/leiden-communities-openmp-dynamic.git     28c304f
fetch slm4j           https://github.com/deepminder/SLM4J.git                              eed958f
fetch hit-leiden      https://github.com/randomvariable/hit-leiden.git                     9247424
pip install --quiet "leidenalg==0.10.2" "python-igraph>=0.11" || true
