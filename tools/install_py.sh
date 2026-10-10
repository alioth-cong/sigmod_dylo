#!/usr/bin/env bash
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
# User-local Python venv with leidenalg/igraph/networkit + analysis stack.
set -euo pipefail

cd ${DYLO_ROOT}

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel
python -m pip install -q \
  numpy pandas scipy networkx \
  python-igraph leidenalg networkit \
  tqdm pyyaml

python - <<'PY'
import leidenalg, igraph, networkit, networkx, numpy, pandas
print(f"[verify] leidenalg={leidenalg.__version__} igraph={igraph.__version__} networkit={networkit.__version__} networkx={networkx.__version__}")
PY
touch ${DYLO_ROOT}/.venv/.ready
echo "DONE"
