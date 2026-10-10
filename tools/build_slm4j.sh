#!/usr/bin/env bash
# Build SLM4J (Waltman & van Eck "Modularity Optimizer", org.slm4j.ModularityOptimizer;
# optimization_algorithm=3 = genuine SLM) into external/slm4j/build, WITHOUT touching the
# pristine submodule source. ModularityOptimizer.java carries an optional JUNG-based
# helper (readJUNGGraph) that the file-CLI path never uses but whose import breaks
# compilation; we strip it from a staged copy and compile that.
set -euo pipefail
cd "$(dirname "$0")/.."
. .envrc 2>/dev/null || true
S=external/slm4j
SRC=$S/src/main/java/org/slm4j
STAGE=$S/_build_src
rm -rf "$STAGE" "$S/build"
mkdir -p "$STAGE/org/slm4j" "$S/build"

for f in Network Clustering VOSClusteringTechnique Arrays2; do
  cp "$SRC/$f.java" "$STAGE/org/slm4j/"
done

python3 - "$SRC/ModularityOptimizer.java" "$STAGE/org/slm4j/ModularityOptimizer.java" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
src = src.replace('import edu.uci.ics.jung.graph.DirectedOrderedSparseMultigraph;\n', '')
src = src.replace('import edu.uci.ics.jung.graph.util.Pair;\n', '')
# drop the unused JUNG helper (signature .. matching closing brace at method indent)
src = re.sub(r'\n[ \t]*public static <V, E> Network readJUNGGraph\(.*?\n[ \t]{4}\}\n',
             '\n', src, flags=re.S)
open(sys.argv[2], 'w').write(src)
PY

javac -d "$S/build" "$STAGE"/org/slm4j/*.java
rm -rf "$STAGE"
echo "DONE: $(ls "$S"/build/org/slm4j/*.class | wc -l) classes in $S/build"
