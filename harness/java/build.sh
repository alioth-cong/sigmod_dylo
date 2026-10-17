#!/usr/bin/env bash
set -euo pipefail
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
if [[ -f "$DYLO_ROOT/.envrc" ]]; then source "$DYLO_ROOT/.envrc"; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
JAR_DIR="$HERE/lib"
mkdir -p "$JAR_DIR" "$HERE/build"

DYNAMO_JAR="${DYLO_ROOT}/external/dynamo/target/dynamic-1.0.jar"
FASTUTIL_JAR="$HOME/.m2/repository/it/unimi/dsi/fastutil/8.2.2/fastutil-8.2.2.jar"
LIBSVM_JAR="$HOME/.m2/repository/tw/edu/ntu/csie/libsvm/3.23/libsvm-3.23.jar"

CP="$DYNAMO_JAR:$FASTUTIL_JAR:$LIBSVM_JAR"

javac -d "$HERE/build" -cp "$CP" "$HERE/BaselineDriver.java"
echo "[driver] compiled BaselineDriver.class"
echo "$CP" > "$HERE/build/.classpath"
ls "$HERE/build/"
