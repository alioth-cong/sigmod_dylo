#!/usr/bin/env bash
set -euo pipefail
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
source ${DYLO_ROOT}/.envrc
cd ${DYLO_ROOT}/external/dynamo
echo "[dynamo] java=$(javac -version 2>&1) mvn=$(mvn -v 2>&1 | head -1)"
mvn -B -DskipTests -Dmaven.javadoc.skip=true --no-transfer-progress package
ls -la target/*.jar
touch ${DYLO_ROOT}/.dynamo_built
echo "DONE"
