#!/bin/bash
DYLO_ROOT="${DYLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DYLO_MHRAG="${DYLO_MHRAG:-$DYLO_ROOT/mhrag}"
DYLO_SCRATCH="${DYLO_SCRATCH:-/tmp/dylo}"
# Health check and self-healing for the vLLM GPU pool and its proxy.
#
# In long runs a backend can vanish through GPU-memory contention, driver faults, or being killed. Undetected, GraphRAG keeps
# retrying until 10 attempts are exhausted and then silently drops the community -- the pattern that once lost 812 reports:
# failures are hidden by retries and only a batch of data is missing from the output, without any error.
#
# Usage: health.sh check   (returns 0 = healthy)
#        health.sh heal    (repairs unhealthy backends and the proxy one by one)
set -u
# GPUs are detected dynamically: this machine once went from four GPUs to one (NVML enumerated only one even though lspci and
# /dev/nvidia0-3 were present); hard-coded GPU ids would make self-healing keep starting devices that do not exist.
declare -A GPU_PORT=()
_i=0
for _g in $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); do
  GPU_PORT[$_g]=$((8002 + _i)); _i=$((_i + 1))
done
[ ${#GPU_PORT[@]} -eq 0 ] && { echo "no GPU available"; exit 1; }
FRONT=8000
MODEL=Qwen/Qwen2.5-7B-Instruct
LOGDIR=/tmp/vllm_pool
mkdir -p $LOGDIR

up() { curl -s -m 5 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; }

start_backend() {
  local g=$1 p=$2
  echo "  [heal] restarting GPU $g -> :$p"
  local pids
  pids=$(ss -lntpH "sport = :$p" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ')
  [ -n "$pids" ] && kill $pids 2>/dev/null
  sleep 5
  # the GPU memory fraction can be overridden by VLLM_GPU_FRAC (default 0.45, restrained when sharing GPUs).
  # 0.45 leaves about 36G and a small KV cache: the map phase of global_search scores many communities one by one
  # and queues badly -- measured 8-14 s for normal questions and up to 420 s for some. Raising it helps when the GPU is idle.
  # logs are written per timestamp, otherwise each restart overwrites the previous cause of death.
  local frac=${VLLM_GPU_FRAC:-0.45}
  echo "  [heal] gpu-memory-utilization=$frac"
  setsid nohup env CUDA_VISIBLE_DEVICES=$g python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" --max-model-len ${VLLM_MAXLEN:-32768} \
    --gpu-memory-utilization "$frac" --port $p \
    > $LOGDIR/gpu$g.$(date +%m%d-%H%M%S).log 2>&1 </dev/null &
  for _ in $(seq 1 90); do up $p && { echo "  [heal] :$p recovered"; return 0; }; sleep 10; done
  echo "  [heal] :$p recovery failed"; return 1
}

start_proxy() {
  echo "  [heal] restarting proxy :$FRONT"
  local pids
  pids=$(ss -lntpH "sport = :$FRONT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u | tr '\n' ' ')
  [ -n "$pids" ] && kill $pids 2>/dev/null
  sleep 3
  local backends=""
  for g in "${!GPU_PORT[@]}"; do
    up "${GPU_PORT[$g]}" && backends="$backends http://127.0.0.1:${GPU_PORT[$g]}"
  done
  [ -z "$backends" ] && { echo "  [heal] no usable backend, proxy not started"; return 1; }
  setsid nohup /usr/bin/python3 ${DYLO_ROOT}/vllm_proxy.py \
    --port $FRONT --backends $backends > $LOGDIR/proxy.log 2>&1 </dev/null &
  for _ in $(seq 1 30); do up $FRONT && { echo "  [heal] proxy recovered"; return 0; }; sleep 2; done
  echo "  [heal] proxy recovery failed"; return 1
}

case "${1:-check}" in
check)
  bad=0
  for g in "${!GPU_PORT[@]}"; do up "${GPU_PORT[$g]}" || { echo "backend GPU$g:${GPU_PORT[$g]} unhealthy"; bad=1; }; done
  up $FRONT || { echo "proxy :$FRONT unhealthy"; bad=1; }
  [ $bad -eq 0 ] && echo "all healthy"
  exit $bad
  ;;
heal)
  changed=0
  for g in "${!GPU_PORT[@]}"; do
    up "${GPU_PORT[$g]}" || { start_backend "$g" "${GPU_PORT[$g]}"; changed=1; }
  done
  # the proxy must be rebuilt whenever backends changed: its backend list is fixed at startup
  if [ $changed -eq 1 ] || ! up $FRONT; then start_proxy; fi
  # $0 may be a relative path; if the working directory changes during heal it cannot find itself, so use the absolute path
  exec bash "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" check
  ;;
*) echo "usage: $0 {check|heal}"; exit 1;;
esac
