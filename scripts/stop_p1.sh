#!/bin/bash
# Stop all workers + vLLM servers of one P1 run. Usage: bash scripts/stop_p1.sh results/p1_s0
OUT=${1:?usage: stop_p1.sh <run_dir>}
for f in "$OUT"/worker_*.pid "$OUT"/vllm_*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null && echo "stopped $(basename "$f") ($pid)"
  fi
done
