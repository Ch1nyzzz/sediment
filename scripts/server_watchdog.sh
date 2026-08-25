#!/bin/bash
# Keep the four resident 4B vLLM servers alive. Every 60 s: a dead port is
# restarted (restart_server.sh) and every running stream's current adapter is
# re-registered on it (served name = run_id-vNNNN; legacy unprefixed runs are
# not running any more). Run once, detached:
#   cd /data/erv1n/sediment && nohup bash scripts/server_watchdog.sh > results/watchdog.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
GPUS=(4 5 6 7); PORTS=(8104 8105 8106 8107)
while true; do
  for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}; port=${PORTS[$i]}
    curl -sf -m 10 "http://127.0.0.1:$port/v1/models" >/dev/null && continue
    log "port $port down -> restarting"
    args=()
    for rid in $(pgrep -af "run_stream.py" | grep -o "run-id [a-z0-9_]*" | awk '{print $2}'); do
      v=$(ls "results/$rid/registry" 2>/dev/null | grep "^v0" | sort | tail -1)
      [ -n "$v" ] && args+=("$rid-$v=results/$rid/registry/$v")
    done
    bash scripts/restart_server.sh "$gpu" "$port" "${args[@]}" 2>&1 | sed "s/^/  /"
  done
  sleep 60
done
