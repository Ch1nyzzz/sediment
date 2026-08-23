#!/bin/bash
# After v2ext: v3 = v2 protocol (steps-block + gated actions) + NEAR-TRANSFER
# probes — train on task A, additionally evaluate base/xtask-icl/ours/uniform
# on the next task of the same env family. Same tail-200 pool, single seed.
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver5.sh > results/driver5.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }

wait_workers() {
  sleep 60
  while :; do
    local alive=0 f
    for f in "$1"/worker_*.pid; do
      [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && alive=1
    done
    [ "$alive" = 0 ] && return
    sleep 120
  done
}

finish() { log "=== $1 done"; bash scripts/p1_status.sh "results/$1" | tail -10; }

log "driver5: waiting for v2ext to finish"
wait_workers results/p1_v2ext
finish p1_v2ext

log "=== v3 (gated actions + near-transfer probes) launching"
RUN_ID=p1_v3 WORKER_ARGS="--block-mode steps --act-mode gated --probe next" \
  bash scripts/launch_p1.sh || { log "launch FAILED v3"; exit 1; }
wait_workers results/p1_v3
log "=== ALL DONE"
finish p1_v3
