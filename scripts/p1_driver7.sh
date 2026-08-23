#!/bin/bash
# Dose-response midpoint: v3 (5e-5/a16) inert, v4 (5e-4/a64) catastrophic ->
# v5 = lr 1.5e-4, alpha 32 (~6x), same protocol (steps+gated+probes).
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver7.sh > results/driver7.log 2>&1 &
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
log "driver7: waiting for v4 to finish"
wait_workers results/p1_v4
finish p1_v4
log "=== v5 (midpoint writes: lr 1.5e-4, alpha 32) launching"
RUN_ID=p1_v5 WORKER_ARGS="--block-mode steps --act-mode gated --probe next --lr 1.5e-4 --lora-alpha 32" \
  bash scripts/launch_p1.sh || { log "launch FAILED v5"; exit 1; }
wait_workers results/p1_v5
log "=== ALL DONE"
finish p1_v5
