#!/bin/bash
# After v3: v4 = identical protocol but aTTT-strength writes.
# Rationale: v3 half-sample shows zero cross-instance transfer from internalized
# writes while cross-task ICL is positive; our effective update magnitude is
# ~40x below aTTT (lr 5e-5 vs 5e-4; alpha/r 0.5 vs 2). One knob, one run.
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver6.sh > results/driver6.log 2>&1 &
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

log "driver6: waiting for v3 to finish"
wait_workers results/p1_v3
finish p1_v3

log "=== v4 (aTTT-strength writes: lr 5e-4, alpha 64) launching"
RUN_ID=p1_v4 WORKER_ARGS="--block-mode steps --act-mode gated --probe next --lr 5e-4 --lora-alpha 64" \
  bash scripts/launch_p1.sh || { log "launch FAILED v4"; exit 1; }
wait_workers results/p1_v4
log "=== ALL DONE"
finish p1_v4
