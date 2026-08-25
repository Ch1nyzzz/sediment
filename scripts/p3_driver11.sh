#!/bin/bash
# User-directed queue (2026-08-24): dose-matched head-to-head at aTTT's write
# strength (LoRA r=8 alpha=16 lr=5e-4, 2 grad steps/update).
#   1) tier0c_paper: OUR decision-point pricing objective at their dose  <- ours first
#   2) attt_paper:   their objective at their dose (resume, 20/200 done)
# Same 200-task pool, same frozen base, same harness; std/retry join from tier0.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver11.sh > results/driver11.log 2>&1 &
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

log "=== tier0c_paper (ours @ paper dose r8/a16/lr5e-4/2steps) launching on 4 GPUs"
RUN_ID=tier0c_paper WORKER_SCRIPT=tier0c_worker.py WORKER_ARGS="--paper-recipe" \
  bash scripts/launch_p1.sh || { log "tier0c_paper launch FAILED"; exit 1; }
wait_workers results/tier0c_paper
log "=== tier0c_paper done"

log "=== attt_paper resuming (theirs @ paper dose)"
RUN_ID=attt_paper WORKER_SCRIPT=attt_worker.py WORKER_ARGS="--paper-recipe --candidate env" \
  bash scripts/launch_p1.sh || { log "attt_paper launch FAILED"; exit 1; }
wait_workers results/attt_paper
log "=== ALL DONE"
