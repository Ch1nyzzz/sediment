#!/bin/bash
# Matched latest-observation belief residual at the aTTT paper schedule/dose.
# Queue only after confirming no valid four-GPU worker is active.
#
#   cd /data/erv1n/sediment
#   nohup bash scripts/p3_driver12.sh > results/driver12.log 2>&1 &
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

log "=== t0b_latest_k5_paper launching on GPUs 4-7"
RUN_ID=t0b_latest_k5_paper WORKER_SCRIPT=tier0b_latest_worker.py \
  WORKER_ARGS="--k-update 5 --max-updates 5" \
  bash scripts/launch_p1.sh || { log "t0b_latest_k5_paper launch FAILED"; exit 1; }
wait_workers results/t0b_latest_k5_paper
log "=== t0b_latest_k5_paper workers exited"
