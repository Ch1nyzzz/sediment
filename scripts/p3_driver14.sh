#!/bin/bash
# Residual pricing plus aTTT 3-gram discount on GPUs 4-7.
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

log "=== latest_resid_ngram_standalone_paper launching on GPUs 4-7"
RUN_ID=latest_resid_ngram_standalone_paper \
  WORKER_SCRIPT=latest_factorial_worker.py \
  WORKER_ARGS="--k-update 5 --max-updates 5 --weighting residual_ngram --train-context standalone" \
  bash scripts/launch_p1.sh || {
    log "latest_resid_ngram_standalone_paper launch FAILED"
    exit 1
  }
wait_workers results/latest_resid_ngram_standalone_paper
log "=== latest_resid_ngram_standalone_paper workers exited"
