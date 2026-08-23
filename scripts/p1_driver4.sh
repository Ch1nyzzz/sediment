#!/bin/bash
# Single-seed plan (user directive 2026-08-23): no seed repeats; action channel in.
#   wait p1_v1 (steps-block, belief-only; the evidence-completeness pairing) ->
#   p1_v2: steps-block + P1.6 status-gated action channel, same tail-200 pool ->
#   p1_v2ext: same protocol, tail 201-500.
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver4.sh > results/driver4.log 2>&1 &
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

log "driver4: waiting for v1 (belief-only) to finish"
wait_workers results/p1_v1
finish p1_v1

log "=== v2 (steps-block + gated action channel) launching"
RUN_ID=p1_v2 WORKER_ARGS="--block-mode steps --act-mode gated" bash scripts/launch_p1.sh \
  || { log "launch FAILED v2"; exit 1; }
wait_workers results/p1_v2
finish p1_v2

log "=== v2ext (tail 201-500) launching"
RUN_ID=p1_v2ext WORKER_ARGS="--block-mode steps --act-mode gated --pool 300 --pool-skip 200" \
  bash scripts/launch_p1.sh || { log "launch FAILED v2ext"; exit 1; }
wait_workers results/p1_v2ext
log "=== ALL DONE"
finish p1_v2ext
