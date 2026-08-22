#!/bin/bash
# Revised overnight plan (replaces p1_driver.sh mid-flight):
#   wait s0 (v0 outcome-block, already running) ->
#   p1_v1: SAME tail-200 pool, steps-block (evidence-completeness pilot) ->
#   s1, s2 (v0 protocol seeds) -> ext300 (v0, tail 201-500).
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver2.sh > results/driver2.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }

wait_workers() { # $1 = run dir
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

log "driver2: waiting for s0 (v0) to finish"
wait_workers results/p1_s0
finish p1_s0

log "=== v1 steps-block pilot launching (same pool)"
RUN_ID=p1_v1 WORKER_ARGS="--block-mode steps" bash scripts/launch_p1.sh \
  || { log "v1 launch FAILED"; exit 1; }
wait_workers results/p1_v1
finish p1_v1

for tag in s1 s2; do
  log "=== seed $tag (v0) launching"
  RUN_ID=p1_$tag bash scripts/launch_p1.sh || { log "launch FAILED $tag"; exit 1; }
  wait_workers "results/p1_$tag"
  finish "p1_$tag"
done

log "=== ext pool (tail 201-500, v0) launching"
RUN_ID=p1_ext WORKER_ARGS="--pool 300 --pool-skip 200" bash scripts/launch_p1.sh \
  || { log "launch FAILED ext"; exit 1; }
wait_workers results/p1_ext
log "=== ALL DONE"
finish p1_ext
