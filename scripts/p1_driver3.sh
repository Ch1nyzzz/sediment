#!/bin/bash
# Post-s0 replan: v0 falsified by s0 paired data (icl net -3.5pp, retention
# 0.683); remaining queue switches entirely to steps-block (v1 protocol):
#   p1_v1 (same tail-200 pool; resumes if driver2 already launched it) ->
#   p1_v1s1, p1_v1s2 (steps-block seeds) -> p1_v1ext (tail 201-500).
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver3.sh > results/driver3.log 2>&1 &
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

for tag in v1 v1s1 v1s2; do
  log "=== $tag (steps-block) launching"
  RUN_ID=p1_$tag WORKER_ARGS="--block-mode steps" bash scripts/launch_p1.sh \
    || { log "launch FAILED $tag"; exit 1; }
  wait_workers "results/p1_$tag"
  finish "p1_$tag"
done

log "=== v1ext (steps-block, tail 201-500) launching"
RUN_ID=p1_v1ext WORKER_ARGS="--block-mode steps --pool 300 --pool-skip 200" \
  bash scripts/launch_p1.sh || { log "launch FAILED v1ext"; exit 1; }
wait_workers results/p1_v1ext
log "=== ALL DONE"
finish p1_v1ext
