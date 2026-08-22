#!/bin/bash
# Unattended multi-seed P1 driver: seeds s0..s2 on the tail-200 pool, then an
# extension pool (tasks 201..500 from the tail). vLLM servers persist across
# runs (results/servers). nohup me:
#   cd /data/erv1n/sediment && nohup bash scripts/p1_driver.sh > results/driver.log 2>&1 &
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

for tag in s0 s1 s2; do
  log "=== seed $tag launching"
  RUN_ID=p1_$tag bash scripts/launch_p1.sh || { log "launch FAILED for $tag"; exit 1; }
  wait_workers "results/p1_$tag"
  log "=== seed $tag done"
  bash scripts/p1_status.sh "results/p1_$tag" | tail -12
done

log "=== ext pool (tail 201-500) launching"
RUN_ID=p1_ext WORKER_ARGS="--pool 300 --pool-skip 200" bash scripts/launch_p1.sh \
  || { log "launch FAILED for ext"; exit 1; }
wait_workers results/p1_ext
log "=== ALL DONE"
bash scripts/p1_status.sh results/p1_ext | tail -12
