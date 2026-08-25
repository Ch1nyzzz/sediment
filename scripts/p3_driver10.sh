#!/bin/bash
# Queue tier0c (decision-point hint pricing, K=5 fixed-cadence updates) after
# driver9 finishes. Start on box:
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver10.sh > results/driver10.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }

log "=== waiting for driver9 ALL DONE"
while ! grep -q "=== ALL DONE" results/driver9.log 2>/dev/null; do sleep 120; done

log "=== tier0c (decision-point hint, K=5) launching on 4 GPUs"
RUN_ID=tier0c WORKER_SCRIPT=tier0c_worker.py bash scripts/launch_p1.sh \
  || { log "tier0c launch FAILED"; exit 1; }

sleep 60
while :; do
  alive=0
  for f in results/tier0c/worker_*.pid; do
    [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && alive=1
  done
  [ "$alive" = 0 ] && break
  sleep 120
done
log "=== tier0c done"
