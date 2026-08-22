#!/bin/bash
# One-line machine-readable pulse for the P1 pipeline (run on the box).
cd /data/erv1n/sediment 2>/dev/null || exit 1
d=DEAD; pgrep -f "scripts/p1_driver" >/dev/null 2>&1 && d=ALIVE
stage=$(cat results/driver*.log 2>/dev/null | grep -E "^[0-9]{4}_" | tail -1)
total=$(cat results/p1_*/p1_shard*.jsonl 2>/dev/null | wc -l | tr -d " ")
errs=$(cat results/p1_*/p1_shard*.jsonl 2>/dev/null | grep -c '"error"')
w=0
for f in results/p1_*/worker_*.pid; do
  [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && w=$((w + 1))
done
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 4,5,6,7 2>/dev/null | tr "\n" "," | sed "s/,$//")
echo "PULSE driver=$d workers=$w rec=$total err=$errs gpu=[$gpu] stage=[$stage]"
