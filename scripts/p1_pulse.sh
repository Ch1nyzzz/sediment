#!/bin/bash
# One-line machine-readable pulse for the experiment pipelines (run on the box).
cd /data/erv1n/sediment 2>/dev/null || exit 1
stage=$(cat results/driver*.log 2>/dev/null | grep -E "^[0-9]{4}_" | tail -1)
errs=$(cat results/p1_*/p1_shard*.jsonl 2>/dev/null | grep -c '"error"')
gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 4,5,6,7 2>/dev/null | tr "\n" "," | sed "s/,$//")
streams=$(pgrep -fc "run_stream.py" 2>/dev/null || echo 0)
p3s=""
for f in results/p3long*.log results/p3smoke.log results/p3noadapt.log results/p3nogate.log; do
  [ -f "$f" ] || continue
  n=$(grep -c "^\[window" "$f" 2>/dev/null | head -1)
  b=$(basename "$f" .log)
  p3s="$p3s ${b#p3}=${n:-0}"
done
echo "PULSE streams=$streams err=$errs gpu=[$gpu] wins[$p3s ] stage=[$stage]"
