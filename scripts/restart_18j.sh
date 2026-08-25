#!/bin/bash
# relaunch the two floor arms on the retry-enabled client (kills driver18j + its two streams)
cd "$(dirname "$0")/.."
for pid in $(pgrep -f "bash scripts/p3_driver18[j].sh") $(pgrep -f "run_stream.py.*run-id p3m800_refl_act_[f] ") $(pgrep -f "run_stream.py.*run-id p3m800_refl_signed_[f] "); do
  kill "$pid" && echo "killed $pid"
done
sleep 8
rm -rf results/p3m800_refl_act_f results/p3m800_refl_signed_f
nohup bash scripts/p3_driver18j.sh > results/driver18j.log 2>&1 &
sleep 20; cat results/driver18j.log
