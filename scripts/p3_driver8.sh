#!/bin/bash
# Formal P3 pass 1: long stream (160 tasks, W=16, 10 windows, 12-task G3 probe
# set), three arms on the identical stream: gated -> nogate -> noadapt.
# Smoke verdict motivating this: at 48 tasks EMA alone absorbed a -1.0-probe
# poison merge (gate redundant, 4-probe G3 too noisy); the gate's value claim
# needs horizon + probe precision.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver8.sh > results/driver8.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"

run_arm() { # $1 run_id, rest: extra --set overrides
  local rid=$1; shift
  log "=== $rid launching"
  CUDA_VISIBLE_DEVICES=4 "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks=12 --set retrieval_k=4 \
    "$@" --set "extra=$EXTRA" \
    > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
  tail -3 "results/$rid.log" | head -2
}

run_arm p3long
run_arm p3long_ng --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999
run_arm p3long_na --set gate_min_surprise=999 --set retry_on_fail=false
log "=== ALL ARMS DONE"
