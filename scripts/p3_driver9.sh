#!/bin/bash
# User-directed queue (2026-08-23 night):
#   1) tier0: within-episode residual updates, single-task protocol (fast, 4 GPUs)
#   2) p3long_ng: no-gate stream, dose-fixed (no G2/G3 overhead -> fast)
#   3) p3long_na: frozen baseline for the stream comparison
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver9.sh > results/driver9.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"

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

run_arm() {
  local rid=$1; shift
  log "=== $rid launching"
  CUDA_VISIBLE_DEVICES=4 "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
    "$@" --set "extra=$EXTRA" \
    > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
}

log "=== tier0 (within-episode) launching on 4 GPUs"
RUN_ID=tier0 WORKER_SCRIPT=tier0_worker.py bash scripts/launch_p1.sh \
  || { log "tier0 launch FAILED"; exit 1; }
wait_workers results/tier0
log "=== tier0 done"

run_arm p3long_ng --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999
run_arm p3long_na --set gate_min_surprise=999 --set retry_on_fail=false
log "=== ALL DONE"
