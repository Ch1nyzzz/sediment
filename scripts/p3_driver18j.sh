#!/bin/bash
# driver18j: dose-∝-evidence variants of the two 800-task training arms
# (w_norm_floor=20), run alongside the no-floor originals. Trainers GPU 4 / 7.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18j.sh > results/driver18j.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
run_arm() { local gpu=$1 rid=$2; shift 2
  log "=== $rid launching (trainer GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 800 --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 --set w_norm_floor=3 \
    --set gate_probe_tasks=6 --set retrieval_k=4 --set max_candidate_samples=2 \
    --set reflect=true --set train_channels=act \
    --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999 \
    "$@" --set "extra=$EXTRA" > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"; }
run_arm 4 p3m800_refl_act_f --set gate_error_actions=true &
run_arm 7 p3m800_refl_signed_f --set signed=true &
wait; log "=== DONE"
