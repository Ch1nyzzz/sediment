#!/bin/bash
# driver18d: two more arms in parallel on the same 160-task stream, both at the
# aTTT paper dose (LoRA r=8 alpha=16 lr=5e-4, 2 grad steps per update = top-2
# samples x 1 epoch), pricing + EMA, no G2/G3:
#   p3m_ng_pd       paper dose only                     (trainer GPU 6)
#   p3m_ng_refl_pd  + actor reflection stored/rendered  (trainer GPU 7)
#                   + P1.6 error-action gate (own reflection quotes own actions)
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18d.sh > results/driver18d.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"

run_arm() { # $1 gpu $2 run_id, rest: extra --set overrides
  local gpu=$1 rid=$2; shift 2
  log "=== $rid launching (trainer GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lora_r=8 --set lora_alpha=16 --set lr=0.0005 --set epochs=1 \
    --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
    --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999 \
    "$@" --set "extra=$EXTRA" \
    > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
}
run_arm 6 p3m_ng_pd &
run_arm 7 p3m_ng_refl_pd --set reflect=true --set gate_error_actions=true &
wait
log "=== DONE"
