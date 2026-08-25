#!/bin/bash
# driver18g: reflection arms at OUR dose on the 4B servers.
#   p3m_icl_refl  frozen params, first attempt = retrieved block + reflections (now)
#   p3m_refl_act  act-only channel + reflection + P1.6 error-action gate; trainer
#                 GPU 7, launched once p3m_act releases it.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18g.sh > results/driver18g.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"

run_arm() { # $1 gpu $2 run_id, rest: --set overrides
  local gpu=$1 rid=$2; shift 2
  log "=== $rid launching"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
    --set reflect=true "$@" --set "extra=$EXTRA" > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
}
run_arm 7 p3m_icl_refl --set serve_experience=true --set gate_min_surprise=999 --set retry_on_fail=false &
(
  while pgrep -f "run_stream.py.*run-id p3m_ac[t] " >/dev/null; do sleep 120; done
  log "p3m_act finished; GPU 7 trainer free"
  run_arm 7 p3m_refl_act --set train_channels=act --set gate_error_actions=true \
    --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999
) &
wait
log "=== DONE"
