#!/bin/bash
# driver18: memory+TTL (ng main, gated queued) vs ICL-frozen vs frozen on the identical 160-task stream
# (tail pool, W=16, 12 probes). Three arms launched concurrently: only the gated
# arm trains (GPU 4) or loads adapters; the other two hit the shared vLLM servers
# with the base model only.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18.sh > results/driver18.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
TASKS=${TASKS:-160}; WINDOW=${WINDOW:-16}

run_arm() { # $1 run_id, rest: extra --set overrides
  local rid=$1; shift
  log "=== $rid launching"
  CUDA_VISIBLE_DEVICES=4 "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks "$TASKS" --window "$WINDOW" --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
    "$@" --set "extra=$EXTRA" \
    > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
  grep "^\[window" "results/$rid.log" | tail -2
}

for p in 8104 8105 8106 8107; do
  curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null || { log "FATAL vllm :$p down (run launch_p3smoke.sh first)"; exit 1; }
done
# main arm = residual pricing + EMA, no G2/G3 (driver9 ng config); gated arm queued after it
( run_arm p3m_ng --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999
  run_arm p3m_gated --set gate_min_probe_delta=-0.09 ) &
run_arm p3m_icl --set serve_experience=true --set gate_min_surprise=999 --set retry_on_fail=false &
run_arm p3m_frozen --set gate_min_surprise=999 --set retry_on_fail=false &
wait
log "=== ALL ARMS DONE"
