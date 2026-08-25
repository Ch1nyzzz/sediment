#!/bin/bash
# driver18f: Qwen3-8B on the identical 160-task stream — frozen / icl / act-only.
# Two 8B vLLM servers (no-think template) on GPUs 4 and 6 (ports 8114/8116) next
# to the resident 4B servers; act trainer on GPU 5. 4B arms keep running.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18f.sh > results/driver18f.log 2>&1 &
set -u
BASE=/data/erv1n
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
MODEL=Qwen/Qwen3-8B
SERVE_PY=$BASE/resid_venv/bin/python
PY=$BASE/train_venv/bin/python
SRVDIR=$SED/results/servers
GPUS=(4 6); PORTS=(8114 8116)
export HF_HOME=/data/hf_cache TMPDIR=$BASE/.tmp TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
CHAT_TMPL=$SED/sediment/templates/qwen3_nothink.jinja

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}; port=${PORTS[$i]}
  if ! curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then
    CUDA_VISIBLE_DEVICES=$gpu nohup "$SERVE_PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --host 127.0.0.1 --port "$port" --chat-template "$CHAT_TMPL" \
      --enable-lora --max-lora-rank 32 --max-loras 4 --enable-prefix-caching \
      --max-model-len 12288 --gpu-memory-utilization 0.40 > "$SRVDIR/vllm8b_$gpu.log" 2>&1 &
    echo $! > "$SRVDIR/vllm8b_$gpu.pid"; echo "$MODEL|$CHAT_TMPL" > "$SRVDIR/vllm8b_$gpu.serving"
    log "8B vllm gpu$gpu :$port starting"
  fi
done
for i in "${!GPUS[@]}"; do
  port=${PORTS[$i]}; up=0
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && { up=1; break; }; sleep 5; done
  [ "$up" = 1 ] || { log "FATAL: 8B vllm :$port unhealthy"; exit 1; }
done
log "8B servers healthy"
URLS='["http://127.0.0.1:8114/v1","http://127.0.0.1:8116/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"

run_arm() { # $1 gpu $2 run_id, rest: --set overrides
  local gpu=$1 rid=$2; shift 2
  log "=== $rid launching"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
    --set model=$MODEL --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
    "$@" --set "extra=$EXTRA" > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
}
run_arm 5 p3m8_frozen --set gate_min_surprise=999 --set retry_on_fail=false &
run_arm 5 p3m8_icl --set serve_experience=true --set gate_min_surprise=999 --set retry_on_fail=false &
run_arm 5 p3m8_act --set train_channels=act --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999 &
wait
log "=== DONE"
