#!/bin/bash
# P3 streaming smoke on real GPUs: 48-task EnvScaler stream, W=8, gated
# accumulation with mid-dose writes (P1.7 knee: lr 1.5e-4, alpha 32).
# Reuses the persistent vLLM servers (starts any that died). Trainer shares GPU 4.
#   cd /data/erv1n/sediment && bash scripts/launch_p3smoke.sh
set -u
BASE=/data/erv1n
SED="$(cd "$(dirname "$0")/.." && pwd)"
SERVE_PY=$BASE/resid_venv/bin/python
TRAIN_PY=$BASE/train_venv/bin/python
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
GPUS=(4 5 6 7)
PORT0=8104
SRVDIR=$SED/results/servers
RUN_ID=${RUN_ID:-p3smoke}
mkdir -p "$SRVDIR" "$SED/results"

export HF_HOME=/data/hf_cache TMPDIR=$BASE/.tmp PIP_CACHE_DIR=$BASE/.pip_cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}; port=$((PORT0 + i))
  if ! alive "$SRVDIR/vllm_$gpu.pid"; then
    CUDA_VISIBLE_DEVICES=$gpu nohup "$SERVE_PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --host 127.0.0.1 --port "$port" \
      --enable-lora --max-lora-rank 32 --max-loras 4 \
      --enable-prefix-caching --max-model-len 12288 \
      --gpu-memory-utilization 0.40 \
      > "$SRVDIR/vllm_$gpu.log" 2>&1 &
    echo $! > "$SRVDIR/vllm_$gpu.pid"
    echo "vllm gpu$gpu starting (pid $!)"
  fi
done
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}; port=$((PORT0 + i)); up=0
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { up=1; break; }
    alive "$SRVDIR/vllm_$gpu.pid" || break
    sleep 5
  done
  [ "$up" = 1 ] || { echo "FATAL: vllm gpu$gpu unhealthy" >&2; exit 1; }
done
echo "servers healthy"

URLS=$(printf ",\"http://127.0.0.1:%d/v1\"" $(seq $PORT0 $((PORT0 + ${#GPUS[@]} - 1))))
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":[${URLS:1}]}"
CUDA_VISIBLE_DEVICES=${GPUS[0]} nohup "$TRAIN_PY" "$SED/scripts/run_stream.py" \
  --engine vllm --trainer torch --tasks "${TASKS:-48}" --window "${WINDOW:-8}" \
  --run-id "$RUN_ID" --seed 0 \
  --set split=rl --set data_dir=/data/erv1n/resid/data \
  --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
  --set "extra=$EXTRA" \
  > "$SED/results/$RUN_ID.log" 2>&1 &
echo $! > "$SED/results/$RUN_ID.pid"
echo "stream launched (pid $(cat "$SED/results/$RUN_ID.pid")) -> results/$RUN_ID.log"
