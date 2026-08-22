#!/bin/bash
# P1 launcher for the reds-lab box: one vLLM server + one worker per free GPU.
# Usage (on box, from repo root):
#   RUN_ID=p1_s0 bash scripts/launch_p1.sh            # full run, GPUs 4-7
#   GPUS="4" WORKER_ARGS="--limit 2" bash scripts/launch_p1.sh   # smoke
set -u
BASE=/data/erv1n
SED="$(cd "$(dirname "$0")/.." && pwd)"
SERVE_PY=$BASE/resid_venv/bin/python
TRAIN_PY=$BASE/train_venv/bin/python
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
GPUS=(${GPUS:-4 5 6 7})
NW=${#GPUS[@]}
RUN_ID=${RUN_ID:-p1_dev}
OUT=$SED/results/$RUN_ID
SRVDIR=$SED/results/servers  # vLLM servers are shared across runs/seeds
PORT0=${PORT0:-8104}
WORKER_ARGS=${WORKER_ARGS:-}
mkdir -p "$OUT" "$SRVDIR"

export HF_HOME=/data/hf_cache TMPDIR=$BASE/.tmp PIP_CACHE_DIR=$BASE/.pip_cache
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# box has no /usr/local/cuda; avoid flashinfer JIT (needs nvcc) entirely
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}; port=$((PORT0 + i))
  if alive "$SRVDIR/vllm_$gpu.pid"; then
    echo "vllm gpu$gpu already running (pid $(cat "$SRVDIR/vllm_$gpu.pid"))"
  else
    CUDA_VISIBLE_DEVICES=$gpu nohup "$SERVE_PY" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --host 127.0.0.1 --port "$port" \
      --enable-lora --max-lora-rank 32 --max-loras 4 \
      --enable-prefix-caching --max-model-len 12288 \
      --gpu-memory-utilization 0.40 \
      > "$SRVDIR/vllm_$gpu.log" 2>&1 &
    echo $! > "$SRVDIR/vllm_$gpu.pid"
    echo "vllm gpu$gpu -> port $port (pid $!)"
  fi
done

echo "waiting for servers..."
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}; port=$((PORT0 + i)); up=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then up=1; break; fi
    alive "$SRVDIR/vllm_$gpu.pid" || break
    sleep 5
  done
  if [ "$up" != 1 ]; then
    echo "FATAL: vllm gpu$gpu (port $port) not healthy; see $SRVDIR/vllm_$gpu.log" >&2
    exit 1
  fi
  echo "vllm gpu$gpu healthy"
done

for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}; port=$((PORT0 + i))
  if alive "$OUT/worker_$gpu.pid"; then
    echo "worker gpu$gpu already running (pid $(cat "$OUT/worker_$gpu.pid"))"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$gpu nohup "$TRAIN_PY" "$SED/scripts/p1_worker.py" \
    --base-url "http://127.0.0.1:$port/v1" --out "$OUT" \
    --shard "$i" --num-shards "$NW" --tag "$RUN_ID" $WORKER_ARGS \
    > "$OUT/worker_$gpu.log" 2>&1 &
  echo $! > "$OUT/worker_$gpu.pid"
  echo "worker gpu$gpu shard $i/$NW (pid $!)"
done
echo "launched. status: bash scripts/p1_status.sh $OUT"
