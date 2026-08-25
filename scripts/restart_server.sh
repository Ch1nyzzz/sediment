#!/bin/bash
# Restart one dead vLLM server and re-register served LoRAs for streams still
# running against it.  usage: restart_server.sh <gpu> <port> [lora_name=path ...]
#   bash scripts/restart_server.sh 5 8105 v0007=results/p3m_ng/registry/v0007
set -u
BASE=/data/erv1n
SED="$(cd "$(dirname "$0")/.." && pwd)"
gpu=$1; port=$2; shift 2
SRVDIR=$SED/results/servers
export HF_HOME=/data/hf_cache TMPDIR=$BASE/.tmp VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then echo "port $port already up"; else
  mv "$SRVDIR/vllm_$gpu.log" "$SRVDIR/vllm_$gpu.crash.$(date +%m%d_%H%M).log" 2>/dev/null
  CUDA_VISIBLE_DEVICES=$gpu nohup "$BASE/resid_venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL:-Qwen/Qwen3-4B-Instruct-2507}" --host 127.0.0.1 --port "$port" \
    --enable-lora --max-lora-rank 32 --max-loras 4 --enable-prefix-caching \
    --max-model-len 12288 --gpu-memory-utilization 0.40 > "$SRVDIR/vllm_$gpu.log" 2>&1 &
  echo $! > "$SRVDIR/vllm_$gpu.pid"
  for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null || { echo "FATAL: port $port not up"; exit 1; }
  echo "port $port up"
fi
for kv in "$@"; do
  name=${kv%%=*}; path=$SED/${kv#*=}
  curl -s -X POST "http://127.0.0.1:$port/v1/load_lora_adapter" -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$name\",\"lora_path\":\"$path\"}"; echo " <- $name"
done
