#!/bin/bash
# driver18e: action-channel-only stream arm (forensics on ng v0005: obs-only
# candidate wrecked the probe 0.417->0.167, act-only lifted it to 0.667).
# Same dose/protocol as p3m_ng, train_channels=act. Trainer GPU 7.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18e.sh > results/driver18e.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
rid=p3m_act
log "=== $rid launching (trainer GPU 7)"
CUDA_VISIBLE_DEVICES=7 "$PY" scripts/run_stream.py \
  --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
  --set split=rl --set data_dir=/data/erv1n/resid/data \
  --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
  --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
  --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999 \
  --set train_channels=act --set "extra=$EXTRA" \
  > "results/$rid.log" 2>&1
log "=== $rid done rc=$?"
