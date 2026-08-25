#!/bin/bash
# driver18c: run the gated arm NOW in parallel with p3m_ng (trainer on GPU 5;
# served LoRA names are run-id prefixed so both runs share the vLLM servers).
# Kills driver18b's shell first so it does not launch a second p3m_gated later.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18c.sh > results/driver18c.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
for pid in $(pgrep -f "bash scripts/p3_driver18[b].sh"); do kill "$pid" && log "killed driver18b shell $pid (ng child keeps running)"; done
rid=p3m_gated
log "=== $rid launching (GPU 5 trainer)"
CUDA_VISIBLE_DEVICES=5 "$PY" scripts/run_stream.py \
  --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
  --set split=rl --set data_dir=/data/erv1n/resid/data \
  --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
  --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
  --set gate_min_probe_delta=-0.09 --set "extra=$EXTRA" \
  > "results/$rid.log" 2>&1
log "=== $rid done rc=$?"
