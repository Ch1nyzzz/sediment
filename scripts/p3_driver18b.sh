#!/bin/bash
# driver18b: replace the misconfigured p3m_ours (default zero-tolerance G3 + dose 64)
# with the driver9 main config (ng: pricing + EMA, dose 2), then the gated arm (-0.09).
# icl/frozen arms from driver18 keep running. Run via file (pkill self-match hazard).
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18b.sh > results/driver18b.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"

for pid in $(pgrep -f "run_stream.py.*run-id [p]3m_ours"); do kill "$pid" && log "killed p3m_ours pid $pid"; done

run_arm() {
  local rid=$1; shift
  log "=== $rid launching"
  CUDA_VISIBLE_DEVICES=4 "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks 160 --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks=12 --set retrieval_k=4 --set max_candidate_samples=2 \
    "$@" --set "extra=$EXTRA" \
    > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
  grep "^\[window" "results/$rid.log" | tail -2
}
run_arm p3m_ng --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999
run_arm p3m_gated --set gate_min_probe_delta=-0.09
log "=== DONE"
