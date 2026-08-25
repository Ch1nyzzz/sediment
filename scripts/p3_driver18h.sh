#!/bin/bash
# driver18h (user directive 2026-08-25): stop every running stream + the 8B
# servers, then launch:
#   p3m_refl_act        4B 160-task stream, reflection + act channel (trainer GPU 7)
#   p3m800_frozen       4B 800-task stream, frozen
#   p3m800_icl_refl     4B 800, frozen, first attempt = retrieval + reflections
#   p3m800_refl_act     4B 800, reflection + act channel CE            (trainer GPU 5)
#   p3m800_refl_signed  4B 800, reflection + act channel, +δ CE / -δ unlikelihood (GPU 6)
# 800 arms: pool = corpus tail 806 (800 stream + 6 probes), G3 probes 6 to bound
# validation cost; no-gate config (pricing + EMA) for the training arms.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18h.sh > results/driver18h.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- stop (bracket trick so this script never matches itself) ---
for pat in "p3_driver18[fg].sh" "run_stream.py.*run-id p3m[8_]"; do
  for pid in $(pgrep -f "$pat"); do kill "$pid" 2>/dev/null && log "killed $pid ($pat)"; done
done
for gpu in 4 6; do
  f=results/servers/vllm8b_$gpu.pid
  [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null && log "killed 8B vllm gpu$gpu" && rm -f "$f"
done
sleep 20
for p in 8104 8105 8106 8107; do
  curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null || { log "FATAL 4B vllm :$p down"; exit 1; }
done

URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
NOGATE="--set gate_min_behavior_change=0 --set gate_min_probe_delta=-999"
FROZEN="--set gate_min_surprise=999 --set retry_on_fail=false"

run_arm() { # $1 gpu $2 run_id $3 tasks $4 probes, rest: --set overrides
  local gpu=$1 rid=$2 tasks=$3 probes=$4; shift 4
  log "=== $rid launching (tasks=$tasks, trainer GPU $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/run_stream.py \
    --engine vllm --trainer torch --tasks "$tasks" --window 16 --run-id "$rid" --seed 0 \
    --set split=rl --set data_dir=/data/erv1n/resid/data \
    --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
    --set gate_probe_tasks="$probes" --set retrieval_k=4 --set max_candidate_samples=2 \
    --set reflect=true "$@" --set "extra=$EXTRA" > "results/$rid.log" 2>&1
  log "=== $rid done rc=$?"
}
run_arm 7 p3m_refl_act 160 12 --set train_channels=act --set gate_error_actions=true $NOGATE &
run_arm 4 p3m800_frozen 800 6 --set reflect=false $FROZEN &
run_arm 4 p3m800_icl_refl 800 6 --set serve_experience=true $FROZEN &
run_arm 5 p3m800_refl_act 800 6 --set train_channels=act --set gate_error_actions=true $NOGATE &
run_arm 6 p3m800_refl_signed 800 6 --set train_channels=act --set signed=true $NOGATE &
wait
log "=== DONE"
