#!/bin/bash
# driver18i: drop p3m_refl_act (user), relaunch p3m800_refl_signed with the
# status-free dead-band signed credit (pos_thr/neg_thr 0.5). Other 800 arms untouched.
#   cd /data/erv1n/sediment && nohup bash scripts/p3_driver18i.sh > results/driver18i.log 2>&1 &
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
log() { echo "$(date +%m%d_%H:%M) $*"; }
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for pat in "run-id p3m_refl_ac[t] " "run-id p3m800_refl_signe[d] "; do
  for pid in $(pgrep -f "run_stream.py.*$pat"); do kill "$pid" && log "killed $pid ($pat)"; done
done
sleep 10; rm -rf results/p3m800_refl_signed
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
rid=p3m800_refl_signed
log "=== $rid launching (trainer GPU 6)"
CUDA_VISIBLE_DEVICES=6 "$PY" scripts/run_stream.py \
  --engine vllm --trainer torch --tasks 800 --window 16 --run-id "$rid" --seed 0 \
  --set split=rl --set data_dir=/data/erv1n/resid/data \
  --set lr=0.00015 --set lora_alpha=32 --set epochs=2 \
  --set gate_probe_tasks=6 --set retrieval_k=4 --set max_candidate_samples=2 \
  --set reflect=true --set train_channels=act --set signed=true \
  --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999 \
  --set "extra=$EXTRA" > "results/$rid.log" 2>&1
log "=== $rid done rc=$?"
