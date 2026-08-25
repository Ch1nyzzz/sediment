#!/bin/bash
# relaunch p3m800_refl_act_f (floor arm) from scratch on the 5xx-retrying client
cd "$(dirname "$0")/.."
for pid in $(pgrep -f "run_stream.py.*run-id p3m800_refl_act_[f] "); do kill "$pid"; done
sleep 5; rm -rf results/p3m800_refl_act_f
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
URLS='["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS}"
echo "$(date +%m%d_%H:%M) === p3m800_refl_act_f relaunch (trainer GPU 4)"
CUDA_VISIBLE_DEVICES=4 nohup "$PY" scripts/run_stream.py \
  --engine vllm --trainer torch --tasks 800 --window 16 --run-id p3m800_refl_act_f --seed 0 \
  --set split=rl --set data_dir=/data/erv1n/resid/data \
  --set lr=0.00015 --set lora_alpha=32 --set epochs=2 --set w_norm_floor=3 \
  --set gate_probe_tasks=6 --set retrieval_k=4 --set max_candidate_samples=2 \
  --set reflect=true --set train_channels=act --set gate_error_actions=true \
  --set gate_min_behavior_change=0 --set gate_min_probe_delta=-999 \
  --set "extra=$EXTRA" > results/p3m800_refl_act_f.log 2>&1 &
sleep 3; pgrep -af run_stream | grep -v pgrep | sed "s/.*run-id \([a-z0-9_]*\).*/\1/"
