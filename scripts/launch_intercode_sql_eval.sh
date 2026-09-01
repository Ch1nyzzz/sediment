#!/bin/bash
# Matched held-out evaluation for InterCode-SQL base or a completed adapter.
# All arms use the same frozen test534 manifest, temperature 0.7 and a strict
# 10-turn cap.
set -eu
cd "$(dirname "$0")/.."

RUN_ID=${RUN_ID:?}
BASE_URL=${BASE_URL:?}
ARM=${ARM:?base|opd_jsd|fork_ce|trajectory_ce|trajectory_dpo}
OUT_ROOT=${OUT_ROOT:-/data/erv1n/intercode-three-20260901/eval}
DATA_ROOT=${DATA_ROOT:-/data/erv1n/intercode-sql-base-20260901/data}
TASK_MANIFEST=${TASK_MANIFEST:-/data/erv1n/intercode-three-20260901/test534.jsonl}
TASKS=${TASKS:-534}
PY=${PY:-/data/erv1n/train_venv/bin/python}
ADAPTER_PATH=${ADAPTER_PATH:-}

mkdir -p "$OUT_ROOT"
if pgrep -f "run_stream[.]py.*--run-id $RUN_ID" >/dev/null; then
  echo "$RUN_ID is already running" >&2
  exit 2
fi
if [ -e "$OUT_ROOT/$RUN_ID" ] || [ -e "$OUT_ROOT/$RUN_ID.log" ]; then
  echo "refusing to overwrite existing run: $RUN_ID" >&2
  exit 3
fi

INITIAL_ARGS=()
if [ "$ARM" != base ]; then
  if [ -z "$ADAPTER_PATH" ] || [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo "ARM=$ARM requires ADAPTER_PATH containing adapter_config.json" >&2
    exit 4
  fi
  INITIAL_ARGS=(--set "initial_adapter_path=$ADAPTER_PATH")
fi

export HF_HOME=${HF_HOME:-/data/hf_cache}
export TMPDIR=${TMPDIR:-/data/erv1n/.tmp}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export PYTHONPATH="$DATA_ROOT/../python${PYTHONPATH:+:$PYTHONPATH}"

EXTRA="{\"benchmark\":\"intercode_sql\",\"task_manifest\":\"$TASK_MANIFEST\",\"base_urls\":[\"$BASE_URL\"],\"intercode_mysql\":{\"host\":\"127.0.0.1\",\"port\":33307,\"user\":\"admin\",\"password\":\"admin\",\"sql_mode\":\"IGNORE_SPACE\"}}"
echo "$(date -Is) evaluating $RUN_ID arm=$ARM endpoint=$BASE_URL max_steps=10"
nohup "$PY" scripts/run_stream.py \
  --engine vllm --trainer stub --tasks "$TASKS" --window 16 \
  --out "$OUT_ROOT" --run-id "$RUN_ID" --seed 0 \
  --set split=benchmark --set temperature=0.7 \
  --set max_steps=10 --set episode_token_budget=8192 \
  --set max_tokens=512 --set max_model_len=32768 \
  --set retrieval_k=0 --set serve_experience=false --set reflect=false \
  --set score_hindsight=false --set retry_on_fail=false --set max_retries=0 \
  --set gate_validate=false --set gate_probe_tasks=0 \
  --set async_train=false "${INITIAL_ARGS[@]}" --set "extra=$EXTRA" \
  >> "$OUT_ROOT/$RUN_ID.log" 2>&1 &
sleep 3
pgrep -af "run_stream[.]py.*--run-id $RUN_ID"
