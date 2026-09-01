#!/bin/bash
# InterCode-SQL transfer of the three registered online learning algorithms.
# Protocol invariants: Qwen3-4B, train500 only, max 10 SQL turns, 512 generated
# tokens/turn, 8192-token episode cap, one isolated vLLM endpoint per arm.
set -eu
cd "$(dirname "$0")/.."

RUN_ID=${RUN_ID:?}
ARM=${ARM:?opd_jsd|fork_ce|trajectory_ce|trajectory_dpo}
TRAIN_GPU=${TRAIN_GPU:?}
BASE_URL=${BASE_URL:?}
OUT_ROOT=${OUT_ROOT:-/data/erv1n/intercode-three-20260901/train}
DATA_ROOT=${DATA_ROOT:-/data/erv1n/intercode-sql-base-20260901/data}
TASK_MANIFEST=${TASK_MANIFEST:-$DATA_ROOT/train500.jsonl}
TASKS=${TASKS:-500}
PY=${PY:-/data/erv1n/train_venv/bin/python}
REPLAY_MIN=${REPLAY_MIN:-16}
REPLAY_BATCH=${REPLAY_BATCH:-16}
MIN_MERGE=${MIN_MERGE:-4}
OPD_MIN_MERGE=${OPD_MIN_MERGE:-0}
PACE=${PACE:-}
TRAJECTORY_REPLAY_MAX_USES=${TRAJECTORY_REPLAY_MAX_USES:-5}
FORK_REPLAY_MAX_USES=${FORK_REPLAY_MAX_USES:-5}
DPO_REPLAY_MAX_USES=${DPO_REPLAY_MAX_USES:-5}
DPO_BETA=${DPO_BETA:-2.0}
mkdir -p "$OUT_ROOT"

if pgrep -f "run_stream[.]py.*--run-id $RUN_ID" >/dev/null; then
  echo "$RUN_ID is already running" >&2
  exit 2
fi
if [ -e "$OUT_ROOT/$RUN_ID" ] || [ -e "$OUT_ROOT/$RUN_ID.log" ]; then
  echo "refusing to overwrite existing run: $RUN_ID" >&2
  exit 3
fi

export HF_HOME=${HF_HOME:-/data/hf_cache}
export TMPDIR=${TMPDIR:-/data/erv1n/.tmp}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export PYTHONPATH="$DATA_ROOT/../python${PYTHONPATH:+:$PYTHONPATH}"

COMMON_ARGS=(
  --set split=benchmark
  --set temperature=0.7
  --set max_steps=10
  --set episode_token_budget=8192
  --set max_tokens=512
  --set max_model_len=32768
  --set lr=0.00001
  --set lora_r=8
  --set lora_alpha=8
  --set weight_mode=sft
  --set train_channels=act
  --set retrieval_k=0
  --set serve_experience=false
  --set reflect=false
  --set score_hindsight=false
  --set gate_validate=false
  --set gate_probe_tasks=0
  --set gate_replay_states=0
  --set async_train=true
  --set persist_opt_state=true
  --set replay_cap=512
  --set replay_min="$REPLAY_MIN"
  --set replay_batch="$REPLAY_BATCH"
  --set replay_success_boost=1.0
  --set anchor_kl_coef=0.0
  --set merge_alpha=1.0
  --set max_candidate_samples=32
  --set steps_per_merge=1
  --set epochs=1
)

case "$ARM" in
  opd_jsd)
    METHOD_ARGS=(
      --set propose_rule=all
      --set stepwise_feedback_distill=true
      --set stepwise_feedback_failures_only=true
      --set kl_target=true
      --set kl_reverse=false
      --set kl_states=first
      --set kl_teacher=local
      --set teacher_contexts=feedback
      --set kl_jsd_alpha=0.5
      --set kl_student_topk=100
      --set kl_teacher_ema=0.01
      --set turn_decay=1.0
      --set retry_on_fail=false
      --set max_retries=0
      --set replay_max_age=25
      --set replay_is_clip=2.0
      --set async_min_steps_per_window="${PACE:-4}"
      --set pack_samples=true
      --set pack_max_len=8192
      --set min_merge_samples="$OPD_MIN_MERGE"
    )
    ;;
  fork_ce)
    METHOD_ARGS=(
      --set propose_rule=advantage
      --set stepwise_experience=false
      --set stepwise_extract=false
      --set redo_block_samples=0
      --set redo_stepwise_samples=0
      --set redo_feedback_samples=4
      --set kl_target=false
      --set kl_states=fork
      --set own_view=outcome
      --set retry_on_fail=true
      --set max_retries=1
      --set fork_locator=sql
      --set fork_objective=ce_suffix
      --set fork_require_grounded=false
      --set replay_max_age=25
      --set replay_max_uses="$FORK_REPLAY_MAX_USES"
      --set replay_is_clip=0.0
      --set async_min_steps_per_window="${PACE:-1}"
      --set pack_samples=false
      --set min_merge_samples="$MIN_MERGE"
    )
    ;;
  trajectory_dpo)
    METHOD_ARGS=(
      --set propose_rule=advantage
      --set stepwise_experience=false
      --set stepwise_extract=false
      --set redo_block_samples=0
      --set redo_stepwise_samples=0
      --set redo_feedback_samples=4
      --set kl_target=false
      --set kl_states=retry
      --set own_view=outcome
      --set retry_on_fail=true
      --set max_retries=1
      --set fork_objective=trajectory_dpo
      --set dpo_beta="$DPO_BETA"
      --set dpo_reference=parent
      --set replay_max_age=25
      --set replay_max_uses="$DPO_REPLAY_MAX_USES"
      --set replay_is_clip=0.0
      --set async_min_steps_per_window="${PACE:-1}"
      --set pack_samples=false
      --set min_merge_samples="$MIN_MERGE"
    )
    ;;
  trajectory_ce)
    METHOD_ARGS=(
      --set propose_rule=advantage
      --set stepwise_experience=true
      --set stepwise_k=3
      --set stepwise_candidates=24
      --set stepwise_max_chars=900
      --set stepwise_extract=true
      --set redo_block_samples=0
      --set redo_stepwise_samples=4
      --set kl_target=false
      --set kl_states=retry
      --set own_view=outcome
      --set retry_on_fail=true
      --set max_retries=1
      --set replay_max_age=25
      --set replay_max_uses="$TRAJECTORY_REPLAY_MAX_USES"
      --set replay_is_clip=0.0
      --set async_min_steps_per_window="${PACE:-1}"
      --set pack_samples=false
      --set min_merge_samples="$MIN_MERGE"
    )
    ;;
  *)
    echo "unknown ARM=$ARM" >&2
    exit 4
    ;;
esac

EXTRA="{\"benchmark\":\"intercode_sql\",\"task_manifest\":\"$TASK_MANIFEST\",\"base_urls\":[\"$BASE_URL\"],\"intercode_mysql\":{\"host\":\"127.0.0.1\",\"port\":33307,\"user\":\"admin\",\"password\":\"admin\",\"sql_mode\":\"IGNORE_SPACE\"}}"
echo "$(date -Is) launching $RUN_ID arm=$ARM gpu=$TRAIN_GPU endpoint=$BASE_URL max_steps=10"
CUDA_VISIBLE_DEVICES=$TRAIN_GPU nohup "$PY" scripts/run_stream.py \
  --engine vllm --trainer torch --tasks "$TASKS" --window 16 \
  --out "$OUT_ROOT" --run-id "$RUN_ID" --seed 0 \
  "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}" \
  --set "extra=$EXTRA" >> "$OUT_ROOT/$RUN_ID.log" 2>&1 &
sleep 3
pgrep -af "run_stream[.]py.*--run-id $RUN_ID"
