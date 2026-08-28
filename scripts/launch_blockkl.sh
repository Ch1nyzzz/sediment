#!/bin/bash
# p512 mean16 online memory distillation with NO compiler.
#
#   retrieve one successful, task-similar complete trajectory from the frozen
#   source bank -> that block IS the hint (no compiler)
#   bare rollout      = the student's own on-policy sample AND metric (b),
#                       prompt-only performance: the stream curve IS
#                       internalisation, no separate eval needed
#   with-memory teacher = scored on the same bare trajectory (no second actor
#                         rollout); the difference is the distillation signal
#
# Run first, before the compiler arms: it separates "does internalising retrieved
# memory work at all" from "does compiling it into a lesson help", and it is the
# only arm whose teacher context needs no extra trained component.
#   objective         = REVERSE KL(student || teacher) at the student's states,
#                       teacher = the same weights reading prompt+hint
#                       (self-distillation, so the target never goes stale)
#
# Why reverse and why on-policy: at inference the model runs prompt-only, so the
# expectation must be under the student. Forward KL on teacher-sampled
# trajectories measures 0.008 nats (kl_screen 08-26) -- at its own states the
# teacher's sequence is already what the student would say.
#
# Every action token is a KL position (weight_mode=sft), not the ~20 tool-call
# interior tokens the gated arms credited: those left 4000 positions
# unconstrained and burned 0.3 nats of anti-repetition prior per merge
# (scripts/policy_shift.py), which is what every collapse so far has in common.
# anchor_kl_coef holds the untaught positions to the base; grounded_weight damps
# argument values copyable from context, the channel that erosion rides on.
#

# Fixed family-held-out protocol over all 2,550 tasks:
#   offline meta split: env_141..174 (reserved; NO meta training in this run)
#   retrieval bank:     read-only historical frozen trajectories
#   online train:       env_175..182 (400 tasks = exactly 25 windows)
#   held-out test:      env_183..191 (450 tasks; never retrieved or trained on)
set -eu
cd "$(dirname "$0")/.."
RUN_ID=${RUN_ID:-online400_blockkl_rank1_mean16_rkl}
OUT_ROOT=${OUT_ROOT:-results}
mkdir -p "$OUT_ROOT"
if pgrep -f "run_stream[.]py.*run-id $RUN_ID" >/dev/null; then
  echo "$RUN_ID is already running" >&2
  exit 2
fi
if [ -n "${RESUME:-}" ]; then
  RESUME_SET="--set resume=true"
elif [ -e "$OUT_ROOT/$RUN_ID" ] || [ -e "$OUT_ROOT/$RUN_ID.log" ]; then
  echo "refusing to overwrite existing run: $RUN_ID" >&2
  exit 3
else
  RESUME_SET=""
fi
PY=/data/erv1n/train_venv/bin/python
export HF_HOME=/data/hf_cache TMPDIR=/data/erv1n/.tmp TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KEEP=${KEEP:-'["env_175","env_176","env_177","env_178","env_179","env_180","env_181","env_182"]'}
URLS=${URLS:-'["http://127.0.0.1:8104/v1","http://127.0.0.1:8105/v1","http://127.0.0.1:8106/v1","http://127.0.0.1:8107/v1"]'}
DONORS=${DONORS:-'["/data/erv1n/sediment/results/meta_frozen_s0/buffer.jsonl","/data/erv1n/sediment/results/meta_frozen_s1/buffer.jsonl","/data/erv1n/sediment/results/meta_frozen_s2/buffer.jsonl","/data/erv1n/sediment/results/meta_frozen_s3/buffer.jsonl"]'}
EXTRA="{\"lopd_dir\":\"/data/erv1n/resid/third_party/LOPD\",\"base_urls\":$URLS,\"keep_families\":$KEEP}"
echo "$(date +%m%d_%H:%M) === $RUN_ID launching (trainer GPU ${TRAIN_GPU:-5})"
CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-5} nohup "$PY" scripts/run_stream.py \
  --engine vllm --trainer torch --tasks ${ONLINE_TASKS:-400} --window 16 \
  --out "$OUT_ROOT" --run-id "$RUN_ID" --seed 0 \
  --set split=rl --set data_dir=/data/erv1n/resid/data \
  --set lr=${LR:-0.00015} --set lora_r=8 --set lora_alpha=8 \
  --set weight_mode=sft --set grounded_weight=0.2 \
  --set propose_rule=all \
  --set kl_target=true --set kl_reverse=true --set kl_topk=20 \
  --set anchor_kl_coef=${ANCHOR:-0.5} \
  --set retrieval_buffer_paths="$DONORS" \
  --set retrieval_k=1 --set retrieval_offset=0 \
  --set retrieval_scope=cross_family --set retrieval_diversity=family \
  --set retrieval_score=task_text_success --set experience_view=full \
  --set hindsight_include_own_outcome=false \
  --set reflect=false --set train_channels=act \
  --set retry_on_fail=false --set max_retries=0 \
  --set gate_validate=false --set gate_probe_tasks=0 --set gate_replay_states=0 \
  --set max_candidate_samples=16 --set steps_per_merge=1 --set epochs=1 \
  --set generation_seed_mode=bare_prompt_hash \
  $RESUME_SET --set "extra=$EXTRA" >> "$OUT_ROOT/$RUN_ID.log" 2>&1 &
sleep 3
pgrep -af "run_stream[.]py.*run-id $RUN_ID"
