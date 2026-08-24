#!/bin/bash
# 40-task scoring-only matched action-hint audit on GPUs 4-7.
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
RUN_ID=action_hint_control_debug40 \
  WORKER_SCRIPT=action_hint_control_worker.py \
  WORKER_ARGS="--limit 10 --max-scored-steps 25" \
  bash scripts/launch_p1.sh
