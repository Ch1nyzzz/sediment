#!/bin/bash
# Scoring-only residual token audit on GPUs 4-7; performs no training.
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
RUN_ID=residual_token_debug20 \
  WORKER_SCRIPT=residual_debug_worker.py \
  WORKER_ARGS="--limit 5 --k-update 5 --max-selections 5" \
  bash scripts/launch_p1.sh
