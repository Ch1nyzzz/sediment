#!/bin/bash
# Full matched-action signed residual arm on GPUs 4-7.
set -u
SED="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SED"
RUN_ID=signed_action_ul002_paper \
  WORKER_SCRIPT=signed_action_worker.py \
  WORKER_ARGS="--lambda-ul 0.02 --beta-kl 0.01 --post-kl-limit 0.02 --positive-cap 1.55 --negative-cap 4.51 --grad-clip 0.5 --max-backtracks 3 --direction-min-dose 1e-5 --direction-tolerance 1e-5 --steps-per-update 2" \
  bash scripts/launch_p1.sh
