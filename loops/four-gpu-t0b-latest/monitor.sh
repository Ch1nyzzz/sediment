#!/bin/bash
# Read-only monitor for the overnight t0b_latest_k5_paper run.
set -u

REMOTE=${REMOTE:-reds-lab}
INTERVAL=${INTERVAL:-45}
MAX_POLLS=${MAX_POLLS:-120}
RUN=${RUN:-/data/erv1n/sediment/results/t0b_latest_k5_paper}
DRIVER_LOG=${DRIVER_LOG:-/data/erv1n/sediment/results/driver12.log}

for ((poll = 1; poll <= MAX_POLLS; poll++)); do
  ssh -T -o BatchMode=yes -o ConnectTimeout=30 "$REMOTE" "
    cd /data/erv1n/sediment || exit 1
    printf 'POLL %s ' '$poll/$MAX_POLLS'
    date -Is
    printf 'GPU47 '
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader | sed -n '5,8p' | tr '\n' ';'
    printf '\nROWS '
    total=0
    for f in '$RUN'/t0_shard*.jsonl; do
      [ -f \"\$f\" ] || continue
      n=\$(wc -l < \"\$f\")
      total=\$((total + n))
      printf '%s=%s ' \"\$(basename \"\$f\")\" \"\$n\"
    done
    printf 'total=%s\n' \"\$total\"
    printf 'WORKERS '
    alive=0
    for f in '$RUN'/worker_*.pid; do
      [ -f \"\$f\" ] || continue
      p=\$(cat \"\$f\")
      if kill -0 \"\$p\" 2>/dev/null; then
        alive=\$((alive + 1))
        printf '%s:%s:alive ' \"\$(basename \"\$f\")\" \"\$p\"
      else
        printf '%s:%s:dead ' \"\$(basename \"\$f\")\" \"\$p\"
      fi
    done
    printf 'alive=%s\n' \"\$alive\"
    printf 'ERROR_HITS '
    grep -Eih 'traceback|exception|out of memory|fatal|upd_errors=[1-9]' \
      '$RUN'/worker_*.log 2>/dev/null | wc -l
    printf 'DRIVER_DONE '
    grep -c 'workers exited' '$DRIVER_LOG' \
      2>/dev/null || true
  " || true

  sleep "$INTERVAL"
done
