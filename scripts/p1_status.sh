#!/bin/bash
# Progress + live aggregate for a P1 run dir. Usage: bash scripts/p1_status.sh results/p1_s0
OUT=${1:?usage: p1_status.sh <run_dir>}
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "${GPUS_CSV:-4,5,6,7}" 2>/dev/null
for f in "$OUT"/worker_*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f"): $(grep -c '^\[' "$f" 2>/dev/null) lines, last:"
  tail -1 "$f"
done
python3 - "$OUT" <<'PY'
import glob, json, statistics, sys

rows = []
for f in glob.glob(sys.argv[1] + "/p1_shard*.jsonl"):
    for line in open(f):
        if line.strip():
            rows.append(json.loads(line))
ok = [r for r in rows if "error" not in r]
err = [r for r in rows if "error" in r]
print(f"\ntasks: {len(rows)} done ({len(ok)} ok, {len(err)} err)")
if err:
    print("  last err:", err[-1]["task_id"], err[-1]["error"][:120])
for k in ("first", "retry", "icl", "ours", "uniform"):
    xs = [r[k]["success"] for r in ok if k in r]
    if xs:
        print(f"  {k:8s} {sum(xs)/len(xs):.3f}  (n={len(xs)})")
ss = [r["obs_surprise"] for r in ok if "obs_surprise" in r]
if ss:
    print(f"  obs_surprise mean {statistics.mean(ss):.4f}  "
          f"sup_tokens mean {statistics.mean([r['n_supervised'] for r in ok if 'n_supervised' in r]):.0f}")
dts = [sum(r.get("timings", {}).values()) for r in ok if r.get("timings")]
if dts:
    print(f"  sec/task mean {statistics.mean(dts):.0f}")
PY
