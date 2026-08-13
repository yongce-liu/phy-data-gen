#!/usr/bin/env bash
# Run 1000-episode generation for the remaining categories sequentially, each on
# its own branch. Cat01 runs separately. Skips a category whose outputs already
# have 1000 validated episodes (resume-friendly).
set -u

BRANCHES=(
  category_02_two_ball_no_collision
  category_03_ball_hits_object
  category_04_multi_ball
  category_05_granular_medium
  category_06_ball_chain
  category_07_light_heavy
  category_09_ball_container
)

mkdir -p logs

count_validated() {
  local dir="$1"
  [ -d "$dir" ] || { echo 0; return; }
  local n=0
  for f in "$dir"/physics/*/validation.json; do
    [ -f "$f" ] || continue
    if python3 -c "import json,sys; print(json.load(open('$f')).get('passed', False))" 2>/dev/null | grep -q True; then
      n=$((n+1))
    fi
  done
  echo "$n"
}

for b in "${BRANCHES[@]}"; do
  cfg="configs/${b}.yaml"
  out_dir=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$cfg'))['output_root'])" 2>/dev/null)
  done_count=$(count_validated "$out_dir")
  if [ "$done_count" -ge 1000 ]; then
    echo "=== $b already has $done_count validated episodes; skipping ==="
    continue
  fi
  git checkout "$b" 2>/dev/null || { echo "FAILED to checkout $b"; exit 1; }
  git pull --ff-only origin "$b" >/dev/null 2>&1 || echo "warn: pull failed for $b"
  echo "=== $(date +%H:%M:%S) starting $b (have $done_count) ==="
  timeout 108000 .venv/bin/python scripts/generate_high_throughput.py \
    --config "$cfg" --num-episodes 1000 --device cuda:0 > "logs/${b}_full.log" 2>&1
  code=$?
  echo "=== $(date +%H:%M:%S) $b exited code=$code ==="
done
echo "ALL CATEGORIES DONE"
