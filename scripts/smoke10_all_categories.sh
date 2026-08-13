#!/usr/bin/env bash
# Regenerate 10 smoke episodes for each category branch after the ground fix.
# Cleans each category's output root first so no stale fall-through data remains.
set -u

BRANCHES=(
  category_01_two_ball_collision
  category_02_two_ball_no_collision
  category_03_ball_hits_object
  category_04_multi_ball
  category_05_granular_medium
  category_06_ball_chain
  category_07_light_heavy
  category_08_soft_ball_deform
  category_09_ball_container
)
mkdir -p logs/smoke10

for b in "${BRANCHES[@]}"; do
  cfg="configs/${b}.yaml"
  out_dir=$(.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output_root'])" 2>/dev/null)
  git checkout "$b" 2>/dev/null || { echo "FAILED to checkout $b"; exit 1; }
  echo "=== $(date +%H:%M:%S) cleaning + starting $b (root=$out_dir) ==="
  [ -n "$out_dir" ] && rm -rf "$out_dir"
  timeout 2400 .venv/bin/python scripts/generate_high_throughput.py \
    --config "$cfg" --num-episodes 10 --device cuda:0 --rgb_encoder libx264 \
    > "logs/smoke10/${b}_smoke10.log" 2>&1
  code=$?
  # count validated
  n=0
  for f in "$out_dir"/physics/*/validation.json; do
    [ -f "$f" ] || continue
    if .venv/bin/python -c "import json,sys; print(json.load(open('$f')).get('passed', False))" 2>/dev/null | grep -q True; then n=$((n+1)); fi
  done
  echo "=== $(date +%H:%M:%S) $b exited code=$code passed=$n/10 ==="
done
echo "ALL CATEGORIES SMOKE10 DONE"
