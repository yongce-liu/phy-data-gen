#!/usr/bin/env bash
# Run 1000-episode generation for the remaining categories sequentially on the
# current branch. Cat01 runs on its own branch already; this script chains the
# rest. Each category must be checked out on its own branch first.
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
for b in "${BRANCHES[@]}"; do
  git checkout "$b" 2>/dev/null || { echo "FAILED to checkout $b"; exit 1; }
  git pull --ff-only origin "$b" >/dev/null 2>&1
  cfg="configs/${b}.yaml"
  echo "=== $(date +%H:%M:%S) starting $b ==="
  timeout 108000 .venv/bin/python scripts/generate_high_throughput.py \
    --config "$cfg" --num-episodes 1000 --device cuda:0 > "logs/${b}_full.log" 2>&1
  code=$?
  echo "=== $(date +%H:%M:%S) $b exited code=$code ==="
done
echo "ALL CATEGORIES DONE"
