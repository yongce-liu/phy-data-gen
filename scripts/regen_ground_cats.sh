#!/usr/bin/env bash
# Regenerate 10 episodes for categories with the enlarged (200m) ground.
# Uses the venv python for yaml, and force-cleans each output root first so
# stale scenes/videos are never reused.
set -u
BRANCHES=(
  category_02_two_ball_no_collision
  category_03_ball_hits_object
  category_04_multi_ball
  category_06_ball_chain
  category_07_light_heavy
  category_09_ball_container
)
mkdir -p logs/smoke10
for b in "${BRANCHES[@]}"; do
  cfg="configs/${b}.yaml"
  out_dir=$(.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$cfg'))['output_root'])" 2>/dev/null)
  git checkout "$b" 2>/dev/null || { echo "FAILED checkout $b"; exit 1; }
  echo "=== $(date +%H:%M:%S) regenerating $b (root=$out_dir) ==="
  rm -rf "$out_dir"  # force-clean: old scenes were built with the 6m ground
  timeout 2400 .venv/bin/python scripts/generate_high_throughput.py \
    --config "$cfg" --num-episodes 10 --device cuda:0 --rgb_encoder libx264 \
    > "logs/smoke10/${b}_smoke10.log" 2>&1
  echo "=== $(date +%H:%M:%S) $b exited code=$? ==="
done
echo "ALL REGEN DONE"
