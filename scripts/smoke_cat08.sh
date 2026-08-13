#!/usr/bin/env bash
# Smoke-test the deformable runner (category 08) with a few episodes and verify
# the soft ball actually deforms before committing to the full 1000-episode run.
set -u
set -e

BRANCH="category_08_soft_ball_deform"
cd /mnt/workspace/physical_data/phy_data_gen
git checkout "$BRANCH" 2>/dev/null || { echo "checkout failed"; exit 1; }
git pull --ff-only origin "$BRANCH" >/dev/null 2>&1

echo "=== smoke: 4 episodes ==="
timeout 3600 .venv/bin/python scripts/generate_high_throughput.py \
  --config configs/category_08_soft_ball_deform.yaml \
  --num-episodes 4 --device cuda:0 > logs/cat08_smoke.log 2>&1
echo "smoke exit=$?"

# Verify deformation was recorded.
NODES=$(ls outputs/category_08/physics/category_08_*_*/deformable_nodes.jsonl 2>/dev/null | wc -l)
VALID=$(for f in outputs/category_08/physics/category_08_*_*/validation.json; do [ -f "$f" ] && python3 -c "import json;print(json.load(open('$f')).get('passed'))" 2>/dev/null; done | grep -c True)
echo "deformable_nodes files: $NODES, passed validations: $VALID"
echo "=== cat08 smoke done ==="
