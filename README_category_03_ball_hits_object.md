# Category 03 — 球撞击物体 (Ball hits an object)

生成 1000 条「球撞击物体」物理数据，每条含 4 视角 RGB 视频。

## 覆盖的子变体（`seed % 8` 分区）

- **软/硬物体**：被撞物体恢复系数 0–0.4 vs 0.5–0.8。
- **轻/重物体**：物体质量 0.05–0.2 kg vs 2–10 kg。
- **可移动物体**：动态刚体；**固定障碍物**：静态（无刚体）。
- **被障碍物拦截 / 撞击后反弹**：高恢复系数、固定重型物体。

## 物体

从 registry（2025 个 MolmoSpaces 资产）中抽取 Bowl/Plate/Cup/Vase/Pan/Pot
子集，质量按变体覆盖，比例缩放至 ~0.1 m。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_03_ball_hits_object.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat03.log
```

## 参数

| 量 | 范围 |
|---|---|
| 球半径 | 0.04–0.06 m |
| 球速 | 1–10 m/s |
| 物体质量 | 0.05–10 kg（或静态） |
| 物体恢复系数 | 0–0.9 |
