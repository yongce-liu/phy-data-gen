# Category 02 — 两球无碰撞运动 (Two-ball no-collision motion)

生成 1000 条「两球运动中不发生接触」的物理数据，每条含 4 视角 RGB 视频。

## 覆盖的子变体（`seed % 5` 分区）

- **平行通过**：两球并排同向，横向错位 ≥1.05×(r1+r2)。
- **交叉通过**：轨迹成角交叉，交叉点处法向距离足够大。
- **同向不同速**：同向不同速但横向错位，快球追不上。
- **相向错位通过**：对向而行但错位错过。
- **不同时间通过**：同向同车道，后球延迟释放（通过初始位置后移实现）。

错位量 `miss ≥ 1.05×(r1+r2)`，保证全程球心距 > 两球半径和。

## 场景

billiards 桌面背景 + 4 相机，`object_mode: procedural`。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_02_two_ball_no_collision.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat02.log
```

冒烟：`--num-episodes 3` 先验证再全量。

## 参数

| 量 | 范围 |
|---|---|
| 半径 | 0.03–0.05 m |
| 初速 | 0.3–5 m/s |
| 横向错位 | 1.05–1.6×(r1+r2) |
| 释放延迟 | 0.4–0.9 s（变体 5） |
