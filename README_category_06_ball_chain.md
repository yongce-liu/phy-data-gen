# Category 06 — 球列碰撞 (Ball-chain collisions)

生成 1000 条球列碰撞数据，每条含 4 视角 RGB 视频。

## 覆盖的子变体（`seed % 7` 分区）

- **单端撞击**：一球从一端撞入球列。
- **双端同时撞击**：两端各一球同时撞击。
- **单球撞球列 / 多球撞球列**：1 或 2–3 个撞击球。
- **等质量球列 / 不等质量球列**：列内质量一致或沿梯度 ×1.3–1.8/步。
- **有间距球列**：球间距 1.02–1.5× 直径。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_06_ball_chain.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat06.log
```

## 参数

| 量 | 范围 |
|---|---|
| 球列长度 | 2–8 |
| 球半径 | 0.035–0.05 m |
| 撞击速度 | 0.5–8 m/s |
| 恢复系数 | 0.3–1.0 |
| 撞击球质量 | 0.05–5 kg（变体 3 覆盖轻重） |
