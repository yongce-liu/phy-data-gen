# Category 07 — 轻重球碰撞 (Light vs heavy ball collisions)

生成 1000 条轻重球碰撞数据，每条含 4 视角 RGB 视频。质量比 5–50。

## 覆盖的子变体（`seed % 5` 分区）

- **轻球撞重球**：轻球反弹、重球几乎不动。
- **重球撞轻球**：轻球高速飞出。
- **轻球高速反弹**：弹性碰撞下轻球反向高速弹回。
- **轻球被加速**：运动的重球将能量传给轻球。
- **质量梯度碰撞**：3–6 球链条，每步质量 ×2–5。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_07_light_heavy.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat07.log
```

## 参数

| 量 | 范围 |
|---|---|
| 质量比 | 5–50 |
| 轻球质量 | 0.05 kg |
| 速度 | 0.5–8 m/s |
| 恢复系数 | {0, 0.5, 1.0} |
