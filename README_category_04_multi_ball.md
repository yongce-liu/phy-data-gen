# Category 04 — 多球碰撞 (Multi-ball collisions)

生成 1000 条多球碰撞数据，每条含 4 视角 RGB 视频。

## 覆盖的子变体（`seed % 6` 分区）

- **一维链式碰撞**：一列 3–8 球，一端一球滚入。
- **二维随机碰撞**：5–20 球错位排列，一球高速穿过。
- **三维随机碰撞**：5–15 球从桌面以上高度释放自由下落。
- **同时向中心碰撞**：N 球沿圆周指向圆心。
- **由中心向外散射**：N 球从中心向外射出。
- **连续碰撞**：N 球在桌面上高速运动，与挡边连续反弹。

全部球均被记录（N≤20）。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_04_multi_ball.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat04.log
```

## 参数

| 量 | 范围 |
|---|---|
| 球数 | 3–20 |
| 球半径 | 0.03–0.045 m |
| 速度 | 0–8 m/s |
| 恢复系数 | 0.2–1.0 |
