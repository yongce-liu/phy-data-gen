# Category 09 — 球进入/撞击容器 (Ball into / against a container)

生成 1000 条「球与容器交互」数据，每条含 4 视角 RGB 视频。

## 覆盖的子变体（`seed % 6` 分区）

- **垂直落入篮筐**：从开口上方垂直落入。
- **滚入箱体**：沿地面滚入（低沿）。
- **撞击容器侧壁**：直接撞侧壁反弹。
- **落入后停留**：低速落入并在底部静止。
- **落入后反弹**：高速落入，在容器内多次弹跳。
- **未进入容器**：瞄准偏移，球落在容器外。

## 容器

程序化静态开顶箱（底板 + 4 薄壁，0.4×0.4×0.25 m），放在 billiards 桌面。
容器是 `record=False` + `dynamic=False` 的静态物体，只记录球。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_09_ball_container.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat09.log
```

## 参数

| 量 | 范围 |
|---|---|
| 球半径 | 0.03–0.05 m |
| 速度 | 1–8 m/s |
| 落高 | 0.3 m（相对容器顶） |
| 容器恢复系数 | 0.2–0.6（壁面） |
| 球恢复系数 | 0.2–0.9 |
