# Category 01 — 两球碰撞 (Two-ball collision)

生成 1000 条「两球碰撞」物理数据，每条含 4 视角 RGB 视频（+深度+物理注解）。

## 覆盖的子变体

- **碰撞几何**：正面碰撞（impact parameter b=0）、斜向碰撞（0<b<r1+r2）、
  偏心碰撞（b 接近 r1+r2）、相向碰撞（两球对向而行）、追尾碰撞（一球追一球）。
- **弹性**：弹性碰撞（e=1）、非弹性碰撞（e∈{0.3,0.7}）、完全非弹性碰撞（e=0）。
- **质量**：等质量、不等质量（质量比 2/5/10）。
- **初始运动**：一球静止、两球均运动。
- **自旋**：约半数 episode 给母球初始角速度（±50 rad/s）。

子变体按 `seed % 6` 分区到 1000 个 episode，其余参数由 `random.Random(seed)`
确定性采样。

## 场景

以 Cosmos billiards 模板为背景（桌面、挡边、原有相机保留），通过
`object_mode: procedural` 停用模板内的球与母球刚体，插入两枚自研球（Sphere
primitive + RigidBodyAPI + CCD + `physics:velocity`/`physics:angularVelocity`）。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_01_two_ball_collision.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat01.log
```

- 1280×720 × 4 相机（Side/Front/Top/Action），30 fps，5 s。
- 输出：`outputs/category_01/<run_id>/videos/*.mp4`、
  `outputs/category_01/<run_id>/physics/object_states.jsonl` 等。
- 冒烟：`--num-episodes 3` 先验证再全量。

## 参数

| 量 | 范围 |
|---|---|
| 半径 r1, r2 | 0.03–0.05 m |
| 初速 v | 0.5–8 m/s |
| 碰撞偏心距 b | 0–0.95×(r1+r2) |
| 恢复系数 e | {0, 0.3, 0.7, 1.0} |
| 质量比 | {1, 2, 5, 10} |
| 角速度 ω | ±50 rad/s（约半数据） |
