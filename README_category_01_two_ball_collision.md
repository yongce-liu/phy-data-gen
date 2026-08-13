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

每个 episode 会**随机选取不同的 Cosmos billiards 模板**（`template_seed` 留空），
从而获得不同的球台材质与背景球幕环境；两球颜色也逐条随机。相机经修复后
始终正确对准球台中心（见下）。

## 可见接近过程

旧的采样随机给定初速，导致两球在 0.2 s 内就相撞，视频几乎没有"运动—接近"。
现在改为：先采样目标接近时间 `approach_time ∈ [0.9, 1.8] s`，再由初始间距
推导相对速度，保证碰撞前有 **约 1.3–2.0 s** 的可见运动阶段（30 fps 下
40–60 帧）。碰撞后两球继续运动直到视频结束。

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
| 接近时间 approach_time | 0.9–1.8 s（≈1.3–2.0 s 可见接近） |
| 碰撞偏心距 b | 0–0.95×(r1+r2) |
| 恢复系数 e | {0, 0.3, 0.7, 1.0} |
| 质量比 | {1, 2, 5, 10} |
| 角速度 ω | ±50 rad/s（约半数据） |

每个 episode 的 `episode_spec.metadata` 记录了 variant、restitution、质量比、
碰撞偏心距、接近时间、自旋、初始位置/速度，便于复现与下游使用。

## 已修复的问题

- **相机指向错误**：`look_at` 曾用左手系基向量 + `ExtractRotation`，导致所有
  9 个类别的相机都对准球幕环境而非球台。现改为右手系 + 标准 Shepperd
  四元数并取共轭（pxr `xformOp:orient` 施加的是逆旋转）。
- **球掉落出世界**：Cosmos 球台模板没有地板，滚出桌沿的球会掉到 z→-100 m。
  已加入一个 200×200×1 m 的隐形地面碰撞体，球滚落后落在（隐形）地面上。
- **无接近过程**：见上文「可见接近过程」。

## 验证

`validation.json` 现在要求：两球真实接触（`has_contact`）、接触发生在第 6 帧
之后（`approach_ok`，保证有接近阶段）、有限值且最大速度 < 50 m/s。
