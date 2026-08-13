# Category 08 — 软球碰撞与形变 (Soft-ball collision & deformation)

生成 1000 条软体球碰撞/形变数据，每条含 2 视角 RGB 视频（Side + Top，
960×540）。**这是唯一使用 Deformable（软体 FEM）物理路径的类别**。

## 覆盖的子变体（`seed % 6` 分区）

- **弹性压缩**：软球撞刚性壁/桌面被压缩后恢复。
- **压缩后回弹**：软球从高处落下弹起。
- **黏弹性形变**：低杨氏模量 + 高阻尼，恢复缓慢。
- **塑性形变（近似）**：极低杨氏模量，episode 内恢复可忽略。
  PhysX 软体为弹性 FEM，无原生塑性，此变体为黏弹性近似（README 说明）。
- **多球软体接触**：两个软球对撞。

## 实现

- 采样器把软球参数（杨氏模量 1e3–1e6 Pa、泊松比 0.3–0.49、密度 50–1000）
  写入 `EpisodeSpec.metadata`。
- `phy_data_gen/runners/deformable.py` 的 `build_scene_hook`（纯 pxr）把软球
  Sphere 换成 UV-sphere Mesh（烹饪源）。
- `run_simulation` 用 `create_auto_volume_deformable_hierarchy` + 
  `add_deformable_material` 建立软体层级，步进后采集节点位姿：
  - `deformable_states.jsonl`：每帧球心/平均半径/最大最小半径（形变量）。
  - `deformable_nodes.jsonl`：每帧节点坐标。
  - RGB/深度沿用 `FrameRecorder`。

## 生成

```bash
uv run python scripts/generate_high_throughput.py \
  --config configs/category_08_soft_ball_deform.yaml \
  --num-episodes 1000 --device cuda:0 2>&1 | tee logs/cat08.log
```

先冒烟 20 条验证软球确实形变（`max_contraction > 2%`）再全量。

## 参数

| 量 | 范围 |
|---|---|
| 软球半径 | 0.05–0.08 m |
| 杨氏模量 | 1e3–1e6 Pa |
| 泊松比 | 0.3–0.49 |
| 密度 | 50–1000 kg/m³ |
| 碰撞速度 | 1–6 m/s |
| 物理 dt | 1/120（120 Hz） |
| 时长 | 3 s |
