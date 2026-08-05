# phy-data-gen

基于 Isaac Sim / PhysX 的物理交互数据生成工具。输出结构参考 NVIDIA
`PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes` 数据集。

## 生成数据

命令行由 Tyro 管理：

```bash
uv run python -m phy_data_gen.cli generate \
  --config configs/run.yaml \
  --episode-id objects_falling_c8917ca2_42 \
  --viz none
```

如果省略 `--episode-id`，程序会按 `{scene_name}_{scene_hash}_{seed}` 自动生成。

批量生产时复用同一个 Isaac Sim 进程，seed 从配置值开始递增：

```bash
uv run python -m phy_data_gen.cli generate \
  --config configs/run.yaml \
  --num-episodes 100 \
  --viz none
```

查看完整参数：

```bash
uv run python -m phy_data_gen.cli --help
uv run python -m phy_data_gen.cli generate --help
```

## 输出格式

```text
outputs/
├── cameras/<run_id>/<camera>.json
├── depths/<run_id>/<camera>.mkv
├── physics/<run_id>/
│   ├── <camera>_velocity.npz
│   ├── <camera>_spin.npz
│   ├── <camera>_com.npz
│   ├── <camera>_rot.npz
│   ├── <camera>_static.json
│   ├── object_states.jsonl
│   └── validation.json
├── scene/<run_id>/
│   ├── <scene_name>.usda
│   └── episode_spec.json
└── videos/<run_id>/<camera>.mp4
```

- RGB：H.264 MP4，默认使用 `h264_nvenc` 实时编码。
- 深度：FFV1 `gray16le` MKV，无损保存量化后的米制深度。
- `configs/run.yaml` 中 `depth_scale_meters: 0.001` 表示一个整数单位为 1 mm；
  深度值 0 表示无效像素。
- RGB 和深度直接从同一次渲染读取并流式写入 FFmpeg，不生成中间 PNG。
- 配置多个相机时，所有相机共享同一次物理仿真。

读取深度时，将 MKV 解码为 `uint16`，再乘相机 JSON 中的
`depth_encoding.scale_meters`：

```bash
ffmpeg -v error -i outputs/depths/<run_id>/Side.mkv \
  -f rawvideo -pix_fmt gray16le depth.raw
```

## 其他命令

```bash
uv run python -m phy_data_gen.cli plan --config configs/run.yaml

uv run python -m phy_data_gen.cli inspect-template \
  --config configs/run.yaml \
  --viz none

uv run python -m phy_data_gen.cli build-registry \
  --asset-root data/assets/molmospaces \
  --output data/assets/registry.json
```

Registry 只记录物体资产。生成 episode 时会按 `seed` 直接从
`configs/run.yaml` 的 `template_root` 下确定性抽取场景模板。

## 性能配置

- `render.rgb_encoder: h264_nvenc`：最高吞吐量，要求 FFmpeg 支持 NVENC。
- `render.rgb_encoder: libx264`：无 NVIDIA 编码器时使用 CPU 编码。
- 增加 `scene.cameras` 会增加渲染成本，但无需重复物理仿真。
- 使用 `--num-episodes` 批量生成可摊薄 Isaac Sim 启动成本。
- 生产环境建议使用 `--viz none`，并将 CPU governor 设置为 `performance`。
