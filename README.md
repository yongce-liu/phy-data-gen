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

多 GPU 服务器可以用总入口按连续 seed 区间启动独立 worker。例如在 8 张
GPU 上总共生成 8000 个 episode：

```bash
uv run python scripts/generate_multi_gpu.py \
  --config configs/billiards.yaml \
  --num-episodes 8000 \
  --gpus 0 1 2 3 4 5 6 7
```

默认每张 GPU 启动一个 worker。每张卡启动两个 worker 时使用：

```bash
uv run python scripts/generate_multi_gpu.py \
  --config configs/billiards.yaml \
  --num-episodes 8000 \
  --gpus 0 1 2 3 4 5 6 7 \
  --workers-per-gpu 2
```

脚本为每个 worker 显式设置 Isaac Lab 的 `cuda:<GPU>` 设备，并清除子进程的
`CUDA_VISIBLE_DEVICES`，避免 CUDA 与 Vulkan 的 GPU 枚举不一致。worker 默认
通过 `--viz none` 以 headless 模式启动，并自动生成临时配置和日志目录。所有
worker 共享基础配置中的 `output_root`，但 seed 不重叠，因此默认生成的 run ID
不会冲突。任一 worker 失败或收到 Ctrl+C 时，脚本会停止同一批次的其他
worker。正式运行前可以添加 `--dry-run` 检查分片。

查看完整参数：

```bash
uv run python -m phy_data_gen.cli --help
uv run python -m phy_data_gen.cli generate --help
uv run python scripts/generate_multi_gpu.py --help
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

## 离线渲染

使用 `--no-frames` 生成的物理轨迹可以在不重新运行 PhysX 的情况下回放渲染：

```bash
uv run python scripts/render_from_states.py \
  --config configs/billiards_high_throughput.yaml \
  --device cuda:0 \
  --num-episodes 5000 \
  --batch-size 4
```

脚本默认跳过所有选定相机均已有非空 RGB 和深度文件的 episode，因此重复执行
即可断点续渲染。使用 `--overwrite` 强制重渲染；使用 `--seed-start 1000` 从指定
seed 开始；使用 `--cameras TopDown Corner` 仅渲染部分相机。`--frame-stride 2`
每两帧渲染一帧，并将输出帧率同步降为原来的一半。

`--batch-size` 控制一个 Isaac Sim stage 中并行回放的环境数量。每批环境挂载在
独立的 `/World/envs/env_N` 下；同一个逻辑帧只执行一次 Kit update，再读取整批
环境的所有相机。建议从 `--batch-size 2`、`4`、`8` 逐级测试 GPU 显存和吞吐量。

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

`scene.object_mode` 控制模板中的动态物体处理方式：

- `generated_objects`：禁用模板刚体，并插入 registry 中抽取的物体。
- `template_dynamics`：保留模板原始刚体及其初始速度，不插入额外物体；
  仿真会自动发现并记录 `scene.world_prim` 下的刚体。
- `replace_assets`：从模板槽位复制初始位姿、线速度和角速度，停用原始刚体与
  几何，并原位引用 registry 中的本地随机资产。后续碰撞、惯量和运动由替换
  资产的刚体与碰撞体交给 PhysX 重新仿真；资产没有显式质量时由 PhysX 根据
  碰撞体计算。`dynamic_prims` 为空时会替换 `scene.world_prim` 下的全部动态
  物体；不同 seed 会得到不同资产组合。

默认的 objects-falling 与 Billiards 配置都使用 `replace_assets`。生成过程只会
引用 registry 已登记且存在于本机的 USD 文件；文件缺失时直接报错，不会联网
下载资产。

`template_seed` 单独控制场景模板选择，`seed` 控制资产组合。批量生成时只递增
`seed`，因此同一次 `--num-episodes` 运行会保持同一个 scene，并产生不同的
本地资产排列。

Billiards 模板中的球和挡边引用 Cosmos 生成环境下的 `/isaac-sim/...`
绝对资源路径。球体会被本地随机资产替换；未替换的挡边仍会用等尺寸的 USD
Cube primitive 补齐，不修改原始模板文件。

## 性能配置

- `render.rgb_encoder: h264_nvenc`：最高吞吐量，要求 FFmpeg 支持 NVENC。
- `render.rgb_encoder: libx264`：无 NVIDIA 编码器时使用 CPU 编码。
- 增加 `scene.cameras` 会增加渲染成本，但无需重复物理仿真。
- 使用 `--num-episodes` 批量生成可摊薄 Isaac Sim 启动成本。
- 生产环境建议使用 `--viz none`，并将 CPU governor 设置为 `performance`。
