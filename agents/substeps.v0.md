下面按**最小闭环优先**实现。第一版不要先做复杂 agent、provider、campaign 抽象；先完成“打开模板 → 替换一个资产 → 仿真 → 输出视频和状态”，再逐步扩展。整体遵守当前 `AGENTS.md` 中“先读、最小修改、真实验证”的规范。

---

# 一、先确认输入文件

你的 Cosmos 模板路径是：

```bash
data/templates/cosmos3/objects_falling/objects_falling_c8917ca2_871/objects_falling_c8917ca2_871.usda
```

先确认 MolmoSpaces 资产确实存在。`tree` 没有显示出具体文件，可能是软链接或目录尚为空：

```bash
find -L data/assets/molmospaces \
  -type f \
  \( -name "*.usd" -o -name "*.usda" -o -name "*.usdc" \) \
  | head -20
```

统计数量：

```bash
find -L data/assets/molmospaces \
  -type f \
  \( -name "*.usd" -o -name "*.usda" -o -name "*.usdc" \) \
  | wc -l
```

确认模板：

```bash
test -f \
  data/templates/cosmos3/objects_falling/objects_falling_c8917ca2_871/objects_falling_c8917ca2_871.usda \
  && echo "Template OK"
```

---

# 二、建立第一版代码结构

先保持简单：

```text
phy_data_gen/
├── __init__.py
├── app.py
├── config.py
├── schemas.py
├── inspect_template.py
├── registry.py
├── episode.py
├── scene.py
├── simulation.py
├── recording.py
├── validation.py
└── cli.py
```

创建文件：

```bash
touch phy_data_gen/__init__.py

touch \
  phy_data_gen/app.py \
  phy_data_gen/config.py \
  phy_data_gen/schemas.py \
  phy_data_gen/inspect_template.py \
  phy_data_gen/registry.py \
  phy_data_gen/episode.py \
  phy_data_gen/scene.py \
  phy_data_gen/simulation.py \
  phy_data_gen/recording.py \
  phy_data_gen/validation.py \
  phy_data_gen/cli.py

mkdir -p configs outputs
```

第一阶段不要再拆出几十个目录。等单场景闭环稳定后，再根据真实重复逻辑拆分。

---

# 三、Step 1：实现配置读取

## 文件

```text
configs/run.yaml
phy_data_gen/config.py
```

## `configs/run.yaml`

```yaml
template_path: data/templates/cosmos3/objects_falling/objects_falling_c8917ca2_871/objects_falling_c8917ca2_871.usda
asset_root: data/assets/molmospaces
registry_path: data/assets/registry.json
output_root: outputs

backend: physx
seed: 42
num_objects: 3

simulation:
  physics_dt: 0.0020833333333333333
  render_fps: 30
  duration_seconds: 5.0

render:
  width: 640
  height: 360
```

这里先使用 640×360，等流程稳定后再切换 1920×1080。

## `config.py`

实现：

```python
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    physics_dt: float
    render_fps: int
    duration_seconds: float


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int


@dataclass(frozen=True)
class RunConfig:
    template_path: Path
    asset_root: Path
    registry_path: Path
    output_root: Path
    backend: str
    seed: int
    num_objects: int
    simulation: SimulationConfig
    render: RenderConfig


def load_config(path: Path) -> RunConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    return RunConfig(
        template_path=Path(raw["template_path"]),
        asset_root=Path(raw["asset_root"]),
        registry_path=Path(raw["registry_path"]),
        output_root=Path(raw["output_root"]),
        backend=str(raw["backend"]),
        seed=int(raw["seed"]),
        num_objects=int(raw["num_objects"]),
        simulation=SimulationConfig(**raw["simulation"]),
        render=RenderConfig(**raw["render"]),
    )
```

## 验证

```bash
uv run --no-sync python -c \
  "from pathlib import Path; from phy_data_gen.config import load_config; print(load_config(Path('configs/run.yaml')))"
```

---

# 四、Step 2：实现 Isaac Lab 启动 smoke test

## 文件

```text
phy_data_gen/app.py
```

Isaac Lab standalone 脚本需要先通过 `AppLauncher` 启动应用，再导入依赖 Isaac Sim runtime 的模块；`SimulationContext.reset()` 必须在首次 step 前调用。([Isaac Sim][1])

先写最简单版本：

```python
import argparse

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.sim import SimulationCfg, SimulationContext

    sim = SimulationContext(
        SimulationCfg(
            dt=1.0 / 60.0,
            device="cuda:0",
        )
    )

    sim.reset()

    for _ in range(10):
        sim.step(render=False)

    simulation_app.close()


if __name__ == "__main__":
    main()
```

## 验证

```bash
uv run --no-sync python -m phy_data_gen.app --headless
```

这一步不通过时，不要继续写模板和资产逻辑。

---

# 五、Step 3：检查 Cosmos 模板结构

## 文件

```text
phy_data_gen/inspect_template.py
```

目标是输出：

* Stage 默认 prim；
* 所有 Camera；
* PhysicsScene；
* RigidBody；
* Collision；
* 可能的动态物体；
* 每个动态物体的包围盒。

注意：`pxr`、`omni.usd` 等模块应在 `AppLauncher` 启动后再导入。

核心逻辑：

```python
def inspect_stage(stage) -> None:
    from pxr import UsdGeom, UsdPhysics

    for prim in stage.Traverse():
        flags = []

        if prim.IsA(UsdGeom.Camera):
            flags.append("camera")

        if prim.IsA(UsdPhysics.Scene):
            flags.append("physics_scene")

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            flags.append("rigid_body")

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            flags.append("collision")

        if flags:
            print(prim.GetPath(), flags)
```

加载模板：

```python
context = omni.usd.get_context()

if not context.open_stage(str(template_path.resolve())):
    raise RuntimeError(f"Failed to open stage: {template_path}")

for _ in range(20):
    simulation_app.update()

stage = context.get_stage()
inspect_stage(stage)
```

## 验证命令

```bash
uv run --no-sync python -m phy_data_gen.inspect_template \
  --template \
  data/templates/cosmos3/objects_falling/objects_falling_c8917ca2_871/objects_falling_c8917ca2_871.usda
```

## 这一步需要保存的信息

暂时手工记录到：

```text
agents/PLAN.md
```

至少记录：

```yaml
camera_prim: /实际路径
physics_scene_prim: /实际路径
dynamic_prims:
  - /实际动态物体路径1
  - /实际动态物体路径2
ground_prim: /实际地面路径
```

不要根据名字猜 `/World/Camera` 或 `/World/Objects/Object_0`。

---

# 六、Step 4：扫描 MolmoSpaces 资产

## 文件

```text
phy_data_gen/registry.py
```

第一版注册表只保存必要信息：

```python
@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    usd_path: str
    bbox_size: tuple[float, float, float]
    max_dimension: float
    has_rigid_body: bool
    has_collision: bool
    articulated: bool
```

扫描流程：

```text
遍历 USD 文件
    ↓
Usd.Stage.Open()
    ↓
找到 default prim
    ↓
计算包围盒
    ↓
检查 RigidBodyAPI
    ↓
检查 CollisionAPI
    ↓
检查 Joint
    ↓
写 JSON
```

资产查找：

```python
def find_usd_files(root: Path) -> list[Path]:
    suffixes = {".usd", ".usda", ".usdc"}

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
```

包围盒：

```python
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(),
    [UsdGeom.Tokens.default_],
    useExtentsHint=True,
)

bound = bbox_cache.ComputeWorldBound(root_prim)
size = bound.ComputeAlignedBox().GetSize()
```

## 第一版过滤规则

只保留：

```python
valid = (
    has_collision
    and not articulated
    and 0.03 <= max_dimension <= 0.50
)
```

先不实现复杂类别语义。

## 输出

```text
data/assets/registry.json
```

## 验证

```bash
uv run --no-sync python -m phy_data_gen.registry \
  --asset-root data/assets/molmospaces \
  --output data/assets/registry.json
```

然后检查：

```bash
python -m json.tool data/assets/registry.json | head -100
```

---

# 七、Step 5：定义 `EpisodeSpec`

## 文件

```text
phy_data_gen/schemas.py
phy_data_gen/episode.py
```

## `schemas.py`

```python
from pydantic import BaseModel, Field


class PhysicsMaterialSpec(BaseModel):
    static_friction: float = Field(ge=0.0)
    dynamic_friction: float = Field(ge=0.0)
    restitution: float = Field(ge=0.0, le=1.0)


class ObjectSpec(BaseModel):
    object_id: str
    asset_path: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    scale: float = Field(gt=0.0)
    mass: float = Field(gt=0.0)
    material: PhysicsMaterialSpec


class EpisodeSpec(BaseModel):
    episode_id: str
    seed: int
    template_path: str
    backend: str
    duration_seconds: float
    physics_dt: float
    render_fps: int
    objects: list[ObjectSpec]
```

## `episode.py`

负责：

1. 读取注册表；
2. 使用显式 RNG；
3. 采样资产；
4. 采样位置；
5. 采样旋转；
6. 采样质量和摩擦；
7. 保存 JSON。

位置第一版限制在模板中央上方，例如：

```python
x = rng.uniform(-0.5, 0.5)
y = rng.uniform(-0.5, 0.5)
z = rng.uniform(1.0, 2.0)
```

为避免初始重叠，先使用简单网格位置，而不是完全随机：

```text
object_0: (-0.4, 0.0, 1.5)
object_1: ( 0.0, 0.0, 2.0)
object_2: ( 0.4, 0.0, 2.5)
```

## 验证

```bash
uv run --no-sync python -m phy_data_gen.episode \
  --config configs/run.yaml \
  --output outputs/test_episode_spec.json
```

检查：

```bash
python -m json.tool outputs/test_episode_spec.json
```

---

# 八、Step 6：生成替换后的 USDA

## 文件

```text
phy_data_gen/scene.py
```

第一版使用 USD override layer：

```text
Cosmos 原始模板
        ↓ 作为 sublayer
episode.usda
        ↓
禁用模板动态物体
添加 MolmoSpaces reference
设置 transform
设置物理属性
```

不要修改原始 Cosmos 文件，也不要用文本替换 USDA 内容。

## 生成流程

```python
root_layer = Sdf.Layer.CreateNew(str(output_path))
root_layer.subLayerPaths.append(str(template_path.resolve()))

stage = Usd.Stage.Open(root_layer)
```

禁用原物体：

```python
target_prim = stage.OverridePrim(target_path)
target_prim.SetActive(False)
```

引用资产：

```python
prim_path = f"/World/GeneratedObjects/{object_spec.object_id}"
prim = stage.DefinePrim(prim_path, "Xform")
prim.GetReferences().AddReference(
    str(Path(object_spec.asset_path).resolve())
)
```

设置位姿：

```python
xform = UsdGeom.Xformable(prim)
xform.ClearXformOpOrder()

xform.AddTranslateOp().Set(
    Gf.Vec3d(*object_spec.position)
)

xform.AddScaleOp().Set(
    Gf.Vec3d(
        object_spec.scale,
        object_spec.scale,
        object_spec.scale,
    )
)
```

旋转不要第一版手写错误的四元数顺序。集中实现：

```python
def xyzw_to_gf_quat(value):
    x, y, z, w = value
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))
```

## 物理属性

先检查资产内部是否已有：

* RigidBodyAPI；
* CollisionAPI；
* MassAPI。

如果已有，修改已有刚体根节点；不要在外层再创建嵌套刚体。

第一版若资产结构差异较大，先只选一个确认可以正常下落的资产，不急着批量兼容全部 MolmoSpaces。

## 输出

```text
outputs/episode_000000/scene.usda
```

## 验证

先用 Isaac Sim GUI 打开生成的 USDA，确认：

* Cosmos 环境还在；
* 原始动态物体消失；
* 新资产可见；
* 尺寸正常；
* 点击 Play 后能下落。

---

# 九、Step 7：实现 PhysX 仿真

## 文件

```text
phy_data_gen/simulation.py
```

第一版只做 PhysX，不立即实现 Newton。

流程：

```text
AppLauncher
    ↓
打开 episode USDA
    ↓
创建 SimulationContext
    ↓
reset
    ↓
循环 step
    ↓
每帧更新资产状态
```

Isaac Lab 的基础 standalone 流程就是 `AppLauncher + SimulationContext`，并要求在首次 stepping 前调用 `reset()`。([Isaac Sim][1])

时间计算：

```python
num_physics_steps = round(
    duration_seconds / physics_dt
)

capture_every = round(
    1.0 / (render_fps * physics_dt)
)
```

对于：

```text
physics_dt = 1 / 480
render_fps = 30
```

每 16 个物理 step 采集一帧。

## 状态读取建议

第一版优先使用 Isaac Lab 的 `RigidObject` 或 `RigidObjectCollection` 包装生成的刚体 prim。其数据接口提供世界坐标位姿、线速度、角速度和质心状态。([Isaac Sim][2])

实现前先只包装一个对象，验证：

```text
root_pos_w
root_quat_w
root_lin_vel_w
root_ang_vel_w
```

再扩展到多个对象。

---

# 十、Step 8：记录物理状态

## 文件

```text
phy_data_gen/recording.py
```

每个渲染帧保存：

```text
frame
timestamp
object_id
position_x
position_y
position_z
quaternion_x
quaternion_y
quaternion_z
quaternion_w
linear_velocity_x
linear_velocity_y
linear_velocity_z
angular_velocity_x
angular_velocity_y
angular_velocity_z
```

第一版先写 JSONL：

```text
outputs/episode_000000/object_states.jsonl
```

比一开始引入 Parquet 更容易调试。流程稳定后再改为 Parquet。

示例：

```python
class StateRecorder:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(
        self,
        frame: int,
        timestamp: float,
        object_id: str,
        position,
        orientation_xyzw,
        linear_velocity,
        angular_velocity,
    ) -> None:
        self.records.append(
            {
                "frame": frame,
                "timestamp": timestamp,
                "object_id": object_id,
                "position": list(position),
                "orientation_xyzw": list(orientation_xyzw),
                "linear_velocity": list(linear_velocity),
                "angular_velocity": list(angular_velocity),
            }
        )
```

---

# 十一、Step 9：输出 RGB 帧和视频

Replicator 的 `BasicWriter` 可以直接连接 render product 并保存 RGB、语义分割等标注。([Isaac Sim 文档][3])

## `recording.py` 中增加

```python
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=str(frames_dir))

writer = rep.writers.get("BasicWriter")
writer.initialize(
    backend=backend,
    rgb=True,
)
writer.attach(render_product)
```

采集时：

```python
rep.orchestrator.set_capture_on_play(False)
```

每 16 个 physics step：

```python
rep.orchestrator.step(delta_time=0.0)
```

`delta_time=0.0` 可以在采集时保持 timeline 不继续前进，从而让图像与刚体状态对应。([Isaac Sim 文档][3])

完成后：

```python
rep.orchestrator.wait_until_complete()
writer.detach()
render_product.destroy()
```

合成视频：

```bash
ffmpeg \
  -framerate 30 \
  -pattern_type glob \
  -i 'outputs/episode_000000/frames/rgb*.png' \
  -c:v libx264 \
  -crf 18 \
  -pix_fmt yuv420p \
  outputs/episode_000000/rgb.mp4
```

---

# 十二、Step 10：实现最小验证

## 文件

```text
phy_data_gen/validation.py
```

第一版只检查：

```python
def validate_episode(records: list[dict]) -> dict:
    positions = ...
    velocities = ...

    return {
        "finite": bool(...),
        "moved": bool(...),
        "fell": bool(...),
        "max_speed_ok": bool(...),
        "passed": bool(...),
    }
```

建议标准：

```text
所有状态均为有限值
至少一个物体下降超过 0.2 m
至少一个物体速度超过 0.1 m/s
最大速度小于 50 m/s
```

输出：

```text
outputs/episode_000000/validation.json
```

第一版先不做 contact 和视觉可见性检测。

---

# 十三、Step 11：串联 CLI

## 文件

```text
phy_data_gen/cli.py
```

命令设计：

```bash
python -m phy_data_gen.cli inspect-template
python -m phy_data_gen.cli build-registry
python -m phy_data_gen.cli plan
python -m phy_data_gen.cli generate
```

主流程：

```python
def generate(config_path: Path) -> None:
    config = load_config(config_path)

    episode_spec = create_episode_spec(config)
    save_episode_spec(episode_spec)

    scene_path = compile_scene(episode_spec)
    result = run_simulation(scene_path, episode_spec, config)

    save_states(result.states)
    save_validation(validate_episode(result.states))
    encode_video(result.frames)
```

可以在 `pyproject.toml` 增加：

```toml
[project.scripts]
phy-data-gen = "phy_data_gen.cli:main"
```

然后：

```bash
uv sync
phy-data-gen generate --config configs/run.yaml
```

---

# 十四、Step 12：完成单场景闭环

此时完整验收命令：

```bash
uv run --no-sync python -m phy_data_gen.cli \
  generate \
  --config configs/run.yaml \
  --headless
```

期望输出：

```text
outputs/
└── episode_000000/
    ├── episode_spec.json
    ├── scene.usda
    ├── frames/
    ├── rgb.mp4
    ├── object_states.jsonl
    └── validation.json
```

只有这个闭环稳定后，再继续 Scale up。

---

# 十五、Step 13：扩展到多个资产

单资产成功后再实现：

1. 根据注册表筛选有效资产；
2. 先采样类别，再采样实例；
3. 自动计算缩放；
4. 使用包围盒检测初始重叠；
5. 一个场景放置 3～7 个物体；
6. 失败时记录失败原因，不静默换参数。

不要一开始就尝试兼容所有 MolmoSpaces 资产。

建议先人工选出 10 个成功资产，建立：

```text
data/assets/falling_allowlist.txt
```

批量流程稳定后，再替换为自动 QA。

---

# 十六、Step 14：批量生成

增加配置：

```yaml
episodes: 100
start_index: 0
```

每个 episode 使用：

```python
episode_seed = base_seed + episode_index
```

不要在一个 episode 内启动一个新的 Isaac Sim 进程。

正确方式：

```text
启动一次 AppLauncher
    ↓
循环生成多个 episode
    ↓
每个 episode reset / reload stage
    ↓
全部完成后关闭
```

第一版可先每个 episode 启动一次，确保正确；性能优化放在批量功能稳定以后。

---

# 十七、Step 15：最后接入 Newton

Newton 当前仍属于 Isaac Lab 的实验集成，官方安装路径要求 Python 3.12、Isaac Lab 开发版本，并可选配 Isaac Sim 6.0，因此应在 PhysX 路径稳定后单独验证。([Isaac Sim][4])

接入时只新增：

```text
phy_data_gen/backends/
├── __init__.py
├── physx.py
└── newton.py
```

统一接口：

```python
class SimulationBackend(Protocol):
    def run(
        self,
        episode_spec: EpisodeSpec,
        scene_path: Path,
    ) -> SimulationResult:
        ...
```

要求：

```text
EpisodeSpec 不变
资产路径不变
初始位置不变
物理参数不变
只切换 backend 实现
```

不要在 `episode.py`、`scene.py`、`recording.py` 中散布：

```python
if backend == "newton":
```

---

# 推荐实际开发顺序

严格按下面顺序，每一步单独提交：

```text
1. config + environment smoke test
2. Cosmos template inspector
3. MolmoSpaces asset scanner
4. EpisodeSpec
5. 单资产 USD replacement
6. GUI 验证替换场景
7. PhysX headless simulation
8. 单物体状态记录
9. RGB 帧输出
10. FFmpeg 视频
11. episode validation
12. 多资产替换
13. 批量 episode
14. Newton backend
```

**最重要的第一个里程碑不是批量数据，而是：**

```text
一个 Cosmos 场景
+ 一个 MolmoSpaces 资产
+ 一个新的 USDA
+ 一段 5 秒 MP4
+ 一份逐帧状态文件
```

这个里程碑完成以后，其余工作才是可控的扩展。

[1]: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/create_empty.html "Creating an empty scene — Isaac Lab Documentation"
[2]: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html "isaaclab.assets — Isaac Lab Documentation"
[3]: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_getting_started.html "Getting Started Scripts — Isaac Sim Documentation"
[4]: https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/installation.html "Installation — Isaac Lab Documentation"
