# 项目方案

## 1. 项目目标

基于：

* 一个手动下载的 Cosmos 3 物理场景 USDA；
* 一批手动下载的 MolmoSpaces USD 资产；
* Isaac Sim 6.0；
* Isaac Lab；
* PhysX / Newton；

自动完成：

1. 读取 Cosmos 场景模板；
2. 识别其中可替换的动态物体；
3. 从资产库中选择合适资产；
4. 替换资产并随机化尺寸、位姿和物理参数；
5. 运行物理仿真；
6. 输出 RGB 视频、场景文件和物体状态数据。

第一阶段只支持 `objects_falling` 类型的刚体下落与碰撞场景。

---

## 2. 手动准备的数据

项目不负责自动下载资产。将素材放到固定目录：

```text
data/
├── templates/
│   └── cosmos3/
│       └── objects_falling.usda
│
└── assets/
    └── molmospaces/
        └── thor/
            ├── asset_001.usd
            ├── asset_002.usd
            └── ...
```

要求：

* Cosmos USDA 能在 Isaac Sim 中直接打开；
* MolmoSpaces 资产保持原始目录结构；
* 原始文件只读，不直接修改。

---

## 3. 核心流程

```text
Cosmos USDA
    ↓
分析场景结构
    ↓
识别地面、相机、灯光和动态物体
    ↓
扫描 MolmoSpaces 资产
    ↓
生成资产注册表
    ↓
按类别和尺寸筛选资产
    ↓
生成 EpisodeSpec
    ↓
替换 Cosmos 场景中的动态物体
    ↓
设置质量、摩擦、弹性和初始位姿
    ↓
运行预仿真
    ↓
验证场景是否有效
    ↓
正式渲染
    ↓
保存 MP4、USDA 和物理状态
```

---

## 4. 项目结构

```text
physical-data-gen/
├── pyproject.toml
├── uv.lock
├── AGENTS.md
├── README.md
│
├── configs/
│   ├── run.yaml
│   ├── physx.yaml
│   └── newton.yaml
│
├── data/
│   ├── templates/
│   │   └── cosmos3/
│   │       └── objects_falling.usda
│   ├── assets/
│   │   └── molmospaces/
│   │       └── thor/
│   └── registry/
│       └── assets.parquet
│
├── pipeline/
│   ├── config.py
│   ├── schemas.py
│   │
│   ├── assets/
│   │   ├── scanner.py
│   │   ├── registry.py
│   │   └── sampler.py
│   │
│   ├── template/
│   │   ├── inspector.py
│   │   └── compiler.py
│   │
│   ├── simulation/
│   │   ├── runner.py
│   │   ├── physx.py
│   │   └── newton.py
│   │
│   ├── recording/
│   │   ├── states.py
│   │   └── video.py
│   │
│   ├── validation.py
│   └── orchestrator.py
│
├── scripts/
│   ├── inspect_template.py
│   ├── build_registry.py
│   └── generate.py
│
├── outputs/
│   └── episodes/
│
└── tests/
```

不需要一开始建立复杂的 agent、provider、campaign 或多 domain 体系。等最小流程稳定后再扩展。

---

## 5. 关键数据结构

每个生成任务先保存为后端无关的 `EpisodeSpec`：

```json
{
  "episode_id": "000001",
  "seed": 1001,
  "template": "objects_falling",
  "backend": "physx",
  "objects": [
    {
      "asset_id": "mug_001",
      "usd_path": "data/assets/molmospaces/thor/mug_001.usd",
      "position": [0.1, 0.2, 1.5],
      "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
      "scale": 1.0,
      "mass": 0.4,
      "static_friction": 0.6,
      "dynamic_friction": 0.5,
      "restitution": 0.2
    }
  ]
}
```

同一个 `EpisodeSpec` 后续可以分别交给 PhysX 或 Newton。

### 5.1 已解析的模板 prim 路径（第一版实测）

从 Cosmos 模板
`data/templates/cosmos3/objects_falling/objects_falling_c8917ca2_871/objects_falling_c8917ca2_871.usda`
检查得到的关键 prim（已固化到 `configs/run.yaml` 的 `scene` 段，代码不再猜测路径）：

```text
world_prim          = /World
physics_scene_prim  = /PhysicsScene
ground_prim         = /World/Ground
camera_prim         = /World/Camera_Side
dynamic_prims       = /World/Prop_* 和 /World/S871_*（共 13 个）
```

生成的物体统一挂在 `/World/GeneratedObjects/<object_id>` 下；模板原动态
物体通过 override prim `SetActive(False)` 禁用，不修改原始 USDA。

---

## 6. 资产注册表

首次运行时扫描 MolmoSpaces 目录，生成：

```text
data/registry/assets.parquet
```

最低记录：

```text
asset_id
usd_path
category
bbox_x
bbox_y
bbox_z
max_dimension
has_rigid_body
has_collision
articulated
valid_for_falling
```

第一版筛选规则：

```text
具有碰撞体
不是 articulated asset
最大尺寸在合理范围内
属于小型桌面物体
能够通过简单下落测试
```

冰箱、桌子、床、沙发等大型资产不得进入下落资产池。

---

## 7. 场景替换

不要修改原始 Cosmos USDA。

每个 episode：

1. 打开模板；
2. 禁用模板中的原动态物体；
3. 创建新的 prim；
4. reference MolmoSpaces USD；
5. 设置位置、旋转和缩放；
6. 设置刚体和物理材质；
7. 保存新的 episode USDA。

推荐输出：

```text
outputs/episodes/episode_000001/
├── episode_spec.json
├── scene.usda
├── rgb.mp4
├── object_states.parquet
└── validation.json
```

---

## 8. 仿真后端

### PhysX

作为第一阶段默认后端：

* 资产检查；
* 批量数据生成；
* RGB 渲染；
* 状态记录；
* 正式输出。

### Newton

第二阶段接入：

* 使用相同 `EpisodeSpec`；
* 使用相同 USD 资产；
* 只在后端边界层处理差异；
* 初期只验证简单刚体下落；
* 不要求与 PhysX 逐帧完全一致。

项目代码尽量通过 Isaac Lab 的统一接口操作场景和资产。

---

## 9. 两阶段生成

### Preview

先进行低成本仿真，不输出正式视频。

检查：

* 是否出现 NaN；
* 是否存在初始穿透；
* 物体是否发生运动；
* 是否与地面发生碰撞；
* 是否飞出场景；
* 最大速度是否异常。

### Final

只有 Preview 通过后才：

* 正式渲染；
* 记录物体状态；
* 输出视频；
* 保存场景和配置。

---

## 10. 最小 CLI

构建资产注册表：

```bash
uv run --no-sync python scripts/build_registry.py
```

检查 Cosmos 模板：

```bash
uv run --no-sync python scripts/inspect_template.py
```

生成一个 episode：

```bash
uv run --no-sync python scripts/generate.py \
    --config configs/run.yaml \
    --episodes 1 \
    --backend physx
```

批量生成：

```bash
uv run --no-sync python scripts/generate.py \
    --config configs/run.yaml \
    --episodes 100 \
    --backend physx
```

---

## 11. 开发顺序

### 阶段一：最小闭环

* 打开 Cosmos USDA；
* 找到动态物体和相机；
* 引用一个 MolmoSpaces 资产；
* 替换一个物体；
* 运行 PhysX；
* 输出短视频。

### 阶段二：资产自动化

* 扫描资产；
* 生成注册表；
* 按类别和尺寸筛选；
* 随机选择 3～7 个资产；
* 自动防止初始重叠。

### 阶段三：批量数据

* 生成可复现的 `EpisodeSpec`；
* Preview 验证；
* 批量输出视频和状态；
* 支持失败记录和断点恢复。

### 阶段四：Newton

* 相同 `EpisodeSpec` 切换 Newton；
* 验证刚体状态读取；
* 对比 PhysX 和 Newton 结果。

### 阶段五：扩展物理现象

新增一个 Cosmos 模板和对应场景逻辑：

```text
objects_falling
rolling_ramp
dominoes
projectile_collision
```

资产扫描、记录、验证和输出代码保持复用。

---

## 12. 第一版验收标准

第一版只需要满足：

1. `uv sync` 可以构建环境；
2. Isaac Sim 和 Isaac Lab 可以启动；
3. Cosmos USDA 可以正常加载；
4. MolmoSpaces 资产可以被 reference；
5. 可以替换至少一个动态物体；
6. 可以修改质量、摩擦和弹性；
7. 可以运行 5 秒仿真；
8. 可以输出 MP4；
9. 可以保存逐帧物体位姿和速度；
10. 相同 seed 可以复现相同初始场景。
