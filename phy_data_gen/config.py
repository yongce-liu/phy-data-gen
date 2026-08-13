"""Run configuration loading.

Parameters are managed through a YAML file. CLI arguments may override the
loaded values elsewhere. Paths are always resolved to ``pathlib.Path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

ObjectMode = Literal["generated_objects", "template_dynamics", "replace_assets", "procedural"]


@dataclass(frozen=True)
class ProceduralConfig:
    """Backdrop authored from scratch when ``template_path`` is not set."""

    build_ground: bool = True
    ground_size: float = 8.0
    ground_color: tuple[float, float, float] = (0.55, 0.55, 0.58)
    table: bool = False
    table_size: tuple[float, float] = (2.4, 4.2)
    walls: bool = False
    wall_height: float = 0.6
    sand_tray: bool = False


@dataclass(frozen=True)
class SceneConfig:
    """Prim paths resolved from the Cosmos template.

    These are recorded from ``inspect_template`` output and must not be
    guessed from prim names.
    """

    name: str
    object_mode: ObjectMode
    world_prim: str
    physics_scene_prim: str
    ground_prim: str
    cameras: dict[str, str]
    dynamic_prims: tuple[str, ...]
    replace_initially_moving_objects: bool = True
    procedural: ProceduralConfig | None = None


@dataclass(frozen=True)
class SimulationConfig:
    physics_dt: float
    render_fps: int
    duration_seconds: float


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    depth_scale_meters: float
    rgb_encoder: str


@dataclass(frozen=True)
class RunConfig:
    template_root: Path
    asset_root: Path
    registry_path: Path
    output_root: Path
    backend: str
    seed: int
    num_objects: int
    scene: SceneConfig
    simulation: SimulationConfig
    render: RenderConfig
    template_seed: int | None = None
    category: str | None = None


def load_config(path: Path) -> RunConfig:
    """Load a :class:`RunConfig` from a YAML file."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    scene_raw = raw["scene"]
    object_mode = str(scene_raw["object_mode"])
    if object_mode not in {
        "generated_objects",
        "template_dynamics",
        "replace_assets",
        "procedural",
    }:
        raise ValueError(f"Unsupported scene.object_mode: {object_mode}")

    procedural_raw = scene_raw.get("procedural")
    procedural = None
    if procedural_raw:
        procedural = ProceduralConfig(
            build_ground=bool(procedural_raw.get("build_ground", True)),
            ground_size=float(procedural_raw.get("ground_size", 8.0)),
            ground_color=tuple(
                float(value)
                for value in procedural_raw.get(
                    "ground_color", (0.55, 0.55, 0.58)
                )
            ),
            table=bool(procedural_raw.get("table", False)),
            table_size=tuple(
                float(value)
                for value in procedural_raw.get("table_size", (2.4, 4.2))
            ),
            walls=bool(procedural_raw.get("walls", False)),
            wall_height=float(procedural_raw.get("wall_height", 0.6)),
            sand_tray=bool(procedural_raw.get("sand_tray", False)),
        )

    scene = SceneConfig(
        name=str(scene_raw["name"]),
        object_mode=object_mode,
        world_prim=str(scene_raw["world_prim"]),
        physics_scene_prim=str(scene_raw["physics_scene_prim"]),
        ground_prim=str(scene_raw["ground_prim"]),
        cameras={str(name): str(path) for name, path in scene_raw["cameras"].items()},
        dynamic_prims=tuple(str(prim) for prim in scene_raw["dynamic_prims"]),
        replace_initially_moving_objects=bool(
            scene_raw.get("replace_initially_moving_objects", True)
        ),
        procedural=procedural,
    )

    return RunConfig(
        template_root=Path(raw["template_root"]),
        asset_root=Path(raw["asset_root"]),
        registry_path=Path(raw["registry_path"]),
        output_root=Path(raw["output_root"]),
        backend=str(raw["backend"]),
        seed=int(raw["seed"]),
        num_objects=int(raw["num_objects"]),
        scene=scene,
        simulation=SimulationConfig(**raw["simulation"]),
        render=RenderConfig(**raw["render"]),
        template_seed=(
            int(raw["template_seed"])
            if raw.get("template_seed") is not None
            else None
        ),
        category=str(raw["category"]) if raw.get("category") is not None else None,
    )
