"""Run configuration loading.

Parameters are managed through a YAML file. CLI arguments may override the
loaded values elsewhere. Paths are always resolved to ``pathlib.Path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SceneConfig:
    """Prim paths resolved from the Cosmos template.

    These are recorded from ``inspect_template`` output and must not be
    guessed from prim names.
    """

    name: str
    world_prim: str
    physics_scene_prim: str
    ground_prim: str
    cameras: dict[str, str]
    dynamic_prims: tuple[str, ...]


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


def load_config(path: Path) -> RunConfig:
    """Load a :class:`RunConfig` from a YAML file."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    scene_raw = raw["scene"]
    scene = SceneConfig(
        name=str(scene_raw["name"]),
        world_prim=str(scene_raw["world_prim"]),
        physics_scene_prim=str(scene_raw["physics_scene_prim"]),
        ground_prim=str(scene_raw["ground_prim"]),
        cameras={str(name): str(path) for name, path in scene_raw["cameras"].items()},
        dynamic_prims=tuple(str(prim) for prim in scene_raw["dynamic_prims"]),
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
    )
