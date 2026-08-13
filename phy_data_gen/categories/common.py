"""Shared helpers for dataset category samplers.

All helpers are deterministic (no global RNG state) and carry no Isaac Sim
runtime dependency; they only build pydantic specs and small USD-authoring
constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from phy_data_gen.config import RunConfig
from phy_data_gen.schemas import (
    CameraSpec,
    EpisodeSpec,
    ObjectSpec,
    PhysicsMaterialSpec,
)


def load_registry(registry_path: Path) -> list[dict]:
    """Load asset records from the registry JSON (empty if missing)."""

    if not registry_path.is_file():
        return []
    with registry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return list(payload.get("assets", []))


def registry_assets_by_prefix(
    registry_path: Path, prefixes: set[str]
) -> list[dict]:
    """Return registry assets whose category prefix is in ``prefixes``."""

    assets = load_registry(registry_path)
    selected = []
    for asset in assets:
        asset_id = str(asset.get("asset_id", ""))
        parts = asset_id.split("__")
        if len(parts) >= 3:
            category = parts[2].rsplit("_", 1)[0]
        else:
            category = ""
        if category in prefixes:
            selected.append(asset)
    return selected


def make_material(
    restitution: float,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.4,
) -> PhysicsMaterialSpec:
    return PhysicsMaterialSpec(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )


def ball_spec(
    object_id: str,
    position: tuple[float, float, float],
    radius: float,
    mass: float,
    material: PhysicsMaterialSpec,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: tuple[float, float, float] = (0.9, 0.1, 0.1),
    record: bool = True,
    dynamic: bool = True,
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        asset_path="",
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=1.0,
        mass=mass,
        material=material,
        kind="sphere",
        radius=radius,
        color=color,
        initial_linear_velocity=velocity,
        initial_angular_velocity=angular_velocity,
        record=record,
        dynamic=dynamic,
    )


def box_spec(
    object_id: str,
    position: tuple[float, float, float],
    half_extent: float,
    mass: float,
    material: PhysicsMaterialSpec,
    color: tuple[float, float, float] = (0.5, 0.4, 0.3),
    record: bool = True,
    dynamic: bool = True,
) -> ObjectSpec:
    return ObjectSpec(
        object_id=object_id,
        asset_path="",
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=1.0,
        mass=mass,
        material=material,
        kind="box",
        radius=half_extent,
        color=color,
        record=record,
        dynamic=dynamic,
    )


def camera_spec(
    name: str,
    position: tuple[float, float, float],
    look_at: tuple[float, float, float],
    focal_length: float = 24.0,
) -> CameraSpec:
    return CameraSpec(
        prim_path=f"/World/Camera_{name}",
        position=position,
        look_at=look_at,
        focal_length=focal_length,
    )


def standard_cameras(center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, CameraSpec]:
    """A generic 4-camera ring for tabletop ball scenes."""

    return {
        "Side": camera_spec("Side", (3.0, 0.0, 0.8), center),
        "Front": camera_spec("Front", (0.0, 3.0, 0.8), center),
        "Top": camera_spec("Top", (0.0, 0.0, 3.5), center),
        "Action": camera_spec("Action", (1.5, 1.5, 0.6), center),
    }


def standard_config_cameras(names: tuple[str, ...] = ("Side", "Front", "Top", "Action")) -> dict[str, str]:
    return {name: f"/World/Camera_{name}" for name in names}


def build_episode(
    config: RunConfig,
    episode_id: str,
    template_path: str,
    objects: list[ObjectSpec],
    cameras: dict[str, CameraSpec],
    duration_seconds: float | None = None,
    physics_dt: float | None = None,
    runner: str = "rigid",
) -> EpisodeSpec:
    """Assemble an EpisodeSpec for a category episode."""

    return EpisodeSpec(
        episode_id=episode_id,
        seed=config.seed,
        template_path=template_path,
        backend=config.backend,
        object_mode=config.scene.object_mode,
        duration_seconds=duration_seconds or config.simulation.duration_seconds,
        physics_dt=physics_dt or config.simulation.physics_dt,
        render_fps=config.simulation.render_fps,
        objects=objects,
        cameras=cameras,
        runner=runner,
    )


def quat_from_yaw_pitch(yaw: float, pitch: float = 0.0) -> tuple[float, float, float, float]:
    """Return an (x, y, z, w) unit quaternion from yaw/pitch angles (radians)."""

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    return (
        sp * cy,
        0.0,
        -cp * sy,
        cp * cy,
    )
