"""Shared helpers for dataset category samplers.

All helpers are deterministic (no global RNG state) and carry no Isaac Sim
runtime dependency; they only build pydantic specs and small USD-authoring
constants.
"""

from __future__ import annotations

import json
import math
import random
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
    half_extents: tuple[float, float, float] | None = None,
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
        half_extents=half_extents,
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


def jittered_cameras(
    center: tuple[float, float, float],
    rng,
    names: tuple[str, ...] = ("Side", "Front", "Top", "Action"),
) -> dict[str, CameraSpec]:
    """A 4-camera rig that frames the table like the Cosmos billiards cameras.

    Anchor positions are the template's proven table-framing poses (low and
    close to the table), jittered per episode so each shot differs slightly
    without losing the table/balls in frame.
    """

    anchors = {
        "Side": (0.0, -2.0, 0.75),     # behind, looking along the table
        "Front": (0.0, 2.0, 0.9),      # front, slightly higher
        "Top": (0.0, 0.0, 3.2),        # overhead
        "Action": (1.4, 0.7, 1.05),    # near the impact zone
    }
    specs = {}
    for name in names:
        ax, ay, az = anchors.get(name, (0.0, -2.0, 0.75))
        if name == "Top":
            pos = (ax, ay, az + rng.uniform(-0.4, 0.4))
        else:
            # Jitter around the anchor: radius 0.15-0.3 m, height ±0.2 m.
            d = rng.uniform(0.15, 0.3)
            ang = rng.uniform(0, 2 * math.pi)
            pos = (ax + d * math.cos(ang), ay + d * math.sin(ang), az + rng.uniform(-0.2, 0.2))
        focal = rng.uniform(24.0, 38.0)
        specs[name] = camera_spec(name, pos, center, focal_length=focal)
    return specs


def standard_config_cameras(names: tuple[str, ...] = ("Side", "Front", "Top", "Action")) -> dict[str, str]:
    return {name: f"/World/Camera_{name}" for name in names}


def background_props(
    config: RunConfig,
    rng,
    count: int = 2,
    prefixes: set[str] | None = None,
) -> list[ObjectSpec]:
    """Return a few static registry assets as fixed background decoration.

    Props are placed off the billiards playing surface (around the table) and
    are ``dynamic=False`` + ``record=False`` so they only add visual variety,
    never interfere with the physics or get recorded.
    """

    if prefixes is None:
        prefixes = {"Vase_Medium", "Vase_Tall", "Cup", "Bowl", "Plate", "Houseplant", "Candle"}
    assets = registry_assets_by_prefix(config.registry_path, prefixes)
    if not assets:
        return []

    props = []
    rng.shuffle(assets)
    # Table is ~2.4 x 4.2 m; place props off the corners as background.
    for i in range(min(count, len(assets))):
        asset = assets[i]
        dim = float(asset["max_dimension"])
        scale = rng.uniform(0.6, 1.2) * (0.15 / dim if dim > 0 else 1.0)
        # Positions just outside the table corners, at floor level.
        corner = rng.choice([(-1.6, 2.6), (1.6, 2.6), (-1.6, -2.6), (1.6, -2.6)])
        jitter = (rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
        props.append(
            ObjectSpec(
                object_id=f"bg_prop_{i}",
                asset_path=str(asset["usd_path"]),
                position=(corner[0] + jitter[0], corner[1] + jitter[1], 0.1 * scale),
                orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                scale=scale,
                mass=1e9,
                material=make_material(0.1),
                kind="asset",
                dynamic=False,
                record=False,
            )
        )
    return props


def build_episode(
    config: RunConfig,
    episode_id: str,
    template_path: str,
    objects: list[ObjectSpec],
    cameras: dict[str, CameraSpec],
    duration_seconds: float | None = None,
    physics_dt: float | None = None,
    runner: str = "rigid",
    metadata: dict[str, object] | None = None,
    background: bool = True,
) -> EpisodeSpec:
    """Assemble an EpisodeSpec for a category episode.

    When ``background`` is True (default), a couple of static registry assets
    are added around the table for visual variety.
    """

    if background:
        rng = random.Random(config.seed + 999_983)
        objects = list(objects) + background_props(config, rng)

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
        metadata=metadata or {},
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
