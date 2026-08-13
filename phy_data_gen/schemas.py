"""Backend-agnostic episode specification models.

An :class:`EpisodeSpec` fully describes one physics episode: which template
to load, which assets to drop in, their initial poses and physical
properties. It carries no engine-specific state, so the same spec can be
handed to a PhysX or (later) Newton runner.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from phy_data_gen.config import ObjectMode


class PhysicsMaterialSpec(BaseModel):
    """Isotropic physics material for a dropped object."""

    static_friction: float = Field(ge=0.0)
    dynamic_friction: float = Field(ge=0.0)
    restitution: float = Field(ge=0.0, le=1.0)


class ObjectSpec(BaseModel):
    """One object to place into the scene."""

    object_id: str
    asset_path: str
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    scale: float = Field(gt=0.0)
    mass: float = Field(gt=0.0)
    material: PhysicsMaterialSpec
    kind: Literal["asset", "sphere", "box"] = "asset"
    # ``radius`` is the sphere radius (m) for ``sphere`` objects, or the
    # half-extent along every axis for ``box`` objects.
    radius: float | None = None
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    # Initial body velocities, authored as physics:velocity /
    # physics:angularVelocity on the rigid-body prim (m/s, rad/s).
    initial_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ``record=False`` objects are simulated but never get per-object state
    # views (e.g. hundreds of sand grains or static container walls).
    record: bool = True
    # ``dynamic=False`` objects carry collision but no rigid body (static
    # obstacles, container walls, fixed objects for category 03).
    dynamic: bool = True


class CameraSpec(BaseModel):
    """A camera to author into the episode scene."""

    prim_path: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float] | None = None
    orientation_xyzw: tuple[float, float, float, float] | None = None
    focal_length: float = 24.0
    horizontal_aperture: float = 20.955
    vertical_aperture: float = 11.79


class AssetReplacementSpec(BaseModel):
    """A local asset substituted into an existing template rigid body."""

    object_id: str
    target_prim_path: str
    asset_path: str
    scale: float = Field(gt=0.0)
    translation: tuple[float, float, float]
    create_rigid_body: bool = False
    asset_rigid_body_path: str | None = None


class EpisodeSpec(BaseModel):
    """Complete, reproducible description of a single episode."""

    episode_id: str
    seed: int
    template_path: str | None = None
    backend: str
    object_mode: ObjectMode
    duration_seconds: float = Field(gt=0.0)
    physics_dt: float = Field(gt=0.0)
    render_fps: int = Field(gt=0)
    objects: list[ObjectSpec]
    replacements: list[AssetReplacementSpec] = Field(default_factory=list)
    cameras: dict[str, CameraSpec] = Field(default_factory=dict)
    # ``runner`` selects the simulation backend path. Only "rigid" exists on
    # dev; category 08 supplies "deformable" in phy_data_gen/runners/.
    runner: Literal["rigid", "deformable"] = "rigid"
    # Category-specific payload (e.g. deformable material params for the
    # soft-ball runner). Ignored by the rigid path.
    metadata: dict[str, object] = Field(default_factory=dict)
