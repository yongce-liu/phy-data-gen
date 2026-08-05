"""Backend-agnostic episode specification models.

An :class:`EpisodeSpec` fully describes one physics episode: which template
to load, which assets to drop in, their initial poses and physical
properties. It carries no engine-specific state, so the same spec can be
handed to a PhysX or (later) Newton runner.
"""

from __future__ import annotations

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
    template_path: str
    backend: str
    object_mode: ObjectMode
    duration_seconds: float = Field(gt=0.0)
    physics_dt: float = Field(gt=0.0)
    render_fps: int = Field(gt=0)
    objects: list[ObjectSpec]
    replacements: list[AssetReplacementSpec] = Field(default_factory=list)
