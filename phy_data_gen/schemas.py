"""Backend-agnostic episode specification models.

An :class:`EpisodeSpec` fully describes one physics episode: which template
to load, which assets to drop in, their initial poses and physical
properties. It carries no engine-specific state, so the same spec can be
handed to a PhysX or (later) Newton runner.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class EpisodeSpec(BaseModel):
    """Complete, reproducible description of a single episode."""

    episode_id: str
    seed: int
    template_path: str
    backend: str
    duration_seconds: float = Field(gt=0.0)
    physics_dt: float = Field(gt=0.0)
    render_fps: int = Field(gt=0)
    objects: list[ObjectSpec]
