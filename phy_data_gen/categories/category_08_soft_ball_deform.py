"""Category 08 — soft-ball collision and deformation.

One or two soft (volume-deformable) balls collide with a rigid wall, floor or
each other. Sub-variants (``seed % 6``):

* elastic_compression — soft ball squeezed against a rigid wall/floor, recovers.
* rebound — soft ball dropped from a height, bounces.
* viscoelastic — low modulus + high damping, slow recovery.
* plastic_approx — very low modulus, recovery negligible within the episode
  (PhysX has no native plasticity; this is a viscoelastic approximation).
* multi_soft — two soft balls collide head-on.

Soft-ball geometry and material are authored by the deformable runner's
``build_scene_hook`` (requires the PhysX runtime). The sampler only records
parameters in ``EpisodeSpec.metadata``.
"""

from __future__ import annotations

import random

from phy_data_gen.categories.common import (
    ball_spec,
    build_episode,
    camera_spec,
    make_material,
)
from phy_data_gen.config import RunConfig
from phy_data_gen.schemas import CameraSpec, EpisodeSpec, ObjectSpec


def _cameras() -> dict[str, CameraSpec]:
    """Side + Top cameras (matches the 2-camera 960x540 config)."""

    return {
        "Side": camera_spec("Side", (2.5, 0.0, 0.8), _TABLE_CENTER),
        "Top": camera_spec("Top", (0.0, 0.0, 3.0), _TABLE_CENTER),
    }

_TABLE_CENTER = (0.0, 0.0, 0.0)
_VARIANTS = [
    "elastic_compression",
    "rebound",
    "viscoelastic",
    "plastic_approx",
    "multi_soft",
    "elastic_compression",
]


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)

    r = rng.uniform(0.05, 0.08)
    youngs = rng.uniform(1e3, 1e6)
    poisson = rng.uniform(0.3, 0.49)
    density = rng.uniform(50.0, 1000.0)

    if variant in {"viscoelastic", "plastic_approx"}:
        youngs = rng.uniform(1e3, 5e4)  # soft / very soft
    elif variant == "rebound":
        youngs = rng.uniform(2e4, 2e5)

    objects: list[ObjectSpec] = []

    if variant == "multi_soft":
        # Two soft balls colliding head-on.
        objects.append(
            ball_spec(
                "soft_ball_a",
                (-0.3, 0.0, r),
                r,
                0.2,
                make_material(0.5),
                velocity=(rng.uniform(2.0, 6.0), 0.0, 0.0),
                color=(0.8, 0.3, 0.3),
            )
        )
        objects.append(
            ball_spec(
                "soft_ball_b",
                (0.3, 0.0, r),
                r,
                0.2,
                make_material(0.5),
                velocity=(-rng.uniform(2.0, 6.0), 0.0, 0.0),
                color=(0.3, 0.3, 0.8),
            )
        )
    else:
        # One soft ball dropped or pushed against a rigid wall (the billiards
        # table + bumpers act as the rigid obstacle).
        if variant == "elastic_compression":
            pos = (-0.8, 0.0, r)
            vel = (rng.uniform(2.0, 6.0), 0.0, 0.0)
        elif variant == "rebound":
            pos = (0.0, 0.0, r + rng.uniform(0.3, 0.8))
            vel = (0.0, 0.0, 0.0)
        elif variant == "viscoelastic":
            pos = (-0.8, 0.0, r)
            vel = (rng.uniform(1.0, 3.0), 0.0, 0.0)
        else:  # plastic_approx
            pos = (-0.8, 0.0, r)
            vel = (rng.uniform(1.0, 4.0), 0.0, 0.0)
        objects.append(
            ball_spec(
                "soft_ball",
                pos,
                r,
                0.2,
                make_material(0.5),
                velocity=vel,
                color=(0.8, 0.3, 0.3),
            )
        )

    metadata = {
        "deformable": {
            "variant": variant,
            "youngs_modulus": youngs,
            "poissons_ratio": poisson,
            "density": density,
        }
    }

    return build_episode(
        config,
        episode_id,
        template_path,
        objects,
        _cameras(),
        duration_seconds=3.0,
        physics_dt=1.0 / 120.0,
        runner="deformable",
        metadata=metadata,
    )
