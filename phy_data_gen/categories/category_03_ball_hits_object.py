"""Category 03 — ball hits an object.

A ball strikes a registry object placed on the table. Sub-variants:

* object softness/hardness — restitution of the object (and ball) varies.
* object mass — light (0.05-0.2 kg) to heavy (2-10 kg).
* movable vs fixed — the object is a dynamic rigid body or a static obstacle.
* rebound — ball bounces off a heavy fixed object (high ball restitution).
* intercepted — ball hits the object mid-air (object on the table edge, ball
  arrives at a slight downward angle) instead of rolling in.

The object is a MolmoSpaces THOR asset (Bowl/Plate/Cup/Vase/Pan/Pot) whose
mass is overridden to the sampled value.
"""

from __future__ import annotations

import random

from phy_data_gen.categories.common import (
    ball_spec,
    build_episode,
    make_material,
    registry_assets_by_prefix,
    standard_cameras,
)
from phy_data_gen.config import RunConfig
from phy_data_gen.schemas import EpisodeSpec, ObjectSpec

_TABLE_CENTER = (0.0, 0.0, 0.0)
_OBJECT_PREFIXES = {"Bowl", "Plate", "Cup", "Vase_Medium", "Vase_Tall", "Vase_Flat", "Pan", "Pot"}
_VARIANTS = ["soft", "hard", "light", "heavy", "movable", "fixed", "rebound", "intercepted"]


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)

    assets = registry_assets_by_prefix(config.registry_path, _OBJECT_PREFIXES)
    if not assets:
        raise RuntimeError(
            f"Registry has no objects in {sorted(_OBJECT_PREFIXES)}; "
            "run build-registry after downloading assets"
        )
    asset = rng.choice(assets)
    asset_path = str(asset["usd_path"])
    asset_dim = float(asset["max_dimension"])

    r_ball = rng.uniform(0.04, 0.06)
    ball_speed = rng.uniform(1.0, 10.0)
    ball_rest = rng.uniform(0.2, 0.9)

    if variant in {"soft", "hard"}:
        object_rest = rng.uniform(0.0, 0.4) if variant == "soft" else rng.uniform(0.5, 0.8)
        object_mass = rng.uniform(0.3, 2.0)
    elif variant == "light":
        object_rest = rng.uniform(0.1, 0.6)
        object_mass = rng.uniform(0.05, 0.2)
    elif variant == "heavy":
        object_rest = rng.uniform(0.1, 0.5)
        object_mass = rng.uniform(2.0, 10.0)
    elif variant == "movable":
        object_rest = rng.uniform(0.1, 0.5)
        object_mass = rng.uniform(0.2, 1.0)
    elif variant == "fixed":
        object_rest = rng.uniform(0.1, 0.7)
        object_mass = 1e9  # treated as static below
    elif variant == "rebound":
        object_rest = rng.uniform(0.6, 0.9)
        object_mass = 1e9  # heavy fixed
        ball_rest = rng.uniform(0.7, 1.0)
    else:  # intercepted
        object_rest = rng.uniform(0.1, 0.5)
        object_mass = rng.uniform(0.1, 1.0)

    dynamic = variant not in {"fixed", "rebound"}

    # Object resting on the table, ball rolling toward it from the side.
    # Scale so the object's max dimension is ~0.1 m; its bbox centre may not be
    # the origin, so lift by the scaled max dimension to keep it above the table.
    object_scale = rng.uniform(0.8, 1.2) * (0.1 / asset_dim if asset_dim > 0 else 1.0)
    object_pos = (0.3, 0.0, 0.1 * object_scale + 0.02)

    if variant == "intercepted":
        ball_pos = (-0.8, 0.0, r_ball + 0.25)
        ball_vel = (ball_speed, 0.0, -0.3)
    else:
        ball_pos = (-0.8, 0.0, r_ball)
        ball_vel = (ball_speed, 0.0, 0.0)

    objects: list[ObjectSpec] = [
        ball_spec(
            "ball",
            ball_pos,
            r_ball,
            0.15,
            make_material(ball_rest),
            velocity=ball_vel,
            color=(0.95, 0.15, 0.15),
        ),
    ]

    objects.append(
        ObjectSpec(
            object_id="target_object",
            asset_path=asset_path,
            position=object_pos,
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            scale=object_scale,
            mass=1e9 if not dynamic else object_mass,
            material=make_material(object_rest),
            kind="asset",
            dynamic=dynamic,
        )
    )

    return build_episode(
        config,
        episode_id,
        template_path,
        objects,
        standard_cameras(_TABLE_CENTER),
    )
