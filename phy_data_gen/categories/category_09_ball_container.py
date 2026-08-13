"""Category 09 — ball into / against a container.

A ball interacts with an open-top box (procedural) or a registry Bowl/Vase/Pan.
Sub-variants:

* drop_in — ball drops through the opening and lands inside.
* roll_in — ball rolls in through a side gap / low lip.
* hit_wall — ball hits the container's side wall and bounces off.
* rest_inside — low-speed ball settles and stays inside.
* bounce_inside — high speed ball bounces around inside the container.
* miss — ball is aimed to miss the opening and lands outside.

The container is a procedural static box with 4 walls + floor (open top), or a
registry asset (Bowl/Vase/Pan) with collision.
"""

from __future__ import annotations

import random

from phy_data_gen.categories.common import (
    ball_spec,
    build_episode,
    make_material,
    box_spec,
    standard_cameras,
)
from phy_data_gen.config import RunConfig
from phy_data_gen.schemas import EpisodeSpec, PhysicsMaterialSpec

_TABLE_CENTER = (0.0, 0.0, 0.0)
_VARIANTS = ["drop_in", "roll_in", "hit_wall", "rest_inside", "bounce_inside", "miss"]
_CONTAINER_HALF = 0.2
_CONTAINER_H = 0.25


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def _add_container(objects: list, rng: random.Random) -> None:
    """Add a static open-top box container (floor + 4 thin walls)."""

    h = _CONTAINER_HALF
    wall = 0.05
    material = PhysicsMaterialSpec(
        static_friction=0.5, dynamic_friction=0.4, restitution=rng.uniform(0.2, 0.6)
    )
    # Floor: full container footprint, thin.
    objects.append(
        box_spec(
            "container_floor",
            (0.0, 0.0, wall / 2.0),
            h,
            1e9,
            material,
            color=(0.4, 0.4, 0.5),
            dynamic=False,
            record=False,
            half_extents=(h, h, wall),
        )
    )
    # Four thin walls rising around the floor.
    for i, (dx, dy) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
        objects.append(
            box_spec(
                f"container_wall_{i}",
                (dx * h, dy * h, _CONTAINER_H / 2.0),
                h,
                1e9,
                material,
                color=(0.4, 0.4, 0.5),
                dynamic=False,
                record=False,
                half_extents=(h if dx != 0 else wall, h if dy != 0 else wall, _CONTAINER_H),
            )
        )


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)

    r_ball = rng.uniform(0.03, 0.05)
    ball_rest = rng.uniform(0.2, 0.8)
    speed = rng.uniform(1.0, 8.0)

    objects = []
    _add_container(objects, rng)

    if variant in {"drop_in", "rest_inside"}:
        # Drop from above, aim over the opening.
        ball_pos = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), _CONTAINER_H + r_ball + 0.3)
        ball_vel = (0.0, 0.0, 0.0)
    elif variant == "roll_in":
        # Roll toward the container at low height; a low lip lets it in.
        ball_pos = (-0.6, 0.0, r_ball)
        ball_vel = (speed * 0.5, 0.0, 0.0)
    elif variant == "hit_wall":
        # Roll straight into the side wall, bounce off.
        ball_pos = (-0.6, 0.0, r_ball)
        ball_vel = (speed, 0.0, 0.0)
    elif variant == "bounce_inside":
        # Drop with velocity, bounces around inside.
        ball_pos = (0.0, 0.0, _CONTAINER_H + r_ball + 0.3)
        ball_vel = (0.0, 0.0, -speed * 0.5)
        ball_rest = rng.uniform(0.5, 0.9)
    else:  # miss
        # Aim off-center so the ball lands outside the opening.
        ball_pos = (rng.uniform(0.25, 0.35), rng.uniform(-0.1, 0.1), _CONTAINER_H + r_ball + 0.3)
        ball_vel = (0.0, 0.0, 0.0)

    objects.append(
        ball_spec(
            "ball",
            ball_pos,
            r_ball,
            0.1,
            make_material(ball_rest),
            velocity=ball_vel,
            color=(0.95, 0.2, 0.15),
        )
    )

    return build_episode(
        config,
        episode_id,
        template_path,
        objects,
        standard_cameras(_TABLE_CENTER),
    )
