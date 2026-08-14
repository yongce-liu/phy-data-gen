"""Category 04 — multi-ball collisions.

Several balls interact. Sub-variants (``seed % 6``):

* one_dimensional_chain — a row of balls, one rolls in from the end.
* random_2d — N balls jittered on the table, one fast ball plows through.
* random_3d — N balls in a volume above the table, released to drop.
* converge — N balls all aim at the table centre simultaneously.
* scatter — N balls start near the centre and shoot outward.
* continuous — a few high-speed balls in a walled box with repeated impacts.

All balls are recorded (N ≤ 40, so per-object views are fine).
"""

from __future__ import annotations

import math
import random

from phy_data_gen.categories.common import (
    ball_spec,
    build_episode,
    make_material,
    standard_cameras,
)
from phy_data_gen.config import RunConfig
from phy_data_gen.schemas import EpisodeSpec

_TABLE_CENTER = (0.0, 0.0, 0.0)
_VARIANTS = [
    "one_dimensional_chain",
    "random_2d",
    "random_3d",
    "converge",
    "scatter",
    "continuous",
]


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def _make_balls(
    rng: random.Random,
    variant: str,
    n: int,
    restitution: float,
) -> list:
    """Build ``n`` non-overlapping balls for the variant."""

    r = rng.uniform(0.03, 0.045)
    balls = []

    if variant == "one_dimensional_chain":
        spacing = 2.05 * r
        for i in range(n):
            balls.append(
                ball_spec(
                    f"ball_{i}",
                    (0.0, -i * spacing, r),
                    r,
                    0.1,
                    make_material(restitution),
                    color=(0.8, 0.1, 0.1),
                )
            )
        # ``ball_0`` is the incoming striker. The resting row runs down -Y
        # (ball_1..ball_{n-1}), so the striker must start *above* the row
        # (positive Y) with a visible approach and roll *toward* it (-Y).
        # The old code gave ball_0 a +Y velocity, sending it away from the
        # row so no chain reaction ever happened.
        approach = rng.uniform(0.30, 0.55)
        balls[0].position = (0.0, approach, r)
        balls[0].initial_linear_velocity = (0.0, -rng.uniform(2.0, 6.0), 0.0)
        return balls

    if variant == "random_2d":
        # Jittered lattice on the table.
        cols = max(2, int(math.ceil(math.sqrt(n))))
        spacing = 2.1 * r
        placed = 0
        for row in range(cols * 2):
            for col in range(cols):
                if placed >= n:
                    break
                x = (col - cols / 2.0) * spacing + rng.uniform(-r * 0.3, r * 0.3)
                y = (row - cols) * spacing + rng.uniform(-r * 0.3, r * 0.3)
                balls.append(
                    ball_spec(
                        f"ball_{placed}",
                        (x, y, r),
                        r,
                        0.1,
                        make_material(restitution),
                        color=(0.2, 0.6, 0.9),
                    )
                )
                placed += 1
            if placed >= n:
                break
        # One fast ball.
        balls[0].initial_linear_velocity = (rng.uniform(3.0, 8.0), 0.0, 0.0)
        return balls

    if variant == "random_3d":
        # Balls in a volume above the table, drop together.
        for i in range(n):
            balls.append(
                ball_spec(
                    f"ball_{i}",
                    (
                        rng.uniform(-0.6, 0.6),
                        rng.uniform(-0.6, 0.6),
                        rng.uniform(0.3, 0.8),
                    ),
                    r,
                    0.1,
                    make_material(restitution),
                    color=(0.6, 0.4, 0.2),
                )
            )
        return balls

    if variant == "converge":
        # Balls on a ring aiming at the centre.
        angle_step = 2.0 * math.pi / n
        speed = rng.uniform(1.0, 4.0)
        radius = rng.uniform(0.5, 0.9)
        for i in range(n):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            vx = -x / radius * speed
            vy = -y / radius * speed
            balls.append(
                ball_spec(
                    f"ball_{i}",
                    (x, y, r),
                    r,
                    0.1,
                    make_material(restitution),
                    velocity=(vx, vy, 0.0),
                    color=(0.9, 0.3, 0.3),
                )
            )
        return balls

    if variant == "scatter":
        # Balls near the centre shooting outward.
        angle_step = 2.0 * math.pi / n
        speed = rng.uniform(1.0, 5.0)
        for i in range(n):
            angle = i * angle_step + rng.uniform(-0.1, 0.1)
            balls.append(
                ball_spec(
                    f"ball_{i}",
                    (0.1 * math.cos(angle), 0.1 * math.sin(angle), r),
                    r,
                    0.1,
                    make_material(restitution),
                    velocity=(
                        speed * math.cos(angle),
                        speed * math.sin(angle),
                        0.0,
                    ),
                    color=(0.3, 0.8, 0.3),
                )
            )
        return balls

    # continuous — high-speed balls bouncing off the table bumpers.
    r = 0.04
    for i in range(n):
        balls.append(
            ball_spec(
                f"ball_{i}",
                (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), r),
                r,
                0.1,
                make_material(rng.uniform(0.3, 0.9)),
                velocity=(
                    rng.uniform(-8.0, 8.0),
                    rng.uniform(-8.0, 8.0),
                    0.0,
                ),
                color=(0.9, 0.7, 0.2),
            )
        )
    return balls


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)

    if variant == "one_dimensional_chain":
        n = rng.randint(3, 8)
    elif variant == "random_3d":
        n = rng.randint(5, 15)
    else:
        n = rng.randint(5, 20)

    restitution = rng.uniform(0.2, 1.0)
    balls = _make_balls(rng, variant, n, restitution)

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        standard_cameras(_TABLE_CENTER),
    )
