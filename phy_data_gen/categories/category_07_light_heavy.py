"""Category 07 — light vs heavy ball collisions.

Extreme mass ratios between two (or a small chain of) balls. Sub-variants:

* light_hits_heavy — light ball strikes a heavy one (heavy barely moves).
* heavy_hits_light — heavy ball strikes a light one (light shoots away).
* light_fast_rebound — elastic collision, light ball rebounds off heavy.
* light_accelerated — moving heavy ball transfers energy to light ball.
* mass_gradient_chain — a 3-6 ball chain with mass ratio per step.
"""

from __future__ import annotations

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
    "light_hits_heavy",
    "heavy_hits_light",
    "light_fast_rebound",
    "light_accelerated",
    "mass_gradient_chain",
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

    r = rng.uniform(0.04, 0.05)
    restitution = rng.choice([0.0, 0.5, 1.0])
    ratio = rng.uniform(5.0, 50.0)
    speed = rng.uniform(0.5, 8.0)

    m_light = 0.05
    m_heavy = m_light * ratio

    balls = []

    if variant in {"light_hits_heavy", "light_fast_rebound"}:
        balls.append(
            ball_spec(
                "light",
                (-0.6, 0.0, r),
                r,
                m_light,
                make_material(restitution),
                velocity=(speed, 0.0, 0.0),
                color=(0.9, 0.2, 0.2),
            )
        )
        balls.append(
            ball_spec(
                "heavy",
                (0.4, 0.0, r),
                r,
                m_heavy,
                make_material(restitution),
                color=(0.2, 0.4, 0.9),
            )
        )
    elif variant == "heavy_hits_light":
        balls.append(
            ball_spec(
                "heavy",
                (-0.6, 0.0, r),
                r,
                m_heavy,
                make_material(restitution),
                velocity=(speed, 0.0, 0.0),
                color=(0.2, 0.4, 0.9),
            )
        )
        balls.append(
            ball_spec(
                "light",
                (0.4, 0.0, r),
                r,
                m_light,
                make_material(restitution),
                color=(0.9, 0.2, 0.2),
            )
        )
    elif variant == "light_accelerated":
        # Both moving toward each other; light is accelerated by heavy.
        balls.append(
            ball_spec(
                "heavy",
                (-0.6, 0.0, r),
                r,
                m_heavy,
                make_material(restitution),
                velocity=(speed, 0.0, 0.0),
                color=(0.2, 0.4, 0.9),
            )
        )
        balls.append(
            ball_spec(
                "light",
                (0.4, 0.0, r),
                r,
                m_light,
                make_material(restitution),
                velocity=(-speed * 0.3, 0.0, 0.0),
                color=(0.9, 0.2, 0.2),
            )
        )
    else:  # mass_gradient_chain
        n = rng.randint(3, 6)
        spacing = 2.05 * r
        gradient = rng.uniform(2.0, 5.0)
        for i in range(n):
            balls.append(
                ball_spec(
                    f"ball_{i}",
                    ((i - n / 2.0) * spacing, 0.0, r),
                    r,
                    0.05 * gradient ** i,
                    make_material(restitution),
                    color=(0.3 + 0.6 * i / n, 0.2, 0.9),
                )
            )
        balls.append(
            ball_spec(
                "striker",
                (-n / 2.0 * spacing - 2.2 * r, 0.0, r),
                r,
                0.05,
                make_material(restitution),
                velocity=(speed, 0.0, 0.0),
                color=(0.95, 0.1, 0.1),
            )
        )

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        standard_cameras(_TABLE_CENTER),
    )
