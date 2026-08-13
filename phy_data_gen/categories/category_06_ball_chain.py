"""Category 06 — ball-chain collisions.

A row (chain) of balls on the table, struck at one or both ends. Sub-variants:

* one_end — single ball rolls into one end of the chain.
* both_ends — two balls strike both ends simultaneously.
* single_ball_chain — the striker ball varies in mass relative to the chain.
* multi_ball_chain — 2-3 balls strike one end.
* equal_mass — all chain balls share one mass.
* unequal_mass — chain mass varies along the row (gradient).
* spaced_chain — gaps between chain balls (1.02-1.5 x diameter).
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
    "one_end",
    "both_ends",
    "single_ball_chain",
    "multi_ball_chain",
    "equal_mass",
    "unequal_mass",
    "spaced_chain",
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

    r = rng.uniform(0.035, 0.05)
    restitution = rng.uniform(0.3, 1.0)
    speed = rng.uniform(0.5, 8.0)
    n_chain = rng.randint(2, 8)

    chain_mass = rng.uniform(0.05, 0.5)
    striker_mass = rng.uniform(0.05, 0.5)

    # Chain along +X, centred at origin.
    spacing = 2.0 * r * (1.0 if variant != "spaced_chain" else rng.uniform(1.02, 1.5))
    balls = []
    for i in range(n_chain):
        mass = chain_mass
        if variant == "unequal_mass":
            # Gradient: each step multiplies by 1.3-1.8.
            mass *= rng.uniform(1.3, 1.8) ** i
        balls.append(
            ball_spec(
                f"chain_{i}",
                ((i - n_chain / 2.0) * spacing, 0.0, r),
                r,
                mass,
                make_material(restitution),
                color=(0.3, 0.6, 0.9),
            )
        )

    def striker(side: float) -> None:
        balls.append(
            ball_spec(
                "striker",
                (side * (n_chain / 2.0 * spacing + 2.2 * r), 0.0, r),
                r,
                striker_mass,
                make_material(restitution),
                velocity=(side * speed, 0.0, 0.0),
                color=(0.9, 0.2, 0.2),
            )
        )

    if variant == "both_ends":
        striker(-1.0)
        striker(1.0)
    elif variant == "multi_ball_chain":
        # 2-3 strikers hit one end.
        for k in range(rng.randint(2, 3)):
            balls.append(
                ball_spec(
                    f"striker_{k}",
                    (-(n_chain / 2.0 * spacing + 2.2 * r) - k * 2.2 * r, 0.0, r),
                    r,
                    striker_mass,
                    make_material(restitution),
                    velocity=(speed, 0.0, 0.0),
                    color=(0.9, 0.2, 0.2),
                )
            )
    elif variant == "single_ball_chain":
        # Striker mass varies widely (light/heavy striker).
        striker_mass = rng.choice([0.05, 0.5, 2.0, 5.0])
        striker(-1.0)
    else:
        striker(-1.0)

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        standard_cameras(_TABLE_CENTER),
    )
