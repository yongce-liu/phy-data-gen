"""Category 02 — two-ball no-collision motion.

Two balls move in the same scene but never touch. Sub-variants:

* parallel pass — side-by-side lanes, same direction, different or equal speed.
* cross pass — trajectories cross at an angle with perpendicular separation
  large enough to miss.
* same-direction different speeds — offset lanes so the faster ball does not
  catch up to touch.
* offset head-on pass — head-on but with an impact-parameter miss.
* different release times — same lane, but the second ball releases after the
  first has passed (guaranteed no contact).

The miss is guaranteed geometrically: the minimum inter-ball centre distance
over the episode stays above 0.98 x (r1+r2).
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
    "parallel_pass",
    "cross_pass",
    "same_direction",
    "offset_head_on",
    "different_release",
]


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def _sample_scenario(rng: random.Random, variant: str) -> dict:
    r1 = rng.uniform(0.03, 0.05)
    r2 = rng.uniform(0.03, 0.05)
    sum_r = r1 + r2
    return {
        "r1": r1,
        "r2": r2,
        "sum_r": sum_r,
        "v": rng.uniform(0.3, 5.0),
        "restitution": rng.uniform(0.2, 1.0),
    }


def _place_balls(scenario: dict, variant: str, rng: random.Random):
    r1 = scenario["r1"]
    r2 = scenario["r2"]
    sum_r = scenario["sum_r"]
    v = scenario["v"]
    miss = sum_r * rng.uniform(1.05, 1.6)  # guaranteed lateral miss

    if variant == "parallel_pass":
        # Two offset lanes, same direction.
        pos1 = (-miss / 2.0, -0.5, r1)
        pos2 = (miss / 2.0, 0.5, r2)
        vel1 = (0.0, v, 0.0)
        vel2 = (0.0, v * rng.uniform(0.5, 1.2), 0.0)
    elif variant == "cross_pass":
        # Cross at the origin; ball1 along +Y, ball2 along +X, large miss.
        angle = rng.uniform(math.radians(30), math.radians(90))
        pos1 = (0.0, -0.6, r1)
        vel1 = (0.0, v, 0.0)
        # Ensure miss: offset trajectories by > sum_r in the cross direction.
        miss_offset = miss * 1.2
        pos2 = (-0.6, miss_offset, r2)
        vel2 = (v, 0.0, 0.0)
    elif variant == "same_direction":
        # Same lane but one slightly ahead and offset laterally.
        pos1 = (0.0, -0.6, r1)
        pos2 = (miss / 2.0, -0.1, r2)
        vel1 = (0.0, v, 0.0)
        vel2 = (0.0, v * rng.uniform(1.05, 1.4), 0.0)  # ball2 faster but offset
    elif variant == "offset_head_on":
        # Head-on lanes offset so they miss.
        pos1 = (-miss / 2.0, -0.6, r1)
        pos2 = (miss / 2.0, 0.6, r2)
        vel1 = (0.0, v, 0.0)
        vel2 = (0.0, -v, 0.0)
    else:  # different_release
        # Same lane, second ball releases after the first passes the crossing
        # point. Realized by starting ball_b further back so it arrives late.
        pos1 = (0.0, -0.6, r1)
        delay = rng.uniform(0.4, 0.9)
        pos2 = (0.0, 0.6 + v * delay, r2)
        vel1 = (0.0, v, 0.0)
        vel2 = (0.0, -v, 0.0)

    return pos1, pos2, vel1, vel2


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)
    scenario = _sample_scenario(rng, variant)
    pos1, pos2, vel1, vel2 = _place_balls(scenario, variant, rng)

    balls = [
        ball_spec(
            "ball_a",
            pos1,
            scenario["r1"],
            0.1,
            make_material(scenario["restitution"]),
            velocity=vel1,
            color=(0.9, 0.2, 0.2),
        ),
        ball_spec(
            "ball_b",
            pos2,
            scenario["r2"],
            0.1,
            make_material(scenario["restitution"]),
            velocity=vel2,
            color=(0.2, 0.3, 0.9),
        ),
    ]

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        standard_cameras(_TABLE_CENTER),
    )
