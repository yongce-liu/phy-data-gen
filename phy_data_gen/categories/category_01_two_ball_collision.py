"""Category 01 — two-ball collision.

Sub-variants covered over 1000 seeds:

* collision geometry: head-on (impact parameter b=0), oblique (0<b<r1+r2),
  eccentric (b near r1+r2), head-on-towards (both balls moving opposite),
  rear-end catch-up (one moving, one ahead).
* elasticity: elastic (e=1), inelastic (e in {0.3, 0.7}), completely
  inelastic (e=0).
* masses: equal mass, unequal mass ratios {2, 5, 10}.
* initial motion: one ball stationary, both balls moving.
* spin: one ball may carry initial angular velocity (rad/s).

The billiards table is used as a static backdrop: its template balls and cue
ball are deactivated by the procedural scene builder while the table, bumpers
and cameras remain.
"""

from __future__ import annotations

import random

from phy_data_gen.categories.common import (
    ball_spec,
    build_episode,
    jittered_cameras,
    make_material,
)
from phy_data_gen.config import RunConfig
from phy_data_gen.schemas import EpisodeSpec

# Billiards table playing field, roughly in metres (bumpers near x=±1.19,
# y=±2.08). Episodes keep the action centred so balls stay on the table.
_TABLE_CENTER = (0.0, 0.0, 0.0)

_VARIANTS = [
    "head_on",
    "oblique",
    "eccentric",
    "head_on_towards",
    "rear_end",
    "both_moving",
]


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def _sample_scenario(rng: random.Random, variant: str) -> dict:
    """Return a parametrized two-ball scenario dict."""

    r1 = rng.uniform(0.03, 0.05)
    r2 = rng.uniform(0.03, 0.05)
    sum_r = r1 + r2
    gap = sum_r * rng.uniform(0.15, 0.5)  # initial separation along travel axis

    # Impact parameter in units of sum_r.
    if variant == "head_on":
        b = 0.0
    elif variant == "oblique":
        b = rng.uniform(0.0, 0.6) * sum_r
    elif variant == "eccentric":
        b = rng.uniform(0.6, 0.95) * sum_r
    else:
        b = 0.0

    speed = rng.uniform(0.5, 8.0)

    # Mass ratio and motion mode.
    ratio = rng.choice([1.0, 2.0, 5.0, 10.0])
    base_mass = rng.uniform(0.08, 0.5)
    if ratio == 1.0:
        m1 = m2 = base_mass
    else:
        m1 = base_mass
        m2 = base_mass * ratio

    # Elasticity.
    e = rng.choice([0.0, 0.3, 0.7, 1.0])
    restitution = e if variant != "head_on" else e

    spin1 = rng.uniform(-50.0, 50.0) if rng.random() < 0.5 else 0.0
    spin2 = 0.0

    return {
        "r1": r1,
        "r2": r2,
        "b": b,
        "gap": gap,
        "speed": speed,
        "m1": m1,
        "m2": m2,
        "restitution": restitution,
        "spin1": spin1,
        "spin2": spin2,
    }


def _place_balls(scenario: dict, variant: str, rng: random.Random):
    """Return (pos1, pos2, vel1, vel2, spin1) for the scenario.

    The collision happens at the origin after a visible approach: ball_a
    starts far away (0.5-1.2 m) and travels in, so the video shows the run-up
    before contact. ball_b sits at (b, 0, r2) — the impact-parameter offset —
    so the collision point is the origin.
    """

    r1 = scenario["r1"]
    r2 = scenario["r2"]
    b = scenario["b"]
    speed = scenario["speed"]
    s1 = scenario["spin1"]
    approach = rng.uniform(0.5, 1.2)  # visible run-up distance (m)

    # Ball 1 travels along +Y toward ball 2 at the origin (offset by b).
    pos2 = (b, 0.0, r2)
    pos1 = (0.0, -approach, r1)
    vel1 = (0.0, speed, 0.0)
    vel2 = (0.0, 0.0, 0.0)

    if variant == "head_on_towards":
        # Both moving head-on, collide at the origin.
        pos2 = (0.0, approach * 0.4, r2)
        vel2 = (0.0, -speed * 0.5, 0.0)
        vel1 = (0.0, speed, 0.0)
    elif variant == "rear_end":
        # Catch-up: ball 2 ahead moving slower, ball 1 behind moving faster.
        pos2 = (b, approach * 0.3, r2)
        vel2 = (0.0, speed * rng.uniform(0.2, 0.6), 0.0)
        vel1 = (0.0, speed, 0.0)
    elif variant == "both_moving":
        # Ball 1 moves along +X, ball 2 along -X, collide at the origin.
        pos1 = (-approach, 0.0, r1)
        pos2 = (approach * 0.3, 0.0, r2)
        vel1 = (speed, 0.0, 0.0)
        vel2 = (-speed * 0.5, 0.0, 0.0)

    return pos1, pos2, vel1, vel2, s1


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    """Deterministically sample one two-ball collision episode."""

    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)
    scenario = _sample_scenario(rng, variant)
    pos1, pos2, vel1, vel2, s1 = _place_balls(scenario, variant, rng)

    # Randomize ball colours each episode for visual variety.
    color_a = _random_color(rng)
    color_b = _random_color(rng, exclude=color_a)

    balls = [
        ball_spec(
            "ball_a",
            pos1,
            scenario["r1"],
            scenario["m1"],
            make_material(scenario["restitution"]),
            velocity=vel1,
            angular_velocity=(0.0, 0.0, s1),
            color=color_a,
        ),
        ball_spec(
            "ball_b",
            pos2,
            scenario["r2"],
            scenario["m2"],
            make_material(scenario["restitution"]),
            velocity=vel2,
            color=color_b,
        ),
    ]

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        jittered_cameras(_TABLE_CENTER, rng),
    )


def _random_color(rng, exclude=None):
    """Return a vivid, distinct RGB colour tuple."""

    while True:
        # Avoid muddy dark colours; bias to bright hues.
        color = (
            rng.uniform(0.2, 1.0),
            rng.uniform(0.2, 1.0),
            rng.uniform(0.2, 1.0),
        )
        if exclude is None:
            return color
        # Keep the two balls visually distinct.
        if sum(abs(a - b) for a, b in zip(color, exclude)) > 0.8:
            return color
