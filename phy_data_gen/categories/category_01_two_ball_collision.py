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

    # Impact parameter in units of sum_r.
    if variant == "head_on":
        b = 0.0
    elif variant == "oblique":
        b = rng.uniform(0.0, 0.6) * sum_r
    elif variant == "eccentric":
        b = rng.uniform(0.6, 0.95) * sum_r
    else:
        b = 0.0

    # How long the approach (pre-collision motion) should stay on screen.
    # 0.9-1.8 s of run-up at 30 fps is 27-54 clearly readable frames of
    # ball-motion before contact — the phase the user asked to see.
    approach_time = rng.uniform(0.9, 1.8)

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

    spin1 = rng.uniform(-50.0, 50.0) if rng.random() < 0.5 else 0.0

    return {
        "r1": r1,
        "r2": r2,
        "b": b,
        "approach_time": approach_time,
        "m1": m1,
        "m2": m2,
        "restitution": e,
        "spin1": spin1,
        "spin2": 0.0,
    }


def _place_balls(scenario: dict, variant: str, rng: random.Random):
    """Return (pos1, pos2, vel1, vel2, spin1) for the scenario.

    The collision happens at the origin.  Instead of sampling an arbitrary
    speed and letting the balls reach each other in a frame or two, the
    approach distance and the target run-up time are sampled directly, and
    the speed is *derived* from them — so every episode shows a clear
    pre-contact motion phase of roughly ``approach_time`` seconds.

    Positions are kept inside a ~0.9 m radius of the table centre: Cosmos
    billiards tables differ in size (the smallest sampled x half-extent is
    ~1.28 m) and the bumpers sit on top, so a ball spawned beyond ~1.0 m can
    start over the rail or fall through a thin edge.  The collision point is
    still the origin.
    """

    r1 = scenario["r1"]
    r2 = scenario["r2"]
    b = scenario["b"]
    s1 = scenario["spin1"]
    t_approach = scenario["approach_time"]
    # Start positions stay within this radius of the origin so every ball is
    # solidly on the playing surface regardless of the template's table size.
    max_radius = 0.9

    if variant in ("head_on", "oblique", "eccentric"):
        # Ball a travels along +Y from y=-D into ball b parked at the
        # impact-parameter offset; contact at the origin.
        D = rng.uniform(0.5, max_radius)
        pos1 = (0.0, -D, r1)
        pos2 = (b, 0.0, r2)
        sep = D - (r1 + r2)
        rel = max(sep / t_approach, 0.3)
        vel1 = (0.0, rel, 0.0)
        vel2 = (0.0, 0.0, 0.0)
    elif variant == "head_on_towards":
        # Both balls moving head-on, meet at the origin.
        D = rng.uniform(0.5, max_radius)
        pos1 = (0.0, -D, r1)
        pos2 = (0.0, D * 0.4, r2)
        sep = (D + D * 0.4) - (r1 + r2)
        rel = max(sep / t_approach, 0.3)
        vel1 = (0.0, rel * 0.67, 0.0)
        vel2 = (0.0, -rel * 0.33, 0.0)
    elif variant == "rear_end":
        # Catch-up: ball 2 ahead moving slower, ball 1 behind moving faster.
        D = rng.uniform(0.5, max_radius)
        pos1 = (0.0, -D, r1)
        pos2 = (b, D * 0.4, r2)
        sep = (D + D * 0.4) - (r1 + r2)
        # The chaser catches up at (v1 - v2) = rel; sample the lead speed as
        # a fraction of rel so the full approach still takes t_approach.
        rel = max(sep / t_approach, 0.3)
        lead_frac = rng.uniform(0.25, 0.5)
        lead_speed = rel * lead_frac / (1.0 - lead_frac)
        vel2 = (0.0, lead_speed, 0.0)
        vel1 = (0.0, rel + lead_speed, 0.0)
    elif variant == "both_moving":
        # Ball 1 moves along +X, ball 2 along -X, collide at the origin.
        D = rng.uniform(0.5, max_radius)
        pos1 = (-D, 0.0, r1)
        pos2 = (D * 0.4, 0.0, r2)
        sep = (D + D * 0.4) - (r1 + r2)
        rel = max(sep / t_approach, 0.3)
        vel1 = (rel * 0.67, 0.0, 0.0)
        vel2 = (-rel * 0.33, 0.0, 0.0)
    else:
        raise ValueError(f"unknown variant: {variant}")

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

    meta = {
        "variant": variant,
        "restitution": scenario["restitution"],
        "mass_ratio": scenario["m2"] / scenario["m1"],
        "impact_parameter_m": scenario["b"],
        "approach_time_s": scenario["approach_time"],
        "spin_rad_s": s1,
        "initial_positions": [list(pos1), list(pos2)],
        "initial_velocities": [list(vel1), list(vel2)],
    }

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        jittered_cameras(_TABLE_CENTER, rng),
        metadata=meta,
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
