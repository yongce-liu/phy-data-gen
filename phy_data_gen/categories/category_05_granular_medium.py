"""Category 05 — ball into a granular medium.

A heavy ball falls into a tray of many small spheres (sand grains). The grains
are placed on a jittered hexagonal-close-packed lattice as a truncated cone so
there are no initial overlaps, and they are simulated but not state-recorded
(``record=False``) to avoid per-grain tensor views.

Sub-variants:

* fall_into_pile — drop from 0.2-0.8 m onto the pile.
* surface_bounce — low-speed, bouncy ball bounces off the pile surface.
* shallow_embed — moderate speed, ball settles into the top of the pile.
* fully_buried — high speed / large ball, ball ends buried.
* impact_crater — high speed creates a crater (visible via grain displacements).
* particle_splash — very high speed scatters grains outward.
"""

from __future__ import annotations

import math
import random

from phy_data_gen.categories.common import (
    ball_spec,
    build_episode,
    make_material,
)
from phy_data_gen.config import ProceduralConfig, RunConfig, SceneConfig
from phy_data_gen.schemas import CameraSpec, EpisodeSpec

_TRAY_SIZE = 0.5
_TRAY_HEIGHT = 0.35
_TRAY_CENTER = (0.0, 0.0, 0.0)
_VARIANTS = [
    "fall_into_pile",
    "surface_bounce",
    "shallow_embed",
    "fully_buried",
    "impact_crater",
    "particle_splash",
]


def _variant_for(seed: int) -> str:
    return _VARIANTS[seed % len(_VARIANTS)]


def _camera_specs() -> dict[str, CameraSpec]:
    from phy_data_gen.categories.common import camera_spec

    center = (0.0, 0.0, 0.2)
    return {
        "Side": camera_spec("Side", (1.6, 0.0, 0.5), center),
        "Top": camera_spec("Top", (0.0, 0.0, 1.8), center),
    }


def _pile_grains(
    rng: random.Random,
    grain_r: float,
    target_count: int,
) -> list:
    """Place grains on a jittered HCP lattice as a truncated cone inside the tray.

    Centre spacing 2.2 x grain_r avoids initial overlaps.
    """

    spacing = 2.2 * grain_r
    base_radius = 0.21
    height = 0.18
    layer_gap = spacing * math.sqrt(2.0 / 3.0)
    grains = []
    count = 0
    row = 0
    while count < target_count:
        z = row * layer_gap
        radius_at_z = base_radius * (1.0 - z / (height + layer_gap))
        if radius_at_z <= spacing:
            break
        rows = max(1, int(radius_at_z / spacing))
        for i in range(rows * 2):
            for j in range(rows * 2):
                x = (i - rows) * spacing * 0.866 + (0.5 if row % 2 else 0.0)
                y = (j - rows) * spacing
                if math.hypot(x, y) > radius_at_z - spacing * 0.5:
                    continue
                jx = rng.uniform(-0.1, 0.1) * spacing
                jy = rng.uniform(-0.1, 0.1) * spacing
                grains.append(
                    ball_spec(
                        f"grain_{count}",
                        (x + jx, y + jy, z + grain_r),
                        grain_r,
                        0.0005,
                        make_material(0.1, static_friction=0.8, dynamic_friction=0.6),
                        color=(0.85, 0.75, 0.5),
                        record=False,
                    )
                )
                count += 1
                if count >= target_count:
                    break
            if count >= target_count:
                break
        row += 1
    return grains


def create_episode_spec(
    config: RunConfig,
    episode_id: str,
    template_path: str | None,
) -> EpisodeSpec:
    rng = random.Random(config.seed)
    variant = _variant_for(config.seed)

    grain_r = rng.uniform(0.012, 0.02)
    target = rng.randint(300, 600)
    grains = _pile_grains(rng, grain_r, target)

    ball_r = rng.uniform(0.03, 0.05)
    ball_mass = rng.uniform(0.3, 1.0)
    ball_rest = rng.choice([0.1, 0.5, 0.8])

    if variant == "fall_into_pile":
        drop_h = rng.uniform(0.2, 0.8)
        ball_vel = (0.0, 0.0, 0.0)
    elif variant == "surface_bounce":
        drop_h = rng.uniform(0.1, 0.3)
        ball_rest = rng.uniform(0.7, 0.95)
        ball_vel = (0.0, 0.0, 0.0)
    elif variant == "shallow_embed":
        drop_h = rng.uniform(0.3, 0.5)
        ball_vel = (0.0, 0.0, -rng.uniform(1.0, 2.0))
    elif variant == "fully_buried":
        ball_r = rng.uniform(0.04, 0.05)
        ball_mass = rng.uniform(0.8, 2.0)
        ball_vel = (0.0, 0.0, -rng.uniform(3.0, 6.0))
        drop_h = rng.uniform(0.5, 0.8)
    elif variant == "impact_crater":
        ball_vel = (0.0, 0.0, -rng.uniform(4.0, 7.0))
        ball_mass = rng.uniform(0.5, 1.5)
        drop_h = rng.uniform(0.4, 0.7)
    else:  # particle_splash
        ball_vel = (0.0, 0.0, -rng.uniform(6.0, 9.0))
        ball_mass = rng.uniform(0.5, 1.5)
        drop_h = rng.uniform(0.5, 0.8)

    # Ball placed above the pile, optionally with a slight lateral offset.
    offset_x = rng.uniform(-0.05, 0.05) if variant not in {"surface_bounce", "fall_into_pile"} else 0.0
    ball_pos = (offset_x, 0.0, _TRAY_HEIGHT + ball_r + drop_h)

    balls = [
        ball_spec(
            "ball",
            ball_pos,
            ball_r,
            ball_mass,
            make_material(ball_rest),
            velocity=ball_vel,
            color=(0.95, 0.2, 0.15),
        )
    ]
    balls.extend(grains)

    # Scene carries the sand tray.
    scene = SceneConfig(
        **{
            **config.scene.__dict__,
            "procedural": ProceduralConfig(
                build_ground=True,
                ground_size=2.0,
                sand_tray=True,
            ),
        }
    )
    config = config.__class__(**{**config.__dict__, "scene": scene})

    return build_episode(
        config,
        episode_id,
        template_path,
        balls,
        _camera_specs(),
    )
