"""Build an :class:`EpisodeSpec` from the asset registry and run config.

Sampling uses an explicit ``random.Random(seed)`` so the same seed always
yields the same episode. Initial positions use a fixed grid above the
template centre rather than fully random placement, to avoid initial
overlaps in the first version.

This module has no Isaac Sim runtime dependency; it only reads JSON/YAML.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import tyro

from phy_data_gen.config import RunConfig, load_config
from phy_data_gen.schemas import EpisodeSpec, ObjectSpec, PhysicsMaterialSpec

# Fixed grid slots (x, y, z) above the template centre, in metres.
_GRID_SLOTS: tuple[tuple[float, float, float], ...] = (
    (-0.4, 0.0, 1.5),
    (0.0, 0.0, 2.0),
    (0.4, 0.0, 2.5),
    (-0.4, 0.4, 3.0),
    (0.0, 0.4, 3.5),
    (0.4, 0.4, 4.0),
    (0.0, -0.4, 4.5),
)


def load_registry(registry_path: Path) -> list[dict]:
    """Load asset records from the registry JSON.

    Returns an empty list if the registry file does not exist yet, so the
    planner can still run in a bootstrap environment without assets.
    """

    if not registry_path.is_file():
        return []

    with registry_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return list(payload.get("assets", []))


def select_template_path(config: RunConfig) -> Path:
    """Select a scene template deterministically from ``template_root``."""

    from phy_data_gen.registry import find_usd_files

    templates = find_usd_files(config.template_root)
    if not templates:
        raise FileNotFoundError(
            f"No USD scene templates found under: {config.template_root}"
        )
    scene_rng = random.Random(config.seed)
    return scene_rng.choice(templates)


def _sample_orientation(rng: random.Random) -> tuple[float, float, float, float]:
    """Sample a uniformly random unit quaternion in (x, y, z, w) order."""

    # Marsaglia's method for a uniform random rotation.
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()

    from math import cos, pi, sin, sqrt

    r1 = sqrt(1.0 - u1)
    r2 = sqrt(u1)
    two_pi = 2.0 * pi

    x = r1 * sin(two_pi * u2)
    y = r1 * cos(two_pi * u2)
    z = r2 * sin(two_pi * u3)
    w = r2 * cos(two_pi * u3)
    return (x, y, z, w)


def _sample_material(rng: random.Random) -> PhysicsMaterialSpec:
    return PhysicsMaterialSpec(
        static_friction=rng.uniform(0.4, 0.9),
        dynamic_friction=rng.uniform(0.3, 0.8),
        restitution=rng.uniform(0.0, 0.4),
    )


def create_episode_spec(
    config: RunConfig,
    episode_id: str = "episode_000000",
) -> EpisodeSpec:
    """Sample an :class:`EpisodeSpec` deterministically from ``config``."""

    rng = random.Random(config.seed)
    registry = load_registry(config.registry_path)
    selected_template = select_template_path(config)

    num_objects = min(config.num_objects, len(_GRID_SLOTS))
    objects: list[ObjectSpec] = []

    for index in range(num_objects):
        if registry:
            asset = rng.choice(registry)
            asset_path = str(asset["usd_path"])
        else:
            # Bootstrap fallback: no assets scanned yet. Reference a
            # placeholder path so the spec is still well-formed and the
            # downstream scene builder can surface a clear error.
            asset_path = str(
                (config.asset_root / f"__missing_asset_{index}.usd").resolve()
            )

        position = _GRID_SLOTS[index]
        orientation = _sample_orientation(rng)

        objects.append(
            ObjectSpec(
                object_id=f"object_{index}",
                asset_path=asset_path,
                position=position,
                orientation_xyzw=orientation,
                scale=rng.uniform(0.8, 1.2),
                mass=rng.uniform(0.05, 0.5),
                material=_sample_material(rng),
            )
        )

    return EpisodeSpec(
        episode_id=episode_id,
        seed=config.seed,
        template_path=str(selected_template.resolve()),
        backend=config.backend,
        duration_seconds=config.simulation.duration_seconds,
        physics_dt=config.simulation.physics_dt,
        render_fps=config.simulation.render_fps,
        objects=objects,
    )


def save_episode_spec(spec: EpisodeSpec, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(spec.model_dump_json(indent=2))
        file.write("\n")


@dataclass
class EpisodeOptions:
    """Build an EpisodeSpec from the run config and registry."""

    config: Path
    output: Path
    episode_id: str = "episode_000000"


def main() -> None:
    options = tyro.cli(EpisodeOptions)

    config = load_config(options.config)
    spec = create_episode_spec(config, episode_id=options.episode_id)
    save_episode_spec(spec, options.output)
    print(f"Wrote EpisodeSpec with {len(spec.objects)} object(s) to {options.output}")


if __name__ == "__main__":
    main()
