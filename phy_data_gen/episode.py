"""Build an :class:`EpisodeSpec` from the asset registry and run config.

Sampling uses an explicit ``random.Random(seed)`` so the same seed always
yields the same episode. Initial positions use a fixed grid above the
template centre rather than fully random placement, to avoid initial
overlaps in the first version.

This module has no Isaac Sim runtime dependency; it only reads JSON/YAML.
"""

from __future__ import annotations

import functools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import tyro

from phy_data_gen.config import RunConfig, load_config
from phy_data_gen.schemas import (
    AssetReplacementSpec,
    EpisodeSpec,
    ObjectSpec,
    PhysicsMaterialSpec,
)

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
    scene_seed = (
        config.template_seed if config.template_seed is not None else config.seed
    )
    scene_rng = random.Random(scene_seed)
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


def _replacement_object_id(prim_path: str, world_prim_path: str) -> str:
    relative = prim_path.removeprefix(f"{world_prim_path}/")
    return relative.replace("/", "__")


def _replacement_targets(
    template_path: Path,
    config: RunConfig,
) -> list[tuple[str, float, bool]]:
    """Return target paths, local dimensions, and missing-body flags.

    Results are cached by template path + seed to avoid re-parsing the same
    USD template across episodes that share it.
    """

    return _replacement_targets_impl(
        str(template_path.resolve()),
        config.scene.world_prim,
        config.scene.replace_initially_moving_objects,
        tuple(sorted(config.scene.dynamic_prims)),
    )


@functools.lru_cache(maxsize=None)
def _replacement_targets_impl(
    template_path_str: str,
    world_prim_path: str,
    replace_initially_moving: bool,
    dynamic_prims: tuple[str, ...],
) -> list[tuple[str, float, bool]]:
    from pathlib import Path

    template_path = Path(template_path_str)

    # Importing Usd registers the USDA file-format plugin used by Sdf.
    from pxr import Sdf, Usd  # noqa: F401

    layer = Sdf.Layer.FindOrOpen(str(template_path.resolve()))
    if layer is None:
        raise RuntimeError(f"Failed to open template: {template_path}")

    world = layer.GetPrimAtPath(world_prim_path)
    if world is None:
        raise RuntimeError(f"World prim not found: {world_prim_path}")

    roots = []
    if dynamic_prims:
        for prim_path in dynamic_prims:
            prim = layer.GetPrimAtPath(prim_path)
            if prim is None:
                raise RuntimeError(f"Replacement prim not found: {prim_path}")
            roots.append(prim)
    else:
        roots.append(world)

    def walk(prim_spec):
        yield prim_spec
        for child in prim_spec.nameChildren:
            yield from walk(child)

    def has_rigid_body_api(prim_spec) -> bool:
        if not prim_spec.HasInfo("apiSchemas"):
            return False
        schemas = prim_spec.GetInfo("apiSchemas").GetAddedOrExplicitItems()
        return "PhysicsRigidBodyAPI" in {str(schema) for schema in schemas}

    def primitive_dimension(prim_spec) -> float:
        dimensions = []
        for item in walk(prim_spec):
            properties = {prop.name: prop for prop in item.properties}
            if item.typeName == "Sphere":
                radius = properties.get("radius")
                dimensions.append(2.0 * float(radius.default if radius else 1.0))
            elif item.typeName == "Cube":
                size = properties.get("size")
                dimensions.append(float(size.default if size else 2.0))
            elif item.typeName == "Mesh":
                extent = properties.get("extent")
                if extent and extent.default and len(extent.default) == 2:
                    low, high = extent.default
                    dimensions.append(max(float(high[i] - low[i]) for i in range(3)))
        return max(dimensions, default=1.0)

    def is_initially_moving(prim_spec) -> bool:
        velocity = next(
            (prop for prop in prim_spec.properties if prop.name == "physics:velocity"),
            None,
        )
        if velocity is None or velocity.default is None:
            return False
        return math.sqrt(sum(float(value) ** 2 for value in velocity.default)) > 1e-6

    targets_by_path = {}
    if dynamic_prims:
        for root in roots:
            matches = [prim for prim in walk(root) if has_rigid_body_api(prim)]
            if len(matches) > 1:
                paths = ", ".join(str(prim.path) for prim in matches)
                raise RuntimeError(
                    f"Expected at most one rigid body below {root.path}, found "
                    f"{len(matches)}: {paths}"
                )
            target = matches[0] if matches else root
            targets_by_path[str(target.path)] = (target, not matches)
    else:
        for child in world.nameChildren:
            matches = [prim for prim in walk(child) if has_rigid_body_api(prim)]
            seed_prefix = child.name.partition("_")[0]
            is_template_prop = child.name.startswith("Prop_") or (
                seed_prefix.startswith("S") and seed_prefix[1:].isdigit()
            )
            if matches:
                for target in matches:
                    targets_by_path[str(target.path)] = (target, False)
            elif is_template_prop:
                # Some Cosmos props author their rigid-body API inside an
                # unavailable reference. Replace the outer slot and restore
                # a rigid body locally instead of resolving that reference.
                targets_by_path[str(child.path)] = (child, True)

    if not targets_by_path:
        raise RuntimeError(
            f"No replacement targets found below: {world_prim_path}"
        )

    targets = []
    for prim_path, (prim, create_rigid_body) in sorted(targets_by_path.items()):
        if (
            not replace_initially_moving
            and is_initially_moving(prim)
        ):
            continue
        targets.append((prim_path, primitive_dimension(prim), create_rigid_body))
    return targets


def _asset_info(
    asset_path: Path,
) -> tuple[tuple[float, float, float], str | None]:
    """Return bbox center and the relative rigid-body path for a local asset.

    Results are cached by asset path to avoid re-opening the same USD file
    across thousands of episodes.
    """

    return _asset_info_impl(str(asset_path.resolve()))


@functools.lru_cache(maxsize=None)
def _asset_info_impl(asset_path_str: str) -> tuple[tuple[float, float, float], str | None]:
    from pathlib import Path

    asset_path = Path(asset_path_str)
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Failed to open local asset: {asset_path}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError(f"Local asset has no default prim: {asset_path}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    center = box.GetMin() + box.GetSize() * 0.5
    rigid_prims = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(rigid_prims) > 1:
        paths = ", ".join(str(prim.GetPath()) for prim in rigid_prims)
        raise RuntimeError(
            f"Replacement asset contains multiple rigid bodies: {asset_path}: {paths}"
        )
    relative_rigid_path = None
    if rigid_prims:
        root_path = str(root.GetPath())
        relative_rigid_path = str(rigid_prims[0].GetPath()).removeprefix(root_path)
    return (
        (float(center[0]), float(center[1]), float(center[2])),
        relative_rigid_path,
    )


def _sample_replacements(
    config: RunConfig,
    template_path: Path,
    rng: random.Random,
) -> list[AssetReplacementSpec]:
    targets = _replacement_targets(template_path, config)
    if not targets:
        return []

    registry = load_registry(config.registry_path)
    if not registry:
        raise RuntimeError(f"Asset registry is empty or missing: {config.registry_path}")

    if len(targets) <= len(registry):
        assets = rng.sample(registry, len(targets))
    else:
        assets = [rng.choice(registry) for _ in targets]

    replacements = []
    for (target_path, target_dimension, create_rigid_body), asset in zip(
        targets,
        assets,
        strict=True,
    ):
        asset_path = Path(str(asset["usd_path"]))
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Registry asset is not available locally: {asset_path}. "
                "Rebuild the registry after downloading assets; generation never downloads."
            )
        asset_dimension = float(asset["max_dimension"])
        if asset_dimension <= 0.0:
            raise ValueError(f"Asset has invalid max_dimension: {asset_path}")
        scale = target_dimension / asset_dimension
        center, asset_rigid_body_path = _asset_info(asset_path)
        replacements.append(
            AssetReplacementSpec(
                object_id=_replacement_object_id(
                    target_path,
                    config.scene.world_prim,
                ),
                target_prim_path=target_path,
                asset_path=str(asset_path.resolve()),
                scale=scale,
                translation=tuple(-value * scale for value in center),
                create_rigid_body=create_rigid_body,
                asset_rigid_body_path=asset_rigid_body_path,
            )
        )
    return replacements


def create_episode_spec(
    config: RunConfig,
    episode_id: str = "episode_000000",
) -> EpisodeSpec:
    """Sample an :class:`EpisodeSpec` deterministically from ``config``."""

    rng = random.Random(config.seed)
    selected_template = select_template_path(config)

    objects: list[ObjectSpec] = []
    replacements: list[AssetReplacementSpec] = []
    if config.scene.object_mode == "generated_objects":
        registry = load_registry(config.registry_path)
        num_objects = min(config.num_objects, len(_GRID_SLOTS))

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
    elif config.scene.object_mode == "replace_assets":
        replacements = _sample_replacements(config, selected_template, rng)

    return EpisodeSpec(
        episode_id=episode_id,
        seed=config.seed,
        template_path=str(selected_template.resolve()),
        backend=config.backend,
        object_mode=config.scene.object_mode,
        duration_seconds=config.simulation.duration_seconds,
        physics_dt=config.simulation.physics_dt,
        render_fps=config.simulation.render_fps,
        objects=objects,
        replacements=replacements,
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
    object_count = len(spec.objects) + len(spec.replacements)
    print(f"Wrote EpisodeSpec with {object_count} object(s) to {options.output}")


if __name__ == "__main__":
    main()
