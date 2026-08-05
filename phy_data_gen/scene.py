"""Build an episode scene as a USD override layer over the Cosmos template.

The original Cosmos template is never modified. Instead a new root layer
sublayers the template, deactivates the template's dynamic props and adds
references to the sampled MolmoSpaces assets under
``/World/GeneratedObjects``.

If a referenced asset already carries physics APIs, overrides are authored
on its existing rigid-body prim instead of creating a second, nested body.

This module only depends on ``pxr`` (USD), not on the Isaac Sim runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from phy_data_gen.config import SceneConfig, load_config
from phy_data_gen.episode import create_episode_spec
from phy_data_gen.schemas import EpisodeSpec, ObjectSpec

_GENERATED_ROOT = "/World/GeneratedObjects"


def xyzw_to_gf_quat(value: tuple[float, float, float, float]):
    """Convert an (x, y, z, w) tuple to a ``Gf.Quatd`` (real-first)."""

    from pxr import Gf

    x, y, z, w = value
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))


def _disable_template_dynamics(stage, scene: SceneConfig) -> None:
    """Deactivate the template's dynamic props via override prims."""

    from pxr import Usd, UsdPhysics

    prim_paths = {
        prim_path
        for prim_path in scene.dynamic_prims
        if stage.GetPrimAtPath(prim_path).IsValid()
    }
    world_prim = stage.GetPrimAtPath(scene.world_prim)
    if world_prim and world_prim.IsValid():
        for child in world_prim.GetChildren():
            name = child.GetName()
            seed_prefix = name.partition("_")[0]
            is_template_prop = name.startswith("Prop_") or (
                seed_prefix.startswith("S") and seed_prefix[1:].isdigit()
            )
            has_rigid_body = any(
                prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in Usd.PrimRange(child)
            )
            if is_template_prop or has_rigid_body:
                prim_paths.add(str(child.GetPath()))

    for prim_path in sorted(prim_paths):
        override = stage.OverridePrim(prim_path)
        override.SetActive(False)


def _set_object_transform(prim, object_spec: ObjectSpec) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    # Author every op at double precision so the Gf.Vec3d / Gf.Quatd values
    # below match the attribute type (the default precision is float).
    precision = UsdGeom.XformOp.PrecisionDouble

    translate_op = xform.AddTranslateOp(precision=precision)
    translate_op.Set(Gf.Vec3d(*object_spec.position))

    orient_op = xform.AddOrientOp(precision=precision)
    orient_op.Set(xyzw_to_gf_quat(object_spec.orientation_xyzw))

    scale_op = xform.AddScaleOp(precision=precision)
    scale_op.Set(Gf.Vec3d(object_spec.scale, object_spec.scale, object_spec.scale))


def _prims_with_api(root_prim, api) -> list:
    """Return prims in ``root_prim``'s subtree carrying ``api``."""

    from pxr import Usd

    return [prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(api)]


def _apply_physics(stage, prim, object_spec: ObjectSpec) -> None:
    """Override physics on the asset's single rigid body.

    MolmoSpaces assets normally place their rigid-body API below the default
    prim. Adding another rigid body to the reference container would create a
    nested rigid-body hierarchy and prevent Isaac Lab from initializing the
    object, so the existing body is reused instead.
    """

    from pxr import UsdPhysics, UsdShade

    rigid_prims = _prims_with_api(prim, UsdPhysics.RigidBodyAPI)
    if len(rigid_prims) > 1:
        paths = ", ".join(str(item.GetPath()) for item in rigid_prims)
        raise ValueError(
            f"Asset for {object_spec.object_id} contains multiple rigid bodies: {paths}"
        )

    collision_prims = _prims_with_api(prim, UsdPhysics.CollisionAPI)
    if not collision_prims:
        raise ValueError(
            f"Asset for {object_spec.object_id} contains no collision prims: "
            f"{object_spec.asset_path}"
        )

    rigid_prim = rigid_prims[0] if rigid_prims else prim
    if not rigid_prims:
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)

    mass_api = UsdPhysics.MassAPI.Apply(rigid_prim)
    mass_api.CreateMassAttr(object_spec.mass)

    # Author a physics material and bind it to the effective rigid body.
    material_path = f"{prim.GetPath()}/PhysicsMaterial"
    material_prim = stage.DefinePrim(material_path, "Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr(object_spec.material.static_friction)
    material_api.CreateDynamicFrictionAttr(object_spec.material.dynamic_friction)
    material_api.CreateRestitutionAttr(object_spec.material.restitution)

    binding_api = UsdShade.MaterialBindingAPI.Apply(rigid_prim)
    material = UsdShade.Material(material_prim)
    binding_api.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _add_object(stage, object_spec: ObjectSpec) -> str:
    prim_path = f"{_GENERATED_ROOT}/{object_spec.object_id}"
    prim = stage.DefinePrim(prim_path, "Xform")

    asset_path = Path(object_spec.asset_path)
    prim.GetReferences().AddReference(str(asset_path.resolve()))

    _set_object_transform(prim, object_spec)
    _apply_physics(stage, prim, object_spec)
    return prim_path


def build_scene(
    episode_spec: EpisodeSpec,
    scene: SceneConfig,
    output_path: Path,
) -> Path:
    """Create the episode scene USDA and return its path."""

    from pxr import Sdf, Usd

    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = Path(episode_spec.template_path).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")

    if output_path.exists():
        output_path.unlink()

    root_layer = Sdf.Layer.CreateNew(str(output_path))
    root_layer.subLayerPaths.append(str(template_path))

    stage = Usd.Stage.Open(root_layer)

    _disable_template_dynamics(stage, scene)

    stage.DefinePrim(_GENERATED_ROOT, "Xform")
    for object_spec in episode_spec.objects:
        _add_object(stage, object_spec)

    root_layer.Save()
    return output_path


def compile_scene(
    episode_spec: EpisodeSpec,
    scene: SceneConfig,
    output_root: Path,
) -> Path:
    """Build a scene inside the reference dataset layout."""

    output_path = output_root / "scene" / episode_spec.episode_id / f"{scene.name}.usda"
    return build_scene(episode_spec, scene, output_path)


@dataclass
class SceneOptions:
    """Generate an episode scene USDA override layer."""

    config: Path
    episode_id: str = "episode_000000"


def main() -> None:
    options = tyro.cli(SceneOptions)

    config = load_config(options.config)
    episode_spec = create_episode_spec(config, episode_id=options.episode_id)
    scene_path = compile_scene(episode_spec, config.scene, config.output_root)
    print(f"Wrote scene to {scene_path}")


if __name__ == "__main__":
    main()
