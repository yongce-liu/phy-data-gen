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
from phy_data_gen.schemas import (
    AssetReplacementSpec,
    CameraSpec,
    EpisodeSpec,
    ObjectSpec,
)

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


def _bind_physics_material(stage, prim, object_spec: ObjectSpec) -> None:
    """Author a physics material below ``prim`` and bind it to ``prim``."""

    from pxr import UsdPhysics, UsdShade

    material_path = f"{prim.GetPath()}/PhysicsMaterial"
    material_prim = stage.DefinePrim(material_path, "Material")
    material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    material_api.CreateStaticFrictionAttr(object_spec.material.static_friction)
    material_api.CreateDynamicFrictionAttr(object_spec.material.dynamic_friction)
    material_api.CreateRestitutionAttr(object_spec.material.restitution)

    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    material = UsdShade.Material(material_prim)
    binding_api.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _bind_visual_color(stage, geometry_prim, color: tuple[float, float, float]) -> None:
    """Bind a simple USDPreviewSurface material with ``color`` to a gprim."""

    from pxr import Sdf, UsdShade

    prim = geometry_prim.GetPrim()
    material_path = f"{str(prim.GetPath())}/VisualMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = f"{material_path}/PreviewSurface"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        (float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="all",
    )


def _apply_rigid_body_options(stage, prim, object_spec: ObjectSpec) -> None:
    """Author per-body CCD, mass, velocity and angular velocity on a rigid body.

    Velocity is authored on the prim carrying ``PhysicsRigidBodyAPI`` exactly
    like the billiards ``CueBall`` template (physics:velocity in m/s,
    physics:angularVelocity in rad/s). PhysX picks these up when loading
    physics from USD during ``SimulationContext.reset()``.
    """

    from pxr import Sdf, UsdPhysics

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateVelocityAttr().Set(object_spec.initial_linear_velocity)
    rigid_api.CreateAngularVelocityAttr().Set(object_spec.initial_angular_velocity)

    # The PhysxRigidBodyAPI schema class lives in the PhysX plugin, which is
    # unavailable before the app launches. Author the well-known attribute
    # names directly so scene building stays runtime-independent.
    ccd = prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool)
    ccd.Set(True)
    max_depenetration = prim.CreateAttribute(
        "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float
    )
    max_depenetration.Set(100000.0)
    max_linear = prim.CreateAttribute(
        "physxRigidBody:maxLinearVelocity", Sdf.ValueTypeNames.Float
    )
    max_linear.Set(1000.0)

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(object_spec.mass)


def _apply_physics(stage, prim, object_spec: ObjectSpec) -> None:
    """Override physics on the asset's single rigid body.

    MolmoSpaces assets normally place their rigid-body API below the default
    prim. Adding another rigid body to the reference container would create a
    nested rigid-body hierarchy and prevent Isaac Lab from initializing the
    object, so the existing body is reused instead.
    """

    from pxr import UsdPhysics

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

    if not object_spec.dynamic:
        # Static object: keep collision but drop any rigid body so PhysX
        # treats it as a fixed obstacle.
        for rigid in rigid_prims:
            rigid.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if rigid.HasAPI(UsdPhysics.MassAPI):
                rigid.RemoveAPI(UsdPhysics.MassAPI)
        _bind_physics_material(stage, prim, object_spec)
        return

    rigid_prim = rigid_prims[0] if rigid_prims else prim
    if not rigid_prims:
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)

    _apply_rigid_body_options(stage, rigid_prim, object_spec)
    _bind_physics_material(stage, rigid_prim, object_spec)


def _add_primitive_object(
    stage,
    object_spec: ObjectSpec,
    scene: SceneConfig,
    prim_path: str,
) -> str:
    """Add a procedurally defined sphere or box with rigid-body physics.

    Exactly one rigid body is authored below the object root (on the Xform)
    so ``_find_generated_rigid_body_paths`` in simulation.py resolves it the
    same way as a referenced asset.
    """

    from pxr import Sdf, UsdGeom, UsdPhysics

    prim = stage.DefinePrim(prim_path, "Xform")
    _set_object_transform(prim, object_spec)

    radius = object_spec.radius if object_spec.radius is not None else 0.05
    geometry_path = f"{prim_path}/Geometry"
    if object_spec.kind == "sphere":
        geometry = UsdGeom.Sphere.Define(stage, geometry_path)
        geometry.CreateRadiusAttr(radius)
    else:
        geometry = UsdGeom.Cube.Define(stage, geometry_path)
        if object_spec.half_extents is not None:
            # Per-axis box: unit cube scaled by 2*half_extents.
            geometry.CreateSizeAttr(1.0)
            from pxr import Gf

            ext = object_spec.half_extents
            geo_xform = UsdGeom.Xformable(geometry.GetPrim())
            geo_xform.AddScaleOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            ).Set(Gf.Vec3d(2.0 * ext[0], 2.0 * ext[1], 2.0 * ext[2]))
        else:
            geometry.CreateSizeAttr(2.0 * radius)

    collision_api = UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
    collision_api.CreateSimulationOwnerRel().SetTargets(
        [Sdf.Path(scene.physics_scene_prim)]
    )

    if object_spec.dynamic:
        _apply_rigid_body_options(stage, prim, object_spec)
    _bind_physics_material(stage, prim, object_spec)
    _bind_visual_color(stage, geometry.GetPrim(), object_spec.color)
    return prim_path


def _object_prim_path(object_spec: ObjectSpec) -> str:
    """Resolve the placement path for a generated object.

    ``record=False`` objects (sand grains, static container walls) are placed
    below ``/World/GeneratedObjects/Bulk`` so the per-object discovery in
    simulation.py can skip them while PhysX still simulates them.
    """

    if object_spec.record:
        return f"{_GENERATED_ROOT}/{object_spec.object_id}"
    return f"{_GENERATED_ROOT}/Bulk/{object_spec.object_id}"


def _add_object(stage, object_spec: ObjectSpec, scene: SceneConfig) -> str:
    prim_path = _object_prim_path(object_spec)
    if object_spec.kind != "asset":
        return _add_primitive_object(stage, object_spec, scene, prim_path)
    prim = stage.DefinePrim(prim_path, "Xform")

    asset_path = Path(object_spec.asset_path)
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Asset for {object_spec.object_id} is not available locally: "
            f"{asset_path}"
        )
    prim.GetReferences().AddReference(str(asset_path.resolve()))

    _set_object_transform(prim, object_spec)
    _apply_physics(stage, prim, object_spec)
    return prim_path


def _replace_asset(
    stage,
    replacement: AssetReplacementSpec,
    scene: SceneConfig,
) -> str:
    """Replace geometry below a template rigid body while preserving its state."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    target = stage.GetPrimAtPath(replacement.target_prim_path)
    if not target or not target.IsValid():
        raise RuntimeError(
            f"Replacement target not found: {replacement.target_prim_path}"
        )
    had_template_rigid_body = target.HasAPI(UsdPhysics.RigidBodyAPI)
    if not replacement.create_rigid_body and not had_template_rigid_body:
        raise RuntimeError(
            f"Replacement target is not a rigid body: {replacement.target_prim_path}"
        )

    template_rigid_api = UsdPhysics.RigidBodyAPI(target)
    linear_velocity = (
        template_rigid_api.GetVelocityAttr().Get()
        if had_template_rigid_body
        else None
    )
    angular_velocity = (
        template_rigid_api.GetAngularVelocityAttr().Get()
        if had_template_rigid_body
        else None
    )

    asset_path = Path(replacement.asset_path)
    if not asset_path.is_file():
        raise FileNotFoundError(
            f"Replacement asset is not available locally: {asset_path}"
        )

    for child in target.GetChildren():
        override = stage.OverridePrim(str(child.GetPath()))
        override.GetReferences().SetReferences([])
        override.SetActive(False)

    container_path = f"{replacement.target_prim_path}/ReplacementAsset"
    container = stage.DefinePrim(container_path, "Xform")
    xform = UsdGeom.Xformable(container)
    precision = UsdGeom.XformOp.PrecisionDouble
    xform.AddTranslateOp(precision=precision).Set(Gf.Vec3d(*replacement.translation))
    xform.AddScaleOp(precision=precision).Set(
        Gf.Vec3d(replacement.scale, replacement.scale, replacement.scale)
    )

    asset_prim = stage.DefinePrim(f"{container_path}/Asset", "Xform")
    asset_prim.GetReferences().AddReference(str(asset_path.resolve()))

    if replacement.asset_rigid_body_path:
        rigid_path = f"{asset_prim.GetPath()}{replacement.asset_rigid_body_path}"
        rigid_prim = stage.GetPrimAtPath(rigid_path)
        if not rigid_prim or not rigid_prim.IsValid():
            raise RuntimeError(f"Referenced asset rigid body not found: {rigid_path}")
    else:
        rigid_prim = asset_prim
        UsdPhysics.RigidBodyAPI.Apply(rigid_prim)

    if had_template_rigid_body:
        target.RemoveAPI(UsdPhysics.RigidBodyAPI)
    if target.HasAPI(UsdPhysics.CollisionAPI):
        target.RemoveAPI(UsdPhysics.CollisionAPI)
    if target.HasAPI(UsdPhysics.MeshCollisionAPI):
        target.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    if target.HasAPI(UsdPhysics.MassAPI):
        target.RemoveAPI(UsdPhysics.MassAPI)
    UsdShade.MaterialBindingAPI(target).UnbindAllBindings()

    replacement_rigid_api = UsdPhysics.RigidBodyAPI.Apply(rigid_prim)
    if linear_velocity is not None:
        replacement_rigid_api.CreateVelocityAttr().Set(linear_velocity)
    if angular_velocity is not None:
        replacement_rigid_api.CreateAngularVelocityAttr().Set(angular_velocity)

    collision_prims = [
        prim
        for prim in Usd.PrimRange(asset_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not collision_prims:
        raise RuntimeError(f"Replacement asset has no collision prims: {asset_path}")
    for prim in collision_prims:
        UsdPhysics.CollisionAPI(prim).CreateSimulationOwnerRel().SetTargets(
            [Sdf.Path(scene.physics_scene_prim)]
        )

    remaining_rigid_prims = [
        prim
        for prim in Usd.PrimRange(target)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if remaining_rigid_prims != [rigid_prim]:
        paths = ", ".join(str(prim.GetPath()) for prim in remaining_rigid_prims)
        raise RuntimeError(
            f"Expected one replacement rigid body below "
            f"{replacement.target_prim_path}, found: {paths or 'none'}"
        )
    return str(rigid_prim.GetPath())


def _repair_missing_template_primitives(stage, scene: SceneConfig) -> list[str]:
    """Replace unresolved Cosmos sphere/prism references with USD primitives."""

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    world_prim = stage.GetPrimAtPath(scene.world_prim)
    if not world_prim or not world_prim.IsValid():
        raise RuntimeError(f"World prim not found: {scene.world_prim}")

    repairs: list[tuple[str, str]] = []
    for prim in Usd.PrimRange(world_prim):
        if prim.GetName() not in {"sphere", "prism"}:
            continue
        has_geometry = any(
            descendant.IsA(UsdGeom.Gprim) for descendant in Usd.PrimRange(prim)
        )
        if not has_geometry:
            repairs.append((str(prim.GetPath()), prim.GetName()))

    repairs.sort()
    repaired_paths: list[str] = []
    for prim_path, shape_name in repairs:
        prim = stage.GetPrimAtPath(prim_path)
        prim.GetReferences().SetReferences([])
        if shape_name == "sphere":
            shape = UsdGeom.Sphere.Define(stage, prim_path)
            shape.CreateRadiusAttr(0.5)
        else:
            shape = UsdGeom.Cube.Define(stage, prim_path)
            shape.CreateSizeAttr(1.0)
        collision_api = UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        collision_api.CreateSimulationOwnerRel().SetTargets(
            [Sdf.Path(scene.physics_scene_prim)]
        )
        repaired_paths.append(prim_path)
    return repaired_paths


def _prepare_replacement_layer(
    root_layer,
    template_path: Path,
    episode_spec,
    scene: SceneConfig,
):
    """Block original asset references before composing the replacement stage."""

    # Importing Usd registers the USDA file-format plugin used by Sdf.
    from pxr import Sdf, Usd  # noqa: F401

    template_layer = Sdf.Layer.FindOrOpen(str(template_path.resolve()))
    if template_layer is None:
        raise RuntimeError(f"Failed to open template layer: {template_path}")

    replacement_paths = {
        replacement.target_prim_path for replacement in episode_spec.replacements
    }
    for target_path in sorted(replacement_paths):
        target_spec = template_layer.GetPrimAtPath(target_path)
        if target_spec is None:
            raise RuntimeError(f"Replacement target not found: {target_path}")
        for child in target_spec.nameChildren:
            override = Sdf.CreatePrimInLayer(root_layer, child.path)
            override.specifier = Sdf.SpecifierOver
            override.SetInfo("references", Sdf.ReferenceListOp.CreateExplicit([]))
            override.active = False

    repaired_paths = []

    def walk(prim_spec):
        yield prim_spec
        for child in prim_spec.nameChildren:
            yield from walk(child)

    world_spec = template_layer.GetPrimAtPath(scene.world_prim)
    if world_spec is None:
        return repaired_paths
    for prim_spec in walk(world_spec):
        prim_path = str(prim_spec.path)
        if prim_spec.name not in {"sphere", "prism"}:
            continue
        if any(prim_path.startswith(f"{target}/") for target in replacement_paths):
            continue
        references = prim_spec.referenceList.GetAddedOrExplicitItems()
        if not references:
            continue
        override = Sdf.CreatePrimInLayer(root_layer, prim_spec.path)
        override.specifier = Sdf.SpecifierDef
        override.SetInfo("references", Sdf.ReferenceListOp.CreateExplicit([]))
        override.typeName = "Sphere" if prim_spec.name == "sphere" else "Cube"
        repaired_paths.append(prim_path)
    return repaired_paths


def _finish_prepared_primitives(stage, scene: SceneConfig, prim_paths: list[str]) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics

    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.GetName() == "sphere":
            UsdGeom.Sphere(prim).CreateRadiusAttr(0.5)
        else:
            UsdGeom.Cube(prim).CreateSizeAttr(1.0)
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateSimulationOwnerRel().SetTargets(
            [Sdf.Path(scene.physics_scene_prim)]
        )


def _define_camera(stage, camera: CameraSpec) -> None:
    """Author a camera prim using the Cosmos translate+rotateXYZ convention.

    ``look_at`` (if given) orients the camera so its local -Z axis points at
    the target. When neither ``look_at`` nor ``orientation_xyzw`` is given the
    camera keeps its identity orientation.
    """

    from pxr import Gf, UsdGeom

    prim = UsdGeom.Camera.Define(stage, camera.prim_path)
    prim.CreateFocalLengthAttr(camera.focal_length)
    prim.CreateHorizontalApertureAttr(camera.horizontal_aperture)
    prim.CreateVerticalApertureAttr(camera.vertical_aperture)
    prim.CreateClippingRangeAttr(Gf.Vec2f(1.0, 1_000_000.0))

    xform = UsdGeom.Xformable(prim.GetPrim())
    xform.ClearXformOpOrder()
    precision = UsdGeom.XformOp.PrecisionDouble
    translate_op = xform.AddTranslateOp(precision=precision)
    translate_op.Set(Gf.Vec3d(*camera.position))

    if camera.orientation_xyzw is not None:
        x, y, z, w = (float(value) for value in camera.orientation_xyzw)
        rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        euler = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
        rotate_op = xform.AddRotateXYZOp(precision=precision)
        rotate_op.Set(Gf.Vec3d(euler[0], euler[1], euler[2]))
    elif camera.look_at is not None:
        position = Gf.Vec3d(*camera.position)
        target = Gf.Vec3d(*camera.look_at)
        forward = (target - position).GetNormalized()
        # Build the camera-to-world basis: local X = right, local Y = up,
        # local Z = -forward (USD cameras look down their -Z axis).
        world_up = Gf.Vec3d(0, 0, 1)
        right = world_up.GetCross(forward).GetNormalized()
        up = forward.GetCross(right)
        basis = Gf.Matrix3d(1)
        basis.SetColumn(0, right)
        basis.SetColumn(1, up)
        basis.SetColumn(2, -forward)
        # Extract the quaternion from the basis and use it directly via an
        # orient op (double precision), matching the Cosmos camera convention.
        quat = basis.ExtractRotation().GetQuat()
        orient_op = xform.AddOrientOp(precision=precision)
        orient_op.Set(Gf.Quatd(quat.GetReal(), quat.GetImaginary()))

    scale_op = xform.AddScaleOp(precision=precision)
    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))


def _add_static_collider(
    stage,
    prim_path: str,
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    scene: SceneConfig,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Add a static (non-rigid) box collider with a visible surface."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage.DefinePrim(prim_path, "Xform")
    geometry = UsdGeom.Cube.Define(stage, f"{prim_path}/Geometry")
    geometry.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(geometry.GetPrim())
    precision = UsdGeom.XformOp.PrecisionDouble
    xform.AddTranslateOp(precision=precision).Set(Gf.Vec3d(*translate))
    xform.AddScaleOp(precision=precision).Set(Gf.Vec3d(size[0], size[1], size[2]))

    collision_api = UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
    collision_api.CreateSimulationOwnerRel().SetTargets(
        [Sdf.Path(scene.physics_scene_prim)]
    )
    _bind_visual_color(stage, geometry.GetPrim(), color)


def _build_procedural_backdrop(stage, scene: SceneConfig) -> None:
    """Author ground/table/walls/tray from ``scene.procedural``."""

    if scene.procedural is None:
        return
    proc = scene.procedural

    stage.DefinePrim(_GENERATED_ROOT, "Xform")

    if proc.build_ground:
        half = proc.ground_size / 2.0
        _add_static_collider(
            stage,
            f"{scene.world_prim}/Ground",
            (proc.ground_size, proc.ground_size, 0.1),
            proc.ground_color,
            scene,
            translate=(0.0, 0.0, -0.05),
        )

    if proc.table:
        width, length = proc.table_size
        height = 0.72
        _add_static_collider(
            stage,
            f"{scene.world_prim}/TableTop",
            (width, length, 0.06),
            (0.25, 0.55, 0.20),
            scene,
            translate=(0.0, 0.0, height),
        )
        for index, offset in ((0, -0.62), (1, 0.62)):
            _add_static_collider(
                stage,
                f"{scene.world_prim}/TableLeg_{index}",
                (0.1, 0.1, height),
                (0.3, 0.3, 0.32),
                scene,
                translate=(0.0, offset, height / 2.0),
            )

    if proc.walls:
        half = proc.ground_size / 2.0
        wall_thickness = 0.1
        walls = (
            (f"{scene.world_prim}/Wall_N", (half, wall_thickness), (0, half)),
            (f"{scene.world_prim}/Wall_S", (half, wall_thickness), (0, -half)),
            (f"{scene.world_prim}/Wall_E", (wall_thickness, half), (half, 0)),
            (f"{scene.world_prim}/Wall_W", (wall_thickness, half), (-half, 0)),
        )
        for path, (sx, sy), (ox, oy) in walls:
            _add_static_collider(
                stage,
                path,
                (sx, sy, proc.wall_height),
                (0.4, 0.4, 0.42),
                scene,
                translate=(ox, oy, proc.wall_height / 2.0),
            )

    if proc.sand_tray:
        tray_size = 0.5
        tray_height = 0.35
        tray_wall = 0.06
        _add_static_collider(
            stage,
            f"{scene.world_prim}/SandTray",
            (tray_size, tray_size, tray_wall),
            (0.45, 0.35, 0.25),
            scene,
            translate=(0.0, 0.0, tray_height),
        )
        half = tray_size / 2.0
        for index, (dx, dy) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
            _add_static_collider(
                stage,
                f"{scene.world_prim}/SandTrayWall_{index}",
                (
                    tray_size if dx == 0 else tray_wall,
                    tray_size if dy == 0 else tray_wall,
                    tray_height,
                ),
                (0.45, 0.35, 0.25),
                scene,
                translate=(dx * half, dy * half, tray_height / 2.0),
            )


def _define_episode_cameras(stage, episode_spec: EpisodeSpec) -> None:
    for camera in episode_spec.cameras.values():
        _define_camera(stage, camera)


def _apply_runner_scene_hook(stage, episode_spec: EpisodeSpec, scene: SceneConfig) -> None:
    """Let a non-rigid runner author its own scene extensions.

    Category 08's deformable runner uses this to build soft-ball meshes after
    the generic object placement. Hook functions must be pure pxr (no runtime).
    """

    if episode_spec.runner == "rigid":
        return
    import importlib

    module = importlib.import_module(f"phy_data_gen.runners.{episode_spec.runner}")
    hook = getattr(module, "build_scene_hook", None)
    if hook is not None:
        hook(stage, episode_spec, scene)


def build_scene(
    episode_spec: EpisodeSpec,
    scene: SceneConfig,
    output_path: Path,
) -> Path:
    """Create the episode scene USDA and return its path."""

    from pxr import Sdf, Usd

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    if episode_spec.template_path is None:
        root_layer = Sdf.Layer.CreateNew(str(output_path))
        stage = Usd.Stage.Open(root_layer)
        stage.DefinePrim(scene.world_prim, "Xform")
        stage.DefinePrim(scene.physics_scene_prim, "PhysicsScene")
        _build_procedural_backdrop(stage, scene)
        _define_episode_cameras(stage, episode_spec)
        for object_spec in episode_spec.objects:
            _add_object(stage, object_spec, scene)
        _apply_runner_scene_hook(stage, episode_spec, scene)
        root_layer.Save()
        return output_path

    template_path = Path(episode_spec.template_path).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")

    root_layer = Sdf.Layer.CreateNew(str(output_path))
    root_layer.subLayerPaths.append(str(template_path))

    prepared_paths = []
    if episode_spec.object_mode == "replace_assets":
        prepared_paths = _prepare_replacement_layer(
            root_layer,
            template_path,
            episode_spec,
            scene,
        )

    stage = Usd.Stage.Open(root_layer)

    if prepared_paths:
        _finish_prepared_primitives(stage, scene, prepared_paths)

    if episode_spec.object_mode in {"generated_objects", "procedural"}:
        _disable_template_dynamics(stage, scene)
        # Procedural backdrop (tray, extra walls, ground) may be authored even
        # when a template supplies the base scene.
        _build_procedural_backdrop(stage, scene)

        stage.DefinePrim(_GENERATED_ROOT, "Xform")
        for object_spec in episode_spec.objects:
            _add_object(stage, object_spec, scene)
        _define_episode_cameras(stage, episode_spec)
        _apply_runner_scene_hook(stage, episode_spec, scene)
    elif episode_spec.object_mode == "replace_assets":
        if prepared_paths:
            print(
                f"Repaired {len(prepared_paths)} unresolved template "
                "sphere/prism reference(s) before replacement"
            )
        for replacement in episode_spec.replacements:
            _replace_asset(stage, replacement, scene)
    else:
        repaired_paths = _repair_missing_template_primitives(stage, scene)
        if repaired_paths:
            print(
                f"Repaired {len(repaired_paths)} unresolved template "
                "sphere/prism reference(s)"
            )

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
