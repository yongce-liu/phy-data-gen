"""Deformable (soft-body) runner for category 08.

The rigid-body pipeline in :mod:`phy_data_gen.simulation` cannot record
soft-body nodal states, so category 08 supplies this dedicated runner. It is
selected via ``EpisodeSpec.runner == "deformable"``.

``build_scene_hook`` is pure pxr and runs during scene compilation: it replaces
each soft-ball ``Sphere`` geometry with a UV-sphere ``UsdGeom.Mesh`` that serves
as the deformable cooking source. ``run_simulation`` (runtime) applies the
volume-deformable schema + material, steps PhysX, and records nodal positions.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from phy_data_gen.config import SceneConfig
from phy_data_gen.schemas import EpisodeSpec

# ``isaacsim.core.experimental.prims`` (DeformablePrim) and its dependencies
# (isaacsim.core.experimental.utils, isaacsim.core.simulation_manager, …) ship
# as pip extensions under isaacsim/exts/ and are not on sys.path unless each
# extension is enabled. The Isaac runtime launches from the venv, so add every
# ``isaacsim.core.*`` ext dir to the path so the imports resolve inside
# run_simulation.
_ISAACSIM_ROOT = Path(__file__).resolve().parents[3]  # .../site-packages/isaacsim
_EXTS = _ISAACSIM_ROOT / "exts"
for _ext_dir in sorted(_EXTS.iterdir()) if _EXTS.is_dir() else []:
    _pkg = _ext_dir / "isaacsim"
    if (_pkg / "core").is_dir() and str(_ext_dir) not in sys.path:
        sys.path.insert(0, str(_ext_dir))


def _uv_sphere_mesh(stage, prim_path: str, radius: float, segments: int = 24, rings: int = 14):
    """Define a UV-sphere UsdGeom.Mesh with normals."""

    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = []
    normals = []
    uvs = []
    face_counts = []
    face_indices = []
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for seg in range(segments + 1):
            phi = 2.0 * math.pi * seg / segments
            x = radius * sin_theta * math.cos(phi)
            y = radius * sin_theta * math.sin(phi)
            z = radius * cos_theta
            points.append(Gf.Vec3f(x, y, z))
            normals.append(Gf.Vec3f(x, y, z))
            uvs.append(Gf.Vec2f(seg / segments, ring / rings))
            if ring < rings and seg < segments:
                a = ring * (segments + 1) + seg
                b = a + 1
                c = a + (segments + 1)
                d = c + 1
                face_counts.append(3)
                face_indices.extend([a, b, c])
                face_counts.append(3)
                face_indices.extend([b, d, c])

    mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray(normals))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_indices))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    from pxr import Sdf, UsdGeom as _Ug

    _Ug.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.Float2Array, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec2fArray(uvs))
    extent = [Gf.Vec3f(-radius, -radius, -radius), Gf.Vec3f(radius, radius, radius)]
    mesh.GetExtentAttr().Set(Vt.Vec3fArray(extent))
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def build_scene_hook(stage, episode_spec: EpisodeSpec, scene: SceneConfig) -> None:
    """Replace soft-ball spheres with UV-sphere meshes (pure pxr)."""

    from pxr import UsdGeom

    for obj in episode_spec.objects:
        geometry_path = f"/World/GeneratedObjects/{obj.object_id}/Geometry"
        prim = stage.GetPrimAtPath(geometry_path)
        if not prim or not prim.IsA(UsdGeom.Sphere):
            continue
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        # Reuse the same path: define a Mesh there.
        stage.RemovePrim(geometry_path)
        mesh = _uv_sphere_mesh(stage, geometry_path, radius)
        # Preserve the visual colour by binding to the parent's material.
        from phy_data_gen.scene import _bind_visual_color

        color = tuple(obj.color)
        _bind_visual_color(stage, mesh.GetPrim(), color)


def run_simulation(
    scene_path: Path,
    episode_spec: EpisodeSpec,
    simulation_app,
    output_root: Path,
    cameras: dict[str, str],
    world_prim_path: str,
    render_width: int,
    render_height: int,
    depth_scale_meters: float,
    rgb_encoder: str,
    capture_frames: bool = True,
):
    """Open the scene, apply deformable schemas, step, record nodal states + frames."""

    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext

    from phy_data_gen.recording import FrameRecorder

    context = omni.usd.get_context()
    if not context.open_stage(str(scene_path.resolve())):
        raise RuntimeError(f"Failed to open scene: {scene_path}")

    for _ in range(20):
        simulation_app.update()

    stage = context.get_stage()

    # ---- Apply deformable schema + material to each soft ball ----
    from omni.physx.scripts import deformableUtils

    dm = episode_spec.metadata.get("deformable", {})
    youngs = float(dm.get("youngs_modulus", 1e5))
    poisson = float(dm.get("poissons_ratio", 0.45))
    density = float(dm.get("density", 300.0))

    from pxr import UsdGeom, UsdShade

    deformable_paths: list[str] = []
    for obj in episode_spec.objects:
        geometry_path = f"/World/GeneratedObjects/{obj.object_id}/Geometry"
        prim = stage.GetPrimAtPath(geometry_path)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            continue
        # The mesh sits under the object Xform; create the deformable hierarchy
        # using the Xform as root (must not be a Gprim).
        root_path = f"/World/GeneratedObjects/{obj.object_id}"
        # The generic object authoring applied a rigid-body API to the Xform;
        # PhysX rejects a deformable volume whose root is also a rigid body.
        # Strip the rigid-body/mass/physics-material APIs and the CCD options
        # so the volume hierarchy can take over.
        root_prim = stage.GetPrimAtPath(root_path)
        if root_prim:
            from pxr import UsdPhysics

            root_prim.RemoveProperty("physics:velocity")
            root_prim.RemoveProperty("physics:angularVelocity")
            root_prim.RemoveProperty("physics:mass")
            root_prim.RemoveProperty("physxRigidBody:enableCCD")
            root_prim.RemoveProperty("physxRigidBody:maxDepenetrationVelocity")
            root_prim.RemoveProperty("physxRigidBody:maxLinearVelocity")
            # Remove the applied schemas so PhysX no longer treats the Xform
            # as a rigid body.
            root_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            root_prim.RemoveAPI(UsdPhysics.MassAPI)
        sim_mesh_path = f"{root_path}/SimMesh"
        cooking = geometry_path
        ok = deformableUtils.create_auto_volume_deformable_hierarchy(
            stage,
            root_path,
            sim_mesh_path,
            sim_mesh_path,
            cooking,
            simulation_hex_mesh_enabled=False,
            cooking_src_simplification_enabled=False,
            set_visibility_with_guide_purpose=True,
        )
        if not ok:
            raise RuntimeError(f"Failed to create deformable hierarchy for {root_path}")
        deformable_paths.append(root_path)

        # Material.
        material_path = f"/World/Materials/SoftMat_{obj.object_id}"
        deformableUtils.add_deformable_material(
            stage,
            material_path,
            density=density,
            youngs_modulus=youngs,
            poissons_ratio=poisson,
        )
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(root_path)).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )

    if not deformable_paths:
        raise RuntimeError("No deformable objects found in episode")

    num_physics_steps = round(episode_spec.duration_seconds / episode_spec.physics_dt)
    capture_every = max(1, round(1.0 / (episode_spec.render_fps * episode_spec.physics_dt)))

    sim = SimulationContext(SimulationCfg(dt=episode_spec.physics_dt))

    from isaacsim.core.experimental.prims import DeformablePrim

    deformables = DeformablePrim(deformable_paths, deformable_type="volume")

    frame_recorder = None
    if capture_frames:
        frame_recorder = FrameRecorder(
            output_root=output_root,
            run_id=episode_spec.episode_id,
            cameras=cameras,
            width=render_width,
            height=render_height,
            fps=episode_spec.render_fps,
            depth_scale_meters=depth_scale_meters,
            rgb_encoder=rgb_encoder,
        )
        frame_recorder.initialize()

    from phy_data_gen.dataset import build_camera_metadata

    metadata = build_camera_metadata(
        stage=stage,
        cameras=cameras,
        frame_count=len(range(0, num_physics_steps, capture_every)),
        width=render_width,
        height=render_height,
        depth_scale_meters=depth_scale_meters,
    )

    sim.reset()

    records: list[dict] = []
    deformable_records: list[dict] = []
    frame_index = 0
    try:
        for step in range(num_physics_steps):
            sim.step(render=False)
            if step % capture_every == 0:
                timestamp = step * episode_spec.physics_dt
                sim_pos, _coll_pos, _rest_pos = deformables.get_nodal_positions()
                sim_np = np.asarray(sim_pos).reshape(-1, 3)
                center = sim_np.mean(axis=0)
                radii = np.linalg.norm(sim_np - center, axis=1)
                records.append(
                    {
                        "frame": frame_index,
                        "timestamp": timestamp,
                        "center": center.tolist(),
                        "mean_radius": float(radii.mean()),
                        "max_radius": float(radii.max()),
                        "min_radius": float(radii.min()),
                        "num_nodes": int(sim_np.shape[0]),
                    }
                )
                # Compact nodal dump for the first soft ball (N<=~800 floats).
                if sim_np.shape[0] < 2000:
                    deformable_records.append(
                        {
                            "frame": frame_index,
                            "nodes": sim_np.tolist(),
                        }
                    )
                if frame_recorder is not None:
                    sim.render()
                    frame_recorder.capture(simulation_app)
                frame_index += 1
        if frame_recorder is not None:
            frame_recorder.finalize()
    except BaseException:
        if frame_recorder is not None:
            frame_recorder.abort()
        raise

    physics_dir = output_root / "physics" / episode_spec.episode_id
    physics_dir.mkdir(parents=True, exist_ok=True)
    with (physics_dir / "deformable_states.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec))
            f.write("\n")
    with (physics_dir / "deformable_nodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in deformable_records:
            f.write(json.dumps(rec))
            f.write("\n")
    # Minimal validation: the ball deformed (radius contraction) if compressed.
    if records:
        max_contraction = max(
            (1.0 - rec["min_radius"] / rec["mean_radius"])
            for rec in records
            if rec["mean_radius"] > 0
        )
        summary = {
            "passed": bool(max_contraction > 0.02),
            "frames": len(records),
            "max_contraction": max_contraction,
        }
    else:
        summary = {"passed": False, "frames": 0, "max_contraction": 0.0}

    (physics_dir / "validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    from phy_data_gen.dataset import save_camera_metadata

    save_camera_metadata(metadata, output_root, episode_spec.episode_id)

    # Mirrors the rigid runner's SimulationResult surface for the caller.
    from dataclasses import dataclass, field

    @dataclass
    class DeformableResult:
        states: object
        num_physics_steps: int
        captured_frames: int
        object_ids: list
        object_paths: list
        camera_metadata: dict
        rgb_paths: dict
        depth_paths: dict
        deformable_records: list = field(default_factory=list)

    return DeformableResult(
        states=None,
        num_physics_steps=num_physics_steps,
        captured_frames=frame_index,
        object_ids=deformable_paths,
        object_paths=deformable_paths,
        camera_metadata=metadata,
        rgb_paths=frame_recorder.rgb_paths if frame_recorder else {},
        depth_paths=frame_recorder.depth_paths if frame_recorder else {},
        deformable_records=deformable_records,
    )
