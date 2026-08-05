"""Run a PhysX simulation over a compiled episode scene.

Flow (Isaac Lab standalone):

    AppLauncher -> open episode USDA -> SimulationContext -> reset ->
    step loop -> sample rigid-body states every ``capture_every`` steps.

Only PhysX is implemented in the first version. Rigid-body state is read
through lightweight PhysX tensor views (world pose + velocities).

``pxr``/``omni``/``isaaclab`` imports happen only after the app launches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from phy_data_gen.recording import FrameRecorder, StateRecorder
from phy_data_gen.schemas import EpisodeSpec

_GENERATED_ROOT = "/World/GeneratedObjects"


@dataclass
class SimulationResult:
    states: StateRecorder
    num_physics_steps: int
    captured_frames: int = 0
    object_ids: list[str] = field(default_factory=list)
    object_paths: list[str] = field(default_factory=list)
    camera_metadata: dict[str, dict] = field(default_factory=dict)
    rgb_paths: dict[str, Path] = field(default_factory=dict)
    depth_paths: dict[str, Path] = field(default_factory=dict)


def compute_step_counts(
    duration_seconds: float,
    physics_dt: float,
    render_fps: int,
) -> tuple[int, int]:
    """Return (num_physics_steps, capture_every)."""

    num_physics_steps = round(duration_seconds / physics_dt)
    capture_every = max(1, round(1.0 / (render_fps * physics_dt)))
    return num_physics_steps, capture_every


def _find_generated_rigid_body_paths(
    stage,
    episode_spec: EpisodeSpec,
) -> dict[str, str]:
    """Resolve the single rigid-body prim below each generated object."""

    from pxr import Usd, UsdPhysics

    paths: dict[str, str] = {}
    for object_spec in episode_spec.objects:
        object_path = f"{_GENERATED_ROOT}/{object_spec.object_id}"
        root_prim = stage.GetPrimAtPath(object_path)
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"Generated object prim not found: {object_path}")

        rigid_prims = [
            prim
            for prim in Usd.PrimRange(root_prim)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(rigid_prims) != 1:
            matched = ", ".join(str(prim.GetPath()) for prim in rigid_prims)
            raise RuntimeError(
                f"Expected one rigid body below {object_path}, found "
                f"{len(rigid_prims)}: {matched or 'none'}"
            )
        paths[object_spec.object_id] = str(rigid_prims[0].GetPath())
    return paths


def _find_template_rigid_body_paths(stage, world_prim_path: str) -> dict[str, str]:
    """Discover every rigid body authored below the template world prim."""

    from pxr import Usd, UsdPhysics

    world_prim = stage.GetPrimAtPath(world_prim_path)
    if not world_prim or not world_prim.IsValid():
        raise RuntimeError(f"World prim not found: {world_prim_path}")

    rigid_prims = sorted(
        (
            prim
            for prim in Usd.PrimRange(world_prim)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        key=lambda prim: str(prim.GetPath()),
    )
    if not rigid_prims:
        raise RuntimeError(f"No template rigid bodies found below: {world_prim_path}")

    name_counts: dict[str, int] = {}
    for prim in rigid_prims:
        name = prim.GetName()
        name_counts[name] = name_counts.get(name, 0) + 1

    paths: dict[str, str] = {}
    for prim in rigid_prims:
        prim_path = str(prim.GetPath())
        name = prim.GetName()
        if name_counts[name] == 1:
            object_id = name
        else:
            object_id = prim_path.removeprefix(f"{world_prim_path}/").replace("/", "__")
        paths[object_id] = prim_path
    return paths


def _find_replacement_rigid_body_paths(
    stage,
    episode_spec: EpisodeSpec,
    world_prim_path: str,
) -> dict[str, str]:
    """Resolve replacement bodies and template bodies that were not replaced."""

    from pxr import Usd, UsdPhysics

    paths = {}
    for replacement in episode_spec.replacements:
        target = stage.GetPrimAtPath(replacement.target_prim_path)
        if not target or not target.IsValid():
            raise RuntimeError(
                f"Replacement target not found: {replacement.target_prim_path}"
            )
        rigid_prims = [
            prim
            for prim in Usd.PrimRange(target)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(rigid_prims) != 1:
            matched = ", ".join(str(prim.GetPath()) for prim in rigid_prims)
            raise RuntimeError(
                f"Expected one replacement rigid body below "
                f"{replacement.target_prim_path}, found {len(rigid_prims)}: "
                f"{matched or 'none'}"
            )
        paths[replacement.object_id] = str(rigid_prims[0].GetPath())

    replacement_targets = {
        replacement.target_prim_path for replacement in episode_spec.replacements
    }
    template_paths = _find_template_rigid_body_paths(stage, world_prim_path)
    for object_id, prim_path in template_paths.items():
        if any(
            prim_path == target or prim_path.startswith(f"{target}/")
            for target in replacement_targets
        ):
            continue
        paths[object_id] = prim_path
    return paths


def _find_rigid_body_paths(
    stage,
    episode_spec: EpisodeSpec,
    world_prim_path: str,
) -> dict[str, str]:
    if episode_spec.object_mode == "generated_objects":
        return _find_generated_rigid_body_paths(stage, episode_spec)
    if episode_spec.object_mode == "template_dynamics":
        return _find_template_rigid_body_paths(stage, world_prim_path)
    if episode_spec.object_mode == "replace_assets":
        return _find_replacement_rigid_body_paths(
            stage,
            episode_spec,
            world_prim_path,
        )
    raise ValueError(f"Unsupported object mode: {episode_spec.object_mode}")


def _create_rigid_body_views(sim, rigid_body_paths: dict[str, str]):
    """Create lightweight PhysX tensor views for state recording.

    Isaac Lab 3.0.0b2's high-level ``RigidObject`` currently passes a PhysX
    ``ProxyArray`` into a Warp kernel during initialization. Direct tensor
    views provide the state data needed here without hitting that code path.
    """

    physics_view = sim.physics_manager.get_physics_sim_view()
    if physics_view is None:
        raise RuntimeError("PhysX simulation view was not initialized")

    views = {}
    for object_id, prim_path in rigid_body_paths.items():
        view = physics_view.create_rigid_body_view(prim_path)
        if view.count != 1:
            raise RuntimeError(
                f"Failed to create one rigid-body view for {object_id}: "
                f"{prim_path} (count={view.count})"
            )
        views[object_id] = view
    return views


def _sample_states(
    views,
    recorder: StateRecorder,
    frame: int,
    timestamp: float,
) -> None:
    import warp as wp

    for object_id, view in views.items():
        transform = wp.to_torch(view.get_transforms().contiguous())[0].cpu().tolist()
        velocity = wp.to_torch(view.get_velocities().contiguous())[0].cpu().tolist()

        position = transform[:3]
        # PhysX tensor transforms store quaternions in (x, y, z, w) order.
        orientation = transform[3:7]
        lin_vel = velocity[:3]
        ang_vel = velocity[3:6]

        recorder.append(
            frame=frame,
            timestamp=timestamp,
            object_id=object_id,
            position=position,
            orientation_xyzw=orientation,
            linear_velocity=lin_vel,
            angular_velocity=ang_vel,
        )


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
) -> SimulationResult:
    """Open the scene, simulate it and record states (and optionally frames)."""

    import omni.usd
    from isaaclab.sim import SimulationCfg, SimulationContext

    context = omni.usd.get_context()
    resolved = scene_path.resolve()
    if not context.open_stage(str(resolved)):
        raise RuntimeError(f"Failed to open scene: {resolved}")

    for _ in range(20):
        simulation_app.update()

    num_physics_steps, capture_every = compute_step_counts(
        episode_spec.duration_seconds,
        episode_spec.physics_dt,
        episode_spec.render_fps,
    )

    sim = SimulationContext(
        SimulationCfg(dt=episode_spec.physics_dt),
    )

    stage = context.get_stage()
    rigid_body_paths = _find_rigid_body_paths(
        stage,
        episode_spec,
        world_prim_path,
    )

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

    sim.reset()
    views = _create_rigid_body_views(sim, rigid_body_paths)

    recorder = StateRecorder()
    result = SimulationResult(
        states=recorder,
        num_physics_steps=num_physics_steps,
        object_ids=list(rigid_body_paths),
        object_paths=list(rigid_body_paths.values()),
    )

    from phy_data_gen.dataset import build_camera_metadata

    result.camera_metadata = build_camera_metadata(
        stage=stage,
        cameras=cameras,
        frame_count=len(range(0, num_physics_steps, capture_every)),
        width=render_width,
        height=render_height,
        depth_scale_meters=depth_scale_meters,
    )
    if frame_recorder is not None:
        result.rgb_paths = frame_recorder.rgb_paths
        result.depth_paths = frame_recorder.depth_paths

    frame_index = 0
    progress_every = max(1, num_physics_steps // 10)
    try:
        for step in range(num_physics_steps):
            # Render only at capture points instead of every physics step.
            sim.step(render=False)

            if step % capture_every == 0:
                timestamp = step * episode_spec.physics_dt
                _sample_states(views, recorder, frame_index, timestamp)
                if frame_recorder is not None:
                    sim.render()
                    frame_recorder.capture(simulation_app)
                frame_index += 1

            completed_steps = step + 1
            if (
                completed_steps % progress_every == 0
                or completed_steps == num_physics_steps
            ):
                percent = 100 * completed_steps // num_physics_steps
                print(
                    f"Simulation progress: {percent}% "
                    f"({completed_steps}/{num_physics_steps} physics steps, "
                    f"{frame_index} frame(s) sampled)",
                    flush=True,
                )

        if frame_recorder is not None:
            frame_recorder.finalize()
            if frame_recorder.captured_frames != frame_index:
                raise RuntimeError(
                    f"Expected {frame_index} captured frame(s), got "
                    f"{frame_recorder.captured_frames}"
                )
    except Exception:
        if frame_recorder is not None:
            frame_recorder.abort()
        raise

    result.captured_frames = frame_index
    return result
