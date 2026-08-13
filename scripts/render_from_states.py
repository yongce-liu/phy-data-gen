#!/usr/bin/env python3
"""Render RGB and depth videos by replaying previously recorded rigid-body states.

The replay path does not advance PhysX. It opens each generated scene, authors
recorded poses into the USD session layer, and captures the configured cameras.
Source USDA files are never modified.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class OfflineRenderArgs:
    """Render episodes generated with ``--no-frames``."""

    config: Path
    device: str = "cuda:0"
    seed_start: int | None = None
    num_episodes: int | None = None
    batch_size: int = 1
    env_spacing: float = 100.0
    frame_stride: int = 1
    cameras: list[str] | None = None
    overwrite: bool = False
    rendering_mode: str = "balanced"


@dataclass(frozen=True)
class EpisodeInput:
    run_id: str
    seed: int
    scene_path: Path
    spec_path: Path
    states_path: Path


def _seed_from_run_id(run_id: str) -> int | None:
    """Extract the trailing integer seed from a run ID."""

    value = run_id.rsplit("_", 1)[-1]
    if not value.lstrip("-").isdigit():
        return None
    return int(value)


def discover_episodes(
    output_root: Path,
    scene_name: str,
    seed_start: int | None = None,
    num_episodes: int | None = None,
) -> list[EpisodeInput]:
    """Find validated episodes that have scenes and recorded states."""

    if num_episodes is not None and num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    episodes: list[EpisodeInput] = []
    physics_root = output_root / "physics"
    if not physics_root.is_dir():
        return episodes

    for physics_dir in physics_root.iterdir():
        if not physics_dir.is_dir():
            continue
        seed = _seed_from_run_id(physics_dir.name)
        if seed is None or (seed_start is not None and seed < seed_start):
            continue

        validation_path = physics_dir / "validation.json"
        states_path = physics_dir / "object_states.jsonl"
        scene_dir = output_root / "scene" / physics_dir.name
        scene_path = scene_dir / f"{scene_name}.usda"
        spec_path = scene_dir / "episode_spec.json"
        if not all(
            path.is_file()
            for path in (validation_path, states_path, scene_path, spec_path)
        ):
            continue
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not validation.get("passed", False):
            continue

        episodes.append(
            EpisodeInput(
                run_id=physics_dir.name,
                seed=seed,
                scene_path=scene_path,
                spec_path=spec_path,
                states_path=states_path,
            )
        )

    episodes.sort(key=lambda item: (item.seed, item.run_id))
    return episodes[:num_episodes]


def load_state_frames(states_path: Path, frame_stride: int = 1) -> list[list[dict]]:
    """Load JSONL records grouped by frame, optionally subsampling frames."""

    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")

    frames: dict[int, list[dict]] = defaultdict(list)
    with states_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame = int(record["frame"])
                object_id = str(record["object_id"])
                position = record["position"]
                orientation = record["orientation_xyzw"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid state record at {states_path}:{line_number}"
                ) from error
            if len(position) != 3 or len(orientation) != 4:
                raise ValueError(
                    f"Invalid pose at {states_path}:{line_number} for {object_id}"
                )
            record["object_id"] = object_id
            frames[frame].append(record)

    selected = [frames[index] for index in sorted(frames) if index % frame_stride == 0]
    if not selected:
        raise ValueError(f"No state frames found in {states_path}")
    return selected


def outputs_complete(
    output_root: Path,
    run_id: str,
    camera_names: list[str],
) -> bool:
    """Return whether every selected camera has non-empty RGB and depth output."""

    paths = []
    for camera_name in camera_names:
        paths.extend(
            [
                output_root / "videos" / run_id / f"{camera_name}.mp4",
                output_root / "depths" / run_id / f"{camera_name}.mkv",
            ]
        )
    return bool(paths) and all(path.is_file() and path.stat().st_size > 0 for path in paths)


def _remove_outputs(output_root: Path, run_id: str, camera_names: list[str]) -> None:
    """Remove partial encoder outputs after a failed or interrupted render."""

    for camera_name in camera_names:
        for path in (
            output_root / "videos" / run_id / f"{camera_name}.mp4",
            output_root / "depths" / run_id / f"{camera_name}.mkv",
        ):
            path.unlink(missing_ok=True)


def _load_object_paths(output_root: Path, run_id: str, camera_name: str) -> dict[str, str]:
    """Load the recorded object-to-prim mapping from static annotations."""

    path = output_root / "physics" / run_id / f"{camera_name}_static.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = {
            str(item["object_id"]): str(item["prim_path"])
            for item in payload["objects"]
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid static object metadata: {path}") from error
    if not mapping:
        raise ValueError(f"No objects found in static metadata: {path}")
    return mapping


def _prefixed_path(env_path: str, source_path: str) -> str:
    """Map a path below source /World into one batch environment."""

    if source_path == "/World":
        return env_path
    if not source_path.startswith("/World/"):
        raise ValueError(f"Expected prim path below /World: {source_path}")
    return f"{env_path}{source_path.removeprefix('/World')}"


def _prepare_pose_ops(
    stage,
    rigid_body_paths: dict[str, str],
    environment_to_world=None,
):
    """Create session-layer matrix ops and retain each body's world scale."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage.SetEditTarget(stage.GetSessionLayer())
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    environment_to_world = environment_to_world or Gf.Matrix4d(1.0)
    prepared = {}
    for object_id, prim_path in rigid_body_paths.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Rigid body prim not found: {prim_path}")

        world_transform = cache.GetLocalToWorldTransform(prim)
        world_scale = Gf.Transform(world_transform).GetScale()
        parent = prim.GetParent()
        parent_to_world = (
            cache.GetLocalToWorldTransform(parent)
            if parent and parent.IsValid()
            else Gf.Matrix4d(1.0)
        )

        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if rigid_api:
            rigid_api.CreateRigidBodyEnabledAttr().Set(False)

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        op = xform.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble)
        prepared[object_id] = (
            op,
            parent_to_world.GetInverse(),
            world_scale,
            environment_to_world,
        )
    return prepared


def _apply_frame(prepared, records: list[dict]) -> None:
    """Apply one recorded frame as world-space poses."""

    from pxr import Gf

    seen = set()
    for record in records:
        object_id = record["object_id"]
        if object_id not in prepared:
            raise RuntimeError(f"State references unknown object: {object_id}")
        if object_id in seen:
            raise RuntimeError(f"Duplicate state for object {object_id} in one frame")
        seen.add(object_id)

        op, parent_inverse, world_scale, environment_to_world = prepared[object_id]
        x, y, z, w = (float(value) for value in record["orientation_xyzw"])
        transform = Gf.Transform()
        transform.SetTranslation(Gf.Vec3d(*record["position"]))
        transform.SetRotation(Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))))
        transform.SetScale(world_scale)
        desired_world = transform.GetMatrix() * environment_to_world
        local_transform = desired_world * parent_inverse
        op.Set(local_transform)

    missing = set(prepared) - seen
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"Frame is missing {len(missing)} object(s): {preview}")


def _render_batch(
    episodes: list[EpisodeInput],
    config,
    cameras: dict[str, str],
    frame_stride: int,
    env_spacing: float,
    simulation_app,
) -> int:
    """Compose, replay, and render several episode environments in one stage."""

    import omni.usd

    from phy_data_gen.recording import FrameRecorder
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()

    from pxr import Gf, UsdGeom

    from phy_data_gen.schemas import EpisodeSpec

    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Scope")
    batch_items = []
    columns = max(1, int(len(episodes) ** 0.5))
    for slot, episode in enumerate(episodes):
        spec = EpisodeSpec.model_validate_json(
            episode.spec_path.read_text(encoding="utf-8")
        )
        if spec.episode_id != episode.run_id or spec.seed != episode.seed:
            raise ValueError(f"Scene spec does not match run ID: {episode.run_id}")
        frames = load_state_frames(episode.states_path, frame_stride)
        env_path = f"/World/envs/env_{slot}"
        env_prim = stage.DefinePrim(env_path, "Xform")
        if not env_prim.GetReferences().AddReference(
            str(episode.scene_path.resolve()), "/World"
        ):
            raise RuntimeError(f"Failed to reference scene: {episode.scene_path}")
        offset = Gf.Vec3d(
            float(slot % columns) * env_spacing,
            float(slot // columns) * env_spacing,
            0.0,
        )
        UsdGeom.Xformable(env_prim).AddTranslateOp().Set(offset)
        environment_to_world = Gf.Matrix4d(1.0)
        environment_to_world.SetTranslate(offset)
        batch_items.append((episode, spec, frames, env_path, environment_to_world))

    for _ in range(20):
        simulation_app.update()
    frame_counts = {len(item[2]) for item in batch_items}
    if len(frame_counts) != 1:
        raise ValueError(f"Batch episodes have different frame counts: {frame_counts}")

    runtime_items = []
    try:
        for episode, spec, frames, env_path, environment_to_world in batch_items:
            source_paths = _load_object_paths(
                config.output_root, episode.run_id, next(iter(cameras))
            )
            rigid_body_paths = {
                object_id: _prefixed_path(env_path, path)
                for object_id, path in source_paths.items()
            }
            prepared = _prepare_pose_ops(
                stage, rigid_body_paths, environment_to_world
            )
            env_cameras = {
                name: _prefixed_path(env_path, path)
                for name, path in cameras.items()
            }
            recorder = FrameRecorder(
                output_root=config.output_root,
                run_id=episode.run_id,
                cameras=env_cameras,
                width=config.render.width,
                height=config.render.height,
                fps=max(1, round(spec.render_fps / frame_stride)),
                depth_scale_meters=config.render.depth_scale_meters,
                rgb_encoder=config.render.rgb_encoder,
            )
            runtime_items.append((episode, frames, prepared, recorder))
            recorder.initialize()

        frame_count = frame_counts.pop()
        for frame_index in range(frame_count):
            for _episode, frames, prepared, _recorder in runtime_items:
                _apply_frame(prepared, frames[frame_index])
            # Every environment and camera is rendered by this single update.
            simulation_app.update()
            for _episode, _frames, _prepared, recorder in runtime_items:
                recorder.capture(simulation_app, update=False)
        for _episode, _frames, _prepared, recorder in runtime_items:
            recorder.finalize()
    except BaseException:
        for episode, _frames, _prepared, recorder in runtime_items:
            recorder.abort()
            _remove_outputs(config.output_root, episode.run_id, list(cameras))
        raise
    finally:
        context.close_stage()
        for _ in range(3):
            simulation_app.update()
    return frame_count


def _batches(items: list[EpisodeInput], batch_size: int):
    """Yield consecutive batches without copying the complete input list."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _select_cameras(configured: dict[str, str], selected: list[str] | None):
    if selected is None:
        return configured
    unknown = sorted(set(selected) - set(configured))
    if unknown:
        raise ValueError(f"Unknown camera(s): {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one camera must be selected")
    return {name: configured[name] for name in selected}


def _probe_nvenc_sessions(rgb_encoder: str, device_index: int = 0) -> int:
    """Return the number of simultaneous NVENC sessions this GPU can hold.

    Consumer NVIDIA drivers cap concurrent NVENC encode sessions (typically 8
    on RTX 50-series). Exceeding the cap makes ffmpeg fail at startup, and
    every failed encoder used to wedge the whole batch by filling an unread
    stderr pipe. Probe by opening real encoders so the caller can size batches
    below the limit.
    """

    if rgb_encoder != "h264_nvenc":
        return 1 << 30  # CPU/software encoders are not GPU-session limited

    import subprocess
    import tempfile

    probe_dir = Path(tempfile.mkdtemp(prefix="nvenc_probe_"))
    src = probe_dir / "src.mp4"
    nvenc_opts = ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
    try:
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=1:size=1280x720:rate=30",
                "-frames:v", "2", "-c:v", "libx264", str(src),
            ],
            check=True,
            capture_output=True,
        )
        max_sessions = 0
        for count in (2, 4, 8, 12, 16, 24, 32):
            procs = [
                subprocess.Popen(
                    [
                        "ffmpeg", "-loglevel", "error", "-y",
                        "-i", str(src),
                        *nvenc_opts,
                        "-gpu", str(device_index),
                        str(probe_dir / f"out_{i}.mp4"),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                for i in range(count)
            ]
            successes = sum(proc.wait(timeout=30) == 0 for proc in procs)
            if successes < count:
                max_sessions = max(2, successes)
                break
            max_sessions = count
        return max_sessions
    finally:
        for path in probe_dir.iterdir():
            path.unlink(missing_ok=True)
        probe_dir.rmdir()


def main() -> int:
    args = tyro.cli(OfflineRenderArgs)
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.env_spacing <= 0.0:
        raise ValueError("env_spacing must be positive")

    from phy_data_gen.config import load_config

    config = load_config(args.config)
    cameras = _select_cameras(config.scene.cameras, args.cameras)
    # Each episode opens one ffmpeg NVENC encoder per camera, all alive for the
    # whole batch. Consumer GPUs cap concurrent NVENC sessions (typically 8);
    # overshooting makes the extra encoders fail at startup. Auto-clamp the
    # batch so the product stays under the cap instead of dying mid-batch.
    nvenc_limit = _probe_nvenc_sessions(config.render.rgb_encoder)
    sessions_needed = args.batch_size * len(cameras)
    if sessions_needed > nvenc_limit:
        clamped = max(1, nvenc_limit // len(cameras))
        print(
            f"NVENC cap is {nvenc_limit} sessions; batch_size {args.batch_size} "
            f"x {len(cameras)} camera(s) needs {sessions_needed}. "
            f"Clamping batch_size to {clamped}.",
            flush=True,
        )
        args.batch_size = clamped
    episodes = discover_episodes(
        config.output_root,
        config.scene.name,
        seed_start=args.seed_start,
        num_episodes=args.num_episodes,
    )
    pending = [
        episode
        for episode in episodes
        if args.overwrite
        or not outputs_complete(config.output_root, episode.run_id, list(cameras))
    ]
    print(
        f"Discovered {len(episodes)} episode(s); rendering {len(pending)}, "
        f"skipping {len(episodes) - len(pending)} complete episode(s); "
        f"batch size {args.batch_size}",
        flush=True,
    )
    if not pending:
        return 0

    from phy_data_gen.app import AppOptions, launch_app

    app = launch_app(
        AppOptions(device=args.device, viz="none", rendering_mode=args.rendering_mode),
        force_enable_cameras=True,
    )
    started = time.monotonic()
    try:
        completed = 0
        batches = list(_batches(pending, args.batch_size))
        for batch_index, batch in enumerate(batches, start=1):
            batch_started = time.monotonic()
            frame_count = _render_batch(
                batch,
                config,
                cameras,
                args.frame_stride,
                args.env_spacing,
                app,
            )
            completed += len(batch)
            elapsed = time.monotonic() - batch_started
            total_elapsed = time.monotonic() - started
            rate = completed / total_elapsed
            eta = (len(pending) - completed) / rate if rate else 0.0
            print(
                f"[batch {batch_index}/{len(batches)}] "
                f"{len(batch)} episode(s), {frame_count} frames each in {elapsed:.1f}s; "
                f"completed {completed}/{len(pending)}, {rate:.3f} eps/s, "
                f"ETA {eta / 60:.1f} min",
                flush=True,
            )
    except KeyboardInterrupt:
        print("Interrupted; completed outputs will be skipped on the next run.")
        return 130
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
