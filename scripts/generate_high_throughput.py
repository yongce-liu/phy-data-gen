#!/usr/bin/env python3
"""High-throughput billiards generator — maximize episode throughput on RTX 5090.

Pre-computes all episode specs (pure pxr, no Isaac runtime) before launching
the simulator, then runs the simulation loop uninterrupted.  Tracks throughput
(eps/min), ETA, and validation pass rate.

Usage:
    # Benchmark 10 episodes
    python scripts/generate_high_throughput.py \\
        --config configs/billiards_high_throughput.yaml \\
        --num-episodes 10

    # Full 50 h run = 36000 episodes
    python scripts/generate_high_throughput.py \\
        --config configs/billiards_high_throughput.yaml \\
        --num-episodes 36000 --device cuda:0

    # Resume after interruption
    python scripts/generate_high_throughput.py \\
        --config configs/billiards_high_throughput.yaml \\
        --num-episodes 36000 --resume

    # Multi-GPU sharding (one terminal per GPU)
    python scripts/generate_high_throughput.py --num-episodes 18000 --device cuda:0 --worker-id 0 &
    python scripts/generate_high_throughput.py --num-episodes 18000 --device cuda:1 --worker-id 1 &
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
import yaml


@dataclass
class HighThroughputArgs:
    """Generate a large number of billiards episodes with maximum throughput."""

    config: Path
    num_episodes: int
    device: str = "cuda:0"
    worker_id: int = 0
    no_frames: bool = False
    resume: bool = False
    seed_start: int | None = None
    rendering_mode: str = "balanced"
    rgb_encoder: str | None = None  # override config; "libx264" avoids NVENC cap


def _nvenc_available(rgb_encoder: str, device_index: int = 0) -> bool:
    """Return whether a fresh NVENC encoder session can open right now.

    Consumer NVIDIA drivers cap concurrent NVENC sessions (8 on RTX 50-series).
    When another process (e.g. an offline render) holds all sessions, a new
    encoder fails at startup and the writer deadlocks on the full pipe. Probe
    by opening one real encoder; fall back to libx264 on failure.
    """

    if rgb_encoder != "h264_nvenc":
        return True

    import subprocess
    import tempfile

    probe_dir = Path(tempfile.mkdtemp(prefix="nvenc_probe_"))
    src = probe_dir / "src.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=1:size=128x128:rate=10",
                "-frames:v", "2", "-c:v", "libx264", str(src),
            ],
            check=True,
            capture_output=True,
        )
        proc = subprocess.Popen(
            [
                "ffmpeg", "-loglevel", "error", "-y",
                "-i", str(src),
                "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                "-rc", "vbr", "-cq", "18", "-b:v", "0",
                "-gpu", str(device_index),
                str(probe_dir / "out.mp4"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok = proc.wait(timeout=30) == 0
    except Exception:
        ok = False
    finally:
        for path in probe_dir.iterdir():
            path.unlink(missing_ok=True)
        probe_dir.rmdir()
    return ok


def _find_last_episode(output_root: Path, min_seed: int) -> int:
    """Return the highest validated seed, or min_seed - 1."""
    physics_dir = output_root / "physics"
    if not physics_dir.is_dir():
        return min_seed - 1
    max_seed = min_seed - 1
    for entry in physics_dir.iterdir():
        if not entry.is_dir():
            continue
        validation = entry / "validation.json"
        if not validation.is_file():
            continue
        try:
            data = json.loads(validation.read_text())
            if data.get("passed"):
                parts = entry.name.rsplit("_", 1)
                if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                    s = int(parts[1])
                    if s > max_seed:
                        max_seed = s
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return max_seed


def _load_existing_scene(
    scene_dir: Path,
    scene_name: str,
    run_id: str,
    seed: int,
):
    """Load a matching pre-computed scene without rebuilding its USD layer."""
    from phy_data_gen.schemas import EpisodeSpec

    scene_path = scene_dir / f"{scene_name}.usda"
    spec_path = scene_dir / "episode_spec.json"
    if not scene_path.is_file() or not spec_path.is_file():
        return None

    try:
        spec = EpisodeSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if spec.episode_id != run_id or spec.seed != seed:
        return None
    return spec, scene_path


def _format_duration(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main() -> int:
    args = tyro.cli(HighThroughputArgs)
    tag = f"[W{args.worker_id}]"

    base_config = yaml.safe_load(args.config.open("r"))
    output_root = Path(base_config["output_root"])
    config_seed = int(base_config["seed"])
    seed = args.seed_start if args.seed_start is not None else config_seed

    # NVENC fallback: if the configured encoder is h264_nvenc but the GPU's
    # NVENC session cap is already exhausted (e.g. a concurrent offline render),
    # fall back to CPU libx264 so generation proceeds without deadlocking.
    configured_encoder = str(base_config.get("render", {}).get("rgb_encoder", "h264_nvenc"))
    effective_encoder = args.rgb_encoder or configured_encoder
    if effective_encoder == "h264_nvenc":
        gpu_index = int(args.device.split(":")[-1]) if ":" in args.device else 0
        if not _nvenc_available(effective_encoder, gpu_index):
            print(
                f"{tag} NVENC session cap exhausted; falling back to libx264 "
                "(CPU encoding) to avoid encoder deadlock",
                flush=True,
            )
            effective_encoder = "libx264"
    if effective_encoder != configured_encoder:
        base_config.setdefault("render", {})["rgb_encoder"] = effective_encoder
        print(f"{tag} Using rgb_encoder={effective_encoder}", flush=True)

    # Resume support.
    if args.resume:
        last_seed = _find_last_episode(output_root, seed)
        start_seed = max(seed, last_seed + 1)
        print(f"{tag} Resume: starting seed {start_seed} (last OK: {last_seed})")
    else:
        start_seed = seed

    remaining = args.num_episodes - (start_seed - seed)
    if remaining <= 0:
        print(f"{tag} All episodes done. Exiting.")
        return 0

    print(f"{tag} {'=' * 54}")
    print(f"{tag}  PHY-DATA-GEN  HIGH-THROUGHPUT  WORKER")
    print(f"{tag}  Episodes: {remaining}  |  Device: {args.device}")
    print(f"{tag}  Frames:   {'OFF' if args.no_frames else 'ON'}")
    print(f"{tag} {'=' * 54}")

    # ==================================================================
    # Phase 1 — Launch Isaac Sim before importing pxr/Isaac modules
    # ==================================================================
    from phy_data_gen.app import AppOptions, launch_app

    app_opts = AppOptions(
        device=args.device,
        viz="none",
        rendering_mode=args.rendering_mode,
    )
    simulation_app = launch_app(app_opts, force_enable_cameras=not args.no_frames)
    print(f"{tag} Isaac Sim launched", flush=True)

    # ==================================================================
    # Phase 2 — Pre-compute all episode specs
    # ==================================================================
    import dataclasses
    from phy_data_gen.config import load_config
    from phy_data_gen.dataset import default_run_id
    from phy_data_gen.episode import create_episode_spec, save_episode_spec, select_template_path
    from phy_data_gen.scene import build_scene

    t_pre = time.monotonic()
    prepped: list = []  # (ep_seed, episode_id, SceneConfig, EpisodeSpec, scene_path, render_cfg)
    reused_scenes = 0
    base_cfg = load_config(args.config)

    for offset in range(remaining):
        ep_seed = start_seed + offset
        cfg = dataclasses.replace(base_cfg, seed=ep_seed)

        template_path = select_template_path(cfg)
        run_id = default_run_id(cfg.scene.name, template_path, ep_seed)

        scene_dir = output_root / "scene" / run_id
        existing = (
            _load_existing_scene(scene_dir, cfg.scene.name, run_id, ep_seed)
            if args.resume
            else None
        )
        if existing is not None:
            spec, scene_path = existing
            reused_scenes += 1
        else:
            spec = create_episode_spec(cfg, episode_id=run_id)
            scene_dir.mkdir(parents=True, exist_ok=True)
            save_episode_spec(spec, scene_dir / "episode_spec.json")
            scene_path = build_scene(
                episode_spec=spec,
                scene=cfg.scene,
                output_path=scene_dir / f"{cfg.scene.name}.usda",
            )
        prepped.append((ep_seed, run_id, cfg.scene, spec, scene_path, cfg.render))

        if (offset + 1) % 500 == 0:
            pct = 100 * (offset + 1) // remaining
            print(f"{tag}  Pre-computing: {offset + 1}/{remaining} ({pct}%)", flush=True)

    t_pre = time.monotonic() - t_pre
    print(f"{tag} Pre-computed {len(prepped)} specs in {_format_duration(t_pre)}"
          f" ({len(prepped) / t_pre:.0f} specs/s, "
          f"reused {reused_scenes} scenes)", flush=True)

    # Warm up PhysX after all runtime imports are safe.
    from isaaclab.sim import SimulationCfg, SimulationContext

    warmup = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=args.device))
    warmup.reset()
    for _ in range(30):
        warmup.step(render=False)
    SimulationContext.clear_instance()
    print(f"{tag} PhysX warmed up", flush=True)

    # ==================================================================
    # Phase 3 — Generation loop
    # ==================================================================
    from phy_data_gen.simulation import run_simulation
    from phy_data_gen.dataset import save_camera_metadata, save_physics_annotations
    from phy_data_gen.validation import save_validation, validate_episode

    timing_hist: list[float] = []
    wall_start = time.monotonic()

    try:
        for idx, (ep_seed, run_id, scene_cfg, spec, scene_path, render_cfg) in enumerate(prepped):
            t0 = time.perf_counter()

            result = run_simulation(
                scene_path=scene_path,
                episode_spec=spec,
                simulation_app=simulation_app,
                output_root=output_root,
                cameras=scene_cfg.cameras,
                world_prim_path=scene_cfg.world_prim,
                render_width=render_cfg.width,
                render_height=render_cfg.height,
                depth_scale_meters=render_cfg.depth_scale_meters,
                rgb_encoder=render_cfg.rgb_encoder,
                capture_frames=not args.no_frames,
            )
            t1 = time.perf_counter()

            # Write outputs
            physics_dir = output_root / "physics" / run_id
            result.states.save(physics_dir / "object_states.jsonl")
            save_camera_metadata(result.camera_metadata, output_root, run_id)
            save_physics_annotations(
                recorder=result.states,
                episode_spec=spec,
                object_ids=result.object_ids,
                object_paths=result.object_paths,
                camera_names=list(scene_cfg.cameras),
                output_root=output_root,
                run_id=run_id,
            )
            is_cat01 = bool(spec.metadata.get("variant"))
            summary = validate_episode(
                result.states.records,
                require_fall=spec.object_mode == "generated_objects",
                # Category 01 is two-ball collision: require an actual
                # contact and a readable approach phase (>=6 frames, i.e.
                # >=0.2 s, of pre-collision motion).
                require_contact=is_cat01,
                min_approach_frames=6 if is_cat01 else None,
            )
            save_validation(summary, physics_dir / "validation.json")
            t2 = time.perf_counter()

            SimulationContext.clear_instance()

            elapsed_ms = (t2 - t0) * 1000
            timing_hist.append(elapsed_ms)
            if len(timing_hist) > 100:
                timing_hist.pop(0)

            wall_elapsed = time.monotonic() - wall_start
            eps_per_sec = (idx + 1) / wall_elapsed
            remaining_eps = remaining - idx - 1
            eta = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
            avg_ms = sum(timing_hist) / len(timing_hist)

            sim_ms = (t1 - t0) * 1000
            io_ms = (t2 - t1) * 1000
            passed = summary.get("passed", False)

            print(
                f"{tag} [{ep_seed}] "
                f"run={run_id} "
                f"t={elapsed_ms:.0f}ms "
                f"(sim={sim_ms:.0f} io={io_ms:.0f}) "
                f"avg={avg_ms:.0f}ms "
                f"⊘ {eps_per_sec:.2f}eps/s "
                f"ETA {_format_duration(eta)}"
                f"{'' if passed else ' FAIL'}",
                flush=True,
            )

    except KeyboardInterrupt:
        print(f"\n{tag} Interrupted by user")
    except BaseException:
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return 1

    simulation_app.close()

    # ==================================================================
    # Final report
    # ==================================================================
    wall_total = time.monotonic() - wall_start
    n_done = len(timing_hist)
    print(f"\n{tag} {'=' * 50}")
    print(f"{tag} Completed: {n_done} episodes in {_format_duration(wall_total)}")
    if n_done:
        avg = sum(timing_hist) / n_done
        eps = n_done / wall_total
        data_hours = n_done * 5 / 3600
        print(f"{tag} Throughput:   {eps:.2f} eps/s  ({eps * 60:.1f} eps/min)")
        print(f"{tag} Avg latency:  {avg:.0f} ms/episode")
        print(f"{tag} Data volume:  {data_hours:.1f} h of ball-collision data")
        # Project to 50 h
        rate = n_done * 5 / wall_total  # data hours per wall hour
        wall_50h = 50 / rate if rate > 0 else float("inf")
        print(f"{tag} To reach 50 h of data (36000 eps): ~{_format_duration(wall_50h)} wall time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
