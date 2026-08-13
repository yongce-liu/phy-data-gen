"""Tyro command line interface for the physics data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated

import tyro

from phy_data_gen.app import AppOptions, launch_app
from phy_data_gen.config import load_config
from phy_data_gen.dataset import default_run_id


@dataclass
class BuildRegistryCommand:
    """Scan assets into the registry JSON."""

    asset_root: Path
    output: Path


@dataclass
class PlanCommand:
    """Sample an EpisodeSpec without launching Isaac Sim."""

    config: Path
    output: Path | None = None
    episode_id: str | None = None


@dataclass
class InspectTemplateCommand:
    """Launch Isaac Sim and report notable template prims."""

    config: Path
    app: tyro.conf.OmitArgPrefixes[AppOptions] = field(default_factory=AppOptions)


@dataclass
class GenerateCommand:
    """Generate RGB, depth and physics data for one simulation run."""

    config: Path
    episode_id: str | None = None
    num_episodes: int = 1
    no_frames: bool = False
    app: tyro.conf.OmitArgPrefixes[AppOptions] = field(default_factory=AppOptions)


Command = (
    Annotated[BuildRegistryCommand, tyro.conf.subcommand(name="build-registry")]
    | Annotated[PlanCommand, tyro.conf.subcommand(name="plan")]
    | Annotated[InspectTemplateCommand, tyro.conf.subcommand(name="inspect-template")]
    | Annotated[GenerateCommand, tyro.conf.subcommand(name="generate")]
)


def _resolve_run_id(config, episode_id: str | None) -> str:
    from phy_data_gen.episode import select_template_path

    template_path = select_template_path(config)
    if episode_id:
        return episode_id
    # Procedural categories always resolve a template (minimal or backdrop);
    # keep the run ID seed-suffixed so resume/`_find_last_episode` work.
    return default_run_id(config.scene.name, template_path, config.seed)


def _run_build_registry(command: BuildRegistryCommand) -> None:
    from phy_data_gen.registry import build_registry, save_registry

    if not command.asset_root.is_dir():
        raise FileNotFoundError(f"Asset root not found: {command.asset_root}")
    records = build_registry(command.asset_root)
    save_registry(records, command.output)
    print(f"Wrote registry with {len(records)} asset(s) to {command.output}")


def _run_plan(command: PlanCommand) -> None:
    from phy_data_gen.episode import create_episode_spec, save_episode_spec

    config = load_config(command.config)
    run_id = _resolve_run_id(config, command.episode_id)
    spec = create_episode_spec(config, episode_id=run_id)
    output = command.output or (
        config.output_root / "scene" / run_id / "episode_spec.json"
    )
    save_episode_spec(spec, output)
    object_count = len(spec.objects) + len(spec.replacements)
    print(f"Wrote EpisodeSpec with {object_count} object(s) to {output}")


def _run_inspect_template(command: InspectTemplateCommand) -> None:
    simulation_app = launch_app(command.app)
    try:
        from phy_data_gen.inspect_template import inspect_template

        config = load_config(command.config)
        from phy_data_gen.episode import select_template_path

        inspect_template(select_template_path(config), simulation_app)
    finally:
        simulation_app.close()


def _run_generate(command: GenerateCommand) -> None:
    if command.num_episodes <= 0:
        raise ValueError("num_episodes must be positive")

    base_config = load_config(command.config)
    capture_frames = not command.no_frames

    simulation_app = launch_app(command.app, force_enable_cameras=capture_frames)
    try:
        for offset in range(command.num_episodes):
            config = replace(base_config, seed=base_config.seed + offset)
            if command.episode_id is None:
                run_id = _resolve_run_id(config, None)
            elif command.num_episodes == 1:
                run_id = command.episode_id
            else:
                run_id = f"{command.episode_id}_{offset:06d}"
            print(
                f"Generating run {offset + 1}/{command.num_episodes}: "
                f"{run_id} (seed={config.seed})"
            )
            _generate_one(config, run_id, capture_frames, simulation_app)
    finally:
        simulation_app.close()


def _generate_one(config, run_id: str, capture_frames: bool, simulation_app) -> None:

    from phy_data_gen.episode import create_episode_spec, save_episode_spec

    spec = create_episode_spec(config, episode_id=run_id)
    scene_dir = config.output_root / "scene" / run_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    save_episode_spec(spec, scene_dir / "episode_spec.json")

    try:
        # Import USD/Isaac modules only after the app owns the runtime.
        from phy_data_gen.scene import build_scene

        scene_path = build_scene(
            episode_spec=spec,
            scene=config.scene,
            output_path=scene_dir / f"{config.scene.name}.usda",
        )
        print(f"Compiled scene: {scene_path}")

        from phy_data_gen.simulation import run_simulation

        result = run_simulation(
            scene_path=scene_path,
            episode_spec=spec,
            simulation_app=simulation_app,
            output_root=config.output_root,
            cameras=config.scene.cameras,
            world_prim_path=config.scene.world_prim,
            render_width=config.render.width,
            render_height=config.render.height,
            depth_scale_meters=config.render.depth_scale_meters,
            rgb_encoder=config.render.rgb_encoder,
            capture_frames=capture_frames,
        )

        physics_dir = config.output_root / "physics" / run_id
        states_path = physics_dir / "object_states.jsonl"
        result.states.save(states_path)
        print(f"Recorded {len(result.states.records)} state row(s) to {states_path}")

        from phy_data_gen.dataset import (
            save_camera_metadata,
            save_physics_annotations,
        )

        save_camera_metadata(result.camera_metadata, config.output_root, run_id)
        save_physics_annotations(
            recorder=result.states,
            episode_spec=spec,
            object_ids=result.object_ids,
            object_paths=result.object_paths,
            camera_names=list(config.scene.cameras),
            output_root=config.output_root,
            run_id=run_id,
        )

        from phy_data_gen.validation import save_validation, validate_episode

        summary = validate_episode(
            result.states.records,
            require_fall=spec.object_mode == "generated_objects",
        )
        save_validation(summary, physics_dir / "validation.json")
        print(f"Validation passed={summary.get('passed')}")
        if capture_frames:
            print(
                f"Captured {result.captured_frames} synchronized frame(s) for "
                f"{len(result.rgb_paths)} camera(s)"
            )
            for camera_name in config.scene.cameras:
                print(f"RGB [{camera_name}]: {result.rgb_paths[camera_name]}")
                print(f"Depth [{camera_name}]: {result.depth_paths[camera_name]}")
    finally:
        # SimulationContext is a singleton. Clear it between runs while keeping
        # the expensive Isaac Sim application alive for batch generation.
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()


def main() -> None:
    command = tyro.cli(Command, prog="phy-data-gen")
    if isinstance(command, BuildRegistryCommand):
        _run_build_registry(command)
    elif isinstance(command, PlanCommand):
        _run_plan(command)
    elif isinstance(command, InspectTemplateCommand):
        _run_inspect_template(command)
    elif isinstance(command, GenerateCommand):
        _run_generate(command)
    else:
        raise TypeError(f"Unsupported command: {type(command)!r}")


if __name__ == "__main__":
    main()
