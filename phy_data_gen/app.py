"""Tyro-managed Isaac Lab application launcher options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import tyro


@dataclass
class AppOptions:
    """Isaac Lab runtime options exposed through Tyro."""

    headless: bool = False
    livestream: int = -1
    enable_cameras: bool = False
    xr: bool = False
    device: str = "cuda:0"
    viz: Annotated[
        str | None,
        tyro.conf.arg(aliases=("--visualizer",)),
    ] = None
    verbose: bool = False
    info: bool = False
    experience: str = ""
    deterministic: bool = False
    rendering_mode: str = "balanced"
    kit_args: str = ""
    max_visible_envs: int | None = None

    def launcher_args(self, force_enable_cameras: bool = False) -> dict:
        visualizer, explicit, disable_all = _parse_visualizers(self.viz)
        return {
            "headless": self.headless,
            "headless_explicit": self.headless,
            "livestream": self.livestream,
            "enable_cameras": self.enable_cameras or force_enable_cameras,
            "xr": self.xr,
            "device": self.device,
            "device_explicit": self.device != "cuda:0",
            "visualizer": visualizer,
            "visualizer_explicit": explicit,
            "visualizer_disable_all": disable_all,
            "verbose": self.verbose,
            "info": self.info,
            "experience": self.experience,
            "deterministic": self.deterministic,
            "rendering_mode": self.rendering_mode,
            "rendering_mode_explicit": self.rendering_mode != "balanced",
            "kit_args": self.kit_args,
            "max_visible_envs": self.max_visible_envs,
        }


def _parse_visualizers(value: str | None) -> tuple[list[str] | None, bool, bool]:
    if value is None:
        return None, False, False
    names = [item.strip().lower() for item in value.split(",")]
    valid = {"kit", "newton", "rerun", "viser", "none"}
    if any(not item for item in names) or any(item not in valid for item in names):
        raise ValueError(f"Invalid visualizer list: {value}")
    if "none" in names:
        if len(names) != 1:
            raise ValueError("Visualizer 'none' cannot be combined with other values")
        return None, True, True
    return list(dict.fromkeys(names)), True, False


def launch_app(options: AppOptions, force_enable_cameras: bool = False):
    """Launch Isaac Sim and return its ``SimulationApp`` instance."""

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(options.launcher_args(force_enable_cameras))
    return launcher.app


def main() -> None:
    options = tyro.cli(AppOptions)
    simulation_app = launch_app(options)
    try:
        from isaaclab.sim import SimulationCfg, SimulationContext

        sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device=options.device))
        sim.reset()
        for _ in range(10):
            sim.step(render=False)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
