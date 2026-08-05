"""Inspect a Cosmos template stage.

Reports the stage default prim, cameras, physics scene, rigid bodies,
collision prims and candidate dynamic objects with their world-space
bounding boxes. Prim paths from this output should be recorded into
``configs/run.yaml`` instead of being guessed from prim names.

``pxr`` and ``omni.usd`` are imported only after the app is launched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from phy_data_gen.app import AppOptions, launch_app


@dataclass
class InspectTemplateOptions:
    """Inspect notable prims in one Cosmos template."""

    template: Path
    app: tyro.conf.OmitArgPrefixes[AppOptions] = field(default_factory=AppOptions)


def _bbox_size(bbox_cache, prim):
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    size = box.GetSize()
    return (float(size[0]), float(size[1]), float(size[2]))


def inspect_stage(stage) -> None:
    """Print flags and bounding boxes for the notable prims in the stage."""

    from pxr import Usd, UsdGeom, UsdPhysics

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )

    default_prim = stage.GetDefaultPrim()
    print(f"default_prim: {default_prim.GetPath() if default_prim else None}")
    print("prims:")

    for prim in stage.Traverse():
        flags: list[str] = []

        if prim.IsA(UsdGeom.Camera):
            flags.append("camera")

        if prim.IsA(UsdPhysics.Scene):
            flags.append("physics_scene")

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            flags.append("rigid_body")

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            flags.append("collision")

        has_references = prim.HasAuthoredReferences()
        if has_references:
            flags.append("references")

        if not flags:
            continue

        line = f"  {prim.GetPath()} {flags}"

        if prim.IsA(UsdGeom.Xformable):
            try:
                size = _bbox_size(bbox_cache, prim)
                line += f" bbox={size}"
            except Exception as error:  # noqa: BLE001 - report and continue
                line += f" bbox_error={error!r}"

        print(line)


def inspect_template(template_path: Path, simulation_app) -> None:
    """Open the template stage and inspect it."""

    import omni.usd

    context = omni.usd.get_context()

    resolved = template_path.resolve()
    if not context.open_stage(str(resolved)):
        raise RuntimeError(f"Failed to open stage: {resolved}")

    for _ in range(20):
        simulation_app.update()

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"Stage not available after open: {resolved}")

    inspect_stage(stage)


def main() -> None:
    options = tyro.cli(InspectTemplateOptions)

    if not options.template.is_file():
        raise FileNotFoundError(f"Template not found: {options.template}")

    simulation_app = launch_app(options.app)
    try:
        inspect_template(options.template, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
