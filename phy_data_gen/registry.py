"""Scan USD assets into a JSON registry.

This module only depends on ``pxr`` (USD), not on the Isaac Sim runtime,
so it can be executed with a plain ``python`` interpreter that has the USD
libraries available. Each asset is opened as its own stage; we read the
default prim, compute its world-space bounding box and probe for physics
APIs and joints.

First-version filter keeps only assets that:

* have a collision API somewhere in the hierarchy,
* are not articulated (no physics joints),
* have a maximum bounding-box dimension in ``[0.03, 0.50]`` metres.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro

_USD_SUFFIXES = {".usd", ".usda", ".usdc"}


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    usd_path: str
    bbox_size: tuple[float, float, float]
    max_dimension: float
    has_rigid_body: bool
    has_collision: bool
    articulated: bool


def find_usd_files(root: Path) -> list[Path]:
    """Return every USD file under ``root`` sorted for determinism.

    Uses ``os.walk(followlinks=True)`` so symlinked asset directories (the
    MolmoSpaces layout links ``objects/thor`` into ``.molmospaces``) are
    traversed as well; ``Path.rglob`` does not descend into symlinks.

    Cached by root path: the template root is scanned once per process and the
    result reused across episodes in a batch.
    """

    return _find_usd_files_impl(str(root.resolve()))


import functools as _functools


@_functools.lru_cache(maxsize=None)
def _find_usd_files_impl(root_str: str) -> list[Path]:
    import os
    from pathlib import Path as _Path

    root = _Path(root_str)
    matches: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            if _Path(name).suffix.lower() in _USD_SUFFIXES:
                matches.append(_Path(dirpath) / name)
    return sorted(matches)


def _bbox_size(root_prim) -> tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    bound = bbox_cache.ComputeWorldBound(root_prim)
    size = bound.ComputeAlignedBox().GetSize()
    return (float(size[0]), float(size[1]), float(size[2]))


def _has_api_in_subtree(root_prim, api) -> bool:
    from pxr import Usd

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(api):
            return True
    return False


def _is_articulated(root_prim) -> bool:
    """Return True if any physics joint is present in the subtree."""

    from pxr import Usd, UsdPhysics

    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdPhysics.Joint):
            return True
    return False


def inspect_asset(usd_path: Path, asset_root: Path) -> AssetRecord | None:
    """Open one asset stage and build an ``AssetRecord``.

    Returns ``None`` if the stage cannot be opened or has no default prim.
    """

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return None

    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        return None

    try:
        bbox_size = _bbox_size(root_prim)
    except Exception:  # noqa: BLE001 - unreadable geometry -> skip asset
        return None

    max_dimension = max(bbox_size)

    has_rigid_body = _has_api_in_subtree(root_prim, UsdPhysics.RigidBodyAPI)
    has_collision = _has_api_in_subtree(root_prim, UsdPhysics.CollisionAPI)
    articulated = _is_articulated(root_prim)

    try:
        asset_id = str(usd_path.relative_to(asset_root).with_suffix(""))
    except ValueError:
        asset_id = usd_path.stem
    asset_id = asset_id.replace("/", "__")

    return AssetRecord(
        asset_id=asset_id,
        usd_path=str(usd_path.resolve()),
        bbox_size=bbox_size,
        max_dimension=max_dimension,
        has_rigid_body=has_rigid_body,
        has_collision=has_collision,
        articulated=articulated,
    )


def is_valid_falling_asset(record: AssetRecord) -> bool:
    """First-version filter for falling-object candidates."""

    return (
        record.has_collision
        and not record.articulated
        and 0.03 <= record.max_dimension <= 0.50
    )


def build_registry(asset_root: Path) -> list[AssetRecord]:
    """Scan ``asset_root`` and return the valid asset records."""

    records: list[AssetRecord] = []
    for usd_path in find_usd_files(asset_root):
        record = inspect_asset(usd_path, asset_root)
        if record is None:
            continue
        if is_valid_falling_asset(record):
            records.append(record)
    return records


def save_registry(records: list[AssetRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(records),
        "assets": [asdict(record) for record in records],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


@dataclass
class BuildRegistryOptions:
    """Scan USD assets into a JSON registry."""

    asset_root: Path
    output: Path


def main() -> None:
    options = tyro.cli(BuildRegistryOptions)

    if not options.asset_root.is_dir():
        raise FileNotFoundError(f"Asset root not found: {options.asset_root}")
    records = build_registry(options.asset_root)
    save_registry(records, options.output)
    print(f"Wrote {len(records)} asset(s) to {options.output}")


if __name__ == "__main__":
    main()
