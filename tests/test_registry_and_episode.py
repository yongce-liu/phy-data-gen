import json
from dataclasses import replace
from pathlib import Path

import pytest

from phy_data_gen.config import RenderConfig, RunConfig, SceneConfig, SimulationConfig
from phy_data_gen.episode import (
    create_episode_spec,
    load_registry,
    select_template_path,
)
from phy_data_gen.registry import AssetRecord, save_registry


def _config(tmp_path: Path, registry_path: Path) -> RunConfig:
    return RunConfig(
        template_root=tmp_path / "templates",
        asset_root=tmp_path / "assets",
        registry_path=registry_path,
        output_root=tmp_path / "outputs",
        backend="physx",
        seed=42,
        num_objects=1,
        scene=SceneConfig(
            name="objects_falling",
            object_mode="generated_objects",
            world_prim="/World",
            physics_scene_prim="/PhysicsScene",
            ground_prim="/World/Ground",
            cameras={},
            dynamic_prims=(),
        ),
        simulation=SimulationConfig(
            physics_dt=1.0 / 480.0,
            render_fps=30,
            duration_seconds=5.0,
        ),
        render=RenderConfig(
            width=640,
            height=360,
            depth_scale_meters=0.001,
            rgb_encoder="h264_nvenc",
        ),
    )


def test_registry_contains_assets_only(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    records = [
        AssetRecord(
            asset_id="cup",
            usd_path="/assets/cup.usda",
            bbox_size=(0.1, 0.1, 0.2),
            max_dimension=0.2,
            has_rigid_body=True,
            has_collision=True,
            articulated=False,
        ),
    ]

    save_registry(records, registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    assert payload["count"] == 1
    assert payload["assets"][0]["asset_id"] == "cup"
    assert "type" not in payload["assets"][0]


def test_episode_selects_scene_from_template_root(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    template_path = tmp_path / "templates" / "falling_1.usda"
    template_path.parent.mkdir()
    template_path.touch()
    save_registry(
        [
            AssetRecord(
                asset_id="cup",
                usd_path="/assets/cup.usda",
                bbox_size=(0.1, 0.1, 0.2),
                max_dimension=0.2,
                has_rigid_body=True,
                has_collision=True,
                articulated=False,
            ),
        ],
        registry_path,
    )
    config = _config(tmp_path, registry_path)

    assert load_registry(registry_path)[0]["asset_id"] == "cup"
    assert select_template_path(config) == template_path

    spec = create_episode_spec(config)
    assert spec.template_path == str(template_path.resolve())
    assert spec.objects[0].asset_path == "/assets/cup.usda"


def test_select_template_path_requires_templates(tmp_path: Path) -> None:
    config = _config(tmp_path, tmp_path / "registry.json")
    with pytest.raises(FileNotFoundError, match="No USD scene templates found"):
        select_template_path(config)


def test_template_seed_keeps_scene_fixed_when_asset_seed_changes(tmp_path: Path) -> None:
    first = tmp_path / "templates" / "first.usda"
    second = tmp_path / "templates" / "second.usda"
    first.parent.mkdir()
    first.touch()
    second.touch()
    config = replace(
        _config(tmp_path, tmp_path / "registry.json"),
        template_seed=7,
    )

    assert select_template_path(config) == select_template_path(
        replace(config, seed=config.seed + 1)
    )


def test_template_dynamics_episode_does_not_generate_objects(tmp_path: Path) -> None:
    template_path = tmp_path / "templates" / "billiards.usda"
    template_path.parent.mkdir()
    template_path.touch()
    config = _config(tmp_path, tmp_path / "registry.json")
    config = replace(
        config,
        scene=replace(config.scene, object_mode="template_dynamics"),
    )

    spec = create_episode_spec(config)

    assert spec.object_mode == "template_dynamics"
    assert spec.objects == []


def test_replace_assets_samples_local_assets_for_template_rigid_bodies(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    template_path = tmp_path / "templates" / "scene.usda"
    template_path.parent.mkdir()
    template_stage = Usd.Stage.CreateNew(str(template_path))
    world = UsdGeom.Xform.Define(template_stage, "/World").GetPrim()
    template_stage.SetDefaultPrim(world)
    body = UsdGeom.Xform.Define(template_stage, "/World/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdGeom.Cube.Define(template_stage, "/World/Body/Geometry")
    template_stage.GetRootLayer().Save()

    asset_path = tmp_path / "assets" / "cup.usda"
    asset_path.parent.mkdir()
    asset_stage = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset_stage, "/Cup").GetPrim()
    asset_stage.SetDefaultPrim(asset_root)
    collider = UsdGeom.Cube.Define(asset_stage, "/Cup/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    asset_stage.GetRootLayer().Save()

    registry_path = tmp_path / "registry.json"
    save_registry(
        [
            AssetRecord(
                asset_id="cup",
                usd_path=str(asset_path),
                bbox_size=(2.0, 2.0, 2.0),
                max_dimension=2.0,
                has_rigid_body=False,
                has_collision=True,
                articulated=False,
            )
        ],
        registry_path,
    )
    config = _config(tmp_path, registry_path)
    config = replace(
        config,
        scene=replace(config.scene, object_mode="replace_assets"),
    )

    spec = create_episode_spec(config)

    assert spec.objects == []
    assert len(spec.replacements) == 1
    replacement = spec.replacements[0]
    assert replacement.object_id == "Body"
    assert replacement.target_prim_path == "/World/Body"
    assert replacement.asset_path == str(asset_path.resolve())
    assert replacement.scale == pytest.approx(1.0)
    assert replacement.asset_rigid_body_path is None
