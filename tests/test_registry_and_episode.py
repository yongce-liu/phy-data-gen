import json
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
