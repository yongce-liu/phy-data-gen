from pathlib import Path

from phy_data_gen.schemas import EpisodeSpec
from scripts.generate_high_throughput import _load_existing_scene


def _spec(run_id: str = "billiards_deadbeef_42", seed: int = 42) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=run_id,
        seed=seed,
        template_path="/tmp/template.usda",
        backend="physx",
        object_mode="replace_assets",
        duration_seconds=5.0,
        physics_dt=1.0 / 240.0,
        render_fps=30,
        objects=[],
    )


def test_load_existing_scene_reuses_matching_scene(tmp_path: Path) -> None:
    scene_dir = tmp_path / "billiards_deadbeef_42"
    scene_dir.mkdir()
    scene_path = scene_dir / "billiards.usda"
    scene_path.write_text("#usda 1.0\n", encoding="utf-8")
    (scene_dir / "episode_spec.json").write_text(
        _spec().model_dump_json(), encoding="utf-8"
    )

    loaded = _load_existing_scene(
        scene_dir, "billiards", "billiards_deadbeef_42", 42
    )

    assert loaded is not None
    assert loaded[0] == _spec()
    assert loaded[1] == scene_path


def test_load_existing_scene_rejects_missing_or_mismatched_spec(
    tmp_path: Path,
) -> None:
    scene_dir = tmp_path / "billiards_deadbeef_42"
    scene_dir.mkdir()
    (scene_dir / "billiards.usda").write_text("#usda 1.0\n", encoding="utf-8")

    assert (
        _load_existing_scene(scene_dir, "billiards", "billiards_deadbeef_42", 42)
        is None
    )

    (scene_dir / "episode_spec.json").write_text(
        _spec(seed=43).model_dump_json(), encoding="utf-8"
    )
    assert (
        _load_existing_scene(scene_dir, "billiards", "billiards_deadbeef_42", 42)
        is None
    )
