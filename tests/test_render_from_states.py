import json
from pathlib import Path

import pytest

from scripts.render_from_states import (
    _batches,
    _prefixed_path,
    _remove_outputs,
    _seed_from_run_id,
    discover_episodes,
    load_state_frames,
    outputs_complete,
)


def test_seed_from_run_id() -> None:
    assert _seed_from_run_id("billiards_deadbeef_42") == 42
    assert _seed_from_run_id("billiards_deadbeef_-2") == -2
    assert _seed_from_run_id("invalid") is None


def test_prefixed_path_maps_world_namespace() -> None:
    assert _prefixed_path("/World/envs/env_2", "/World") == "/World/envs/env_2"
    assert (
        _prefixed_path("/World/envs/env_2", "/World/CueBall")
        == "/World/envs/env_2/CueBall"
    )
    with pytest.raises(ValueError, match="below /World"):
        _prefixed_path("/World/envs/env_2", "/Other/CueBall")


def test_batches_preserve_order_and_partial_tail() -> None:
    assert list(_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_load_state_frames_groups_and_strides(tmp_path: Path) -> None:
    states_path = tmp_path / "object_states.jsonl"
    records = [
        {
            "frame": frame,
            "object_id": object_id,
            "position": [float(frame), 0.0, 0.0],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        for frame in range(3)
        for object_id in ("a", "b")
    ]
    states_path.write_text(
        "\n".join(json.dumps(record) for record in reversed(records)) + "\n",
        encoding="utf-8",
    )

    frames = load_state_frames(states_path, frame_stride=2)

    assert [[record["frame"] for record in frame] for frame in frames] == [
        [0, 0],
        [2, 2],
    ]


def test_discover_episodes_filters_validation_and_seed(tmp_path: Path) -> None:
    for seed, passed in ((41, True), (42, True), (43, False)):
        run_id = f"billiards_deadbeef_{seed}"
        physics_dir = tmp_path / "physics" / run_id
        scene_dir = tmp_path / "scene" / run_id
        physics_dir.mkdir(parents=True)
        scene_dir.mkdir(parents=True)
        (physics_dir / "validation.json").write_text(
            json.dumps({"passed": passed}), encoding="utf-8"
        )
        (physics_dir / "object_states.jsonl").write_text("{}\n", encoding="utf-8")
        (scene_dir / "billiards.usda").write_text("#usda 1.0\n", encoding="utf-8")
        (scene_dir / "episode_spec.json").write_text("{}", encoding="utf-8")

    episodes = discover_episodes(
        tmp_path, "billiards", seed_start=42, num_episodes=10
    )

    assert [episode.seed for episode in episodes] == [42]


def test_outputs_complete_requires_all_nonempty_files(tmp_path: Path) -> None:
    video = tmp_path / "videos" / "run" / "Top.mp4"
    depth = tmp_path / "depths" / "run" / "Top.mkv"
    video.parent.mkdir(parents=True)
    depth.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    depth.write_bytes(b"depth")

    assert outputs_complete(tmp_path, "run", ["Top"])
    depth.write_bytes(b"")
    assert not outputs_complete(tmp_path, "run", ["Top"])


def test_remove_outputs_deletes_only_selected_camera(tmp_path: Path) -> None:
    for camera_name in ("Top", "Side"):
        video = tmp_path / "videos" / "run" / f"{camera_name}.mp4"
        depth = tmp_path / "depths" / "run" / f"{camera_name}.mkv"
        video.parent.mkdir(parents=True, exist_ok=True)
        depth.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        depth.write_bytes(b"depth")

    _remove_outputs(tmp_path, "run", ["Top"])

    assert not (tmp_path / "videos" / "run" / "Top.mp4").exists()
    assert not (tmp_path / "depths" / "run" / "Top.mkv").exists()
    assert (tmp_path / "videos" / "run" / "Side.mp4").exists()
    assert (tmp_path / "depths" / "run" / "Side.mkv").exists()


def test_load_state_frames_rejects_invalid_stride(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame_stride"):
        load_state_frames(tmp_path / "missing.jsonl", frame_stride=0)
