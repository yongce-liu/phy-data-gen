import importlib.util
import os
import sys
from pathlib import Path


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_multi_gpu.py"
    )
    spec = importlib.util.spec_from_file_location("generate_multi_gpu", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_episode_ranges_covers_every_seed_once() -> None:
    module = _load_script_module()

    assert module.split_episode_ranges(10, 3, 42) == [
        (42, 4),
        (46, 3),
        (49, 3),
    ]


def test_split_episode_ranges_skips_idle_workers() -> None:
    module = _load_script_module()

    assert module.split_episode_ranges(2, 8, 100) == [(100, 1), (101, 1)]


def test_build_command_selects_physical_gpu_and_disables_visualizers(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    args = module.MultiGpuArgs(config=tmp_path / "config.yaml", num_episodes=4)

    command = module._build_command(tmp_path / "worker.yaml", 2, 3, args)

    assert command[command.index("--device") + 1] == "cuda:3"
    assert command[command.index("--viz") + 1] == "none"
    assert "--headless" not in command


def test_build_worker_environment_removes_cuda_device_remapping(
    monkeypatch,
) -> None:
    module = _load_script_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("PHY_DATA_GEN_TEST_VALUE", "kept")

    environment = module._build_worker_environment()

    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert environment["PHY_DATA_GEN_TEST_VALUE"] == "kept"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
