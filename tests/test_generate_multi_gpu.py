import importlib.util
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
