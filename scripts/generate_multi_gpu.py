"""Launch independent phy-data-gen workers across multiple GPUs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

import tyro
import yaml


@dataclass
class MultiGpuArgs:
    """Generate one dataset shard per worker and bind workers to GPUs."""

    config: Path
    num_episodes: int
    gpus: list[int] = field(default_factory=lambda: list(range(8)))
    workers_per_gpu: int = 1
    seed_start: int | None = None
    log_root: Path = Path("logs/multi_gpu")
    no_frames: bool = False
    rendering_mode: str = "balanced"
    poll_interval_seconds: float = 2.0
    dry_run: bool = False


@dataclass
class Worker:
    worker_id: int
    gpu: int
    seed_start: int
    num_episodes: int
    config_path: Path
    log_path: Path
    process: subprocess.Popen[bytes]
    log_file: TextIO


def split_episode_ranges(
    num_episodes: int,
    num_workers: int,
    seed_start: int,
) -> list[tuple[int, int]]:
    """Split a contiguous seed range as evenly as possible."""

    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    base_count, remainder = divmod(num_episodes, num_workers)
    ranges: list[tuple[int, int]] = []
    next_seed = seed_start
    for worker_id in range(num_workers):
        count = base_count + (1 if worker_id < remainder else 0)
        if count == 0:
            continue
        ranges.append((next_seed, count))
        next_seed += count
    return ranges


def _validate_args(args: MultiGpuArgs) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.gpus:
        raise ValueError("At least one GPU must be provided")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError(f"GPU indices must be unique: {args.gpus}")
    if any(gpu < 0 for gpu in args.gpus):
        raise ValueError(f"GPU indices must be non-negative: {args.gpus}")
    if args.workers_per_gpu <= 0:
        raise ValueError("workers_per_gpu must be positive")
    if args.poll_interval_seconds <= 0.0:
        raise ValueError("poll_interval_seconds must be positive")


def _build_command(
    config_path: Path,
    num_episodes: int,
    gpu: int,
    args: MultiGpuArgs,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "phy_data_gen.cli",
        "generate",
        "--config",
        str(config_path),
        "--num-episodes",
        str(num_episodes),
        "--device",
        f"cuda:{gpu}",
        "--viz",
        "none",
        "--rendering-mode",
        args.rendering_mode,
    ]
    if args.no_frames:
        command.append("--no-frames")
    return command


def _build_worker_environment() -> dict[str, str]:
    """Build an Isaac Sim environment without CUDA/Vulkan device remapping."""

    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment


def _terminate_workers(workers: list[Worker]) -> None:
    running = [worker for worker in workers if worker.process.poll() is None]
    for worker in running:
        try:
            os.killpg(worker.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 30.0
    while running and time.monotonic() < deadline:
        running = [worker for worker in running if worker.process.poll() is None]
        if running:
            time.sleep(0.2)

    for worker in running:
        try:
            os.killpg(worker.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _close_logs(workers: list[Worker]) -> None:
    for worker in workers:
        worker.log_file.close()


def main() -> int:
    args = tyro.cli(MultiGpuArgs)
    _validate_args(args)

    with args.config.open("r", encoding="utf-8") as file:
        base_config = yaml.safe_load(file)
    if not isinstance(base_config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {args.config}")

    config_seed = int(base_config["seed"])
    seed_start = config_seed if args.seed_start is None else args.seed_start
    gpu_assignments = [
        gpu
        for _worker_slot in range(args.workers_per_gpu)
        for gpu in args.gpus
    ]
    episode_ranges = split_episode_ranges(
        args.num_episodes,
        len(gpu_assignments),
        seed_start,
    )

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir = args.log_root / run_name
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)

    print(f"Run directory: {run_dir}")
    print(
        f"Episodes: {args.num_episodes}, workers: {len(episode_ranges)}, "
        f"GPUs: {args.gpus}, workers/GPU: {args.workers_per_gpu}"
    )

    worker_specs: list[tuple[int, int, int, int, Path, Path, list[str]]] = []
    for worker_id, (worker_seed, worker_episodes) in enumerate(episode_ranges):
        gpu = gpu_assignments[worker_id]
        worker_config = deepcopy(base_config)
        worker_config["seed"] = worker_seed
        config_path = config_dir / f"worker_{worker_id:03d}.yaml"
        config_path.write_text(
            yaml.safe_dump(worker_config, sort_keys=False),
            encoding="utf-8",
        )
        log_path = run_dir / f"worker_{worker_id:03d}_gpu_{gpu}.log"
        command = _build_command(config_path, worker_episodes, gpu, args)
        worker_specs.append(
            (
                worker_id,
                gpu,
                worker_seed,
                worker_episodes,
                config_path,
                log_path,
                command,
            )
        )
        print(
            f"Worker {worker_id:03d}: GPU {gpu}, "
            f"seeds {worker_seed}..{worker_seed + worker_episodes - 1}, "
            f"log {log_path}"
        )

    if args.dry_run:
        print("Dry run complete; no workers were launched.")
        return 0

    workers: list[Worker] = []
    try:
        for (
            worker_id,
            gpu,
            worker_seed,
            worker_episodes,
            config_path,
            log_path,
            command,
        ) in worker_specs:
            environment = _build_worker_environment()
            log_file = log_path.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                log_file.close()
                raise
            workers.append(
                Worker(
                    worker_id=worker_id,
                    gpu=gpu,
                    seed_start=worker_seed,
                    num_episodes=worker_episodes,
                    config_path=config_path,
                    log_path=log_path,
                    process=process,
                    log_file=log_file,
                )
            )

        print("All workers launched. Press Ctrl+C to stop the whole run.")
        while True:
            running = [worker for worker in workers if worker.process.poll() is None]
            failed = [
                worker
                for worker in workers
                if worker.process.poll() not in (None, 0)
            ]
            if failed:
                first = failed[0]
                print(
                    f"Worker {first.worker_id:03d} on GPU {first.gpu} failed "
                    f"with exit code {first.process.returncode}. "
                    f"See {first.log_path}",
                    file=sys.stderr,
                )
                _terminate_workers(workers)
                return first.process.returncode or 1
            if not running:
                print(f"All workers completed successfully. Logs: {run_dir}")
                return 0
            time.sleep(args.poll_interval_seconds)
    except KeyboardInterrupt:
        print("Stopping all workers...", file=sys.stderr)
        _terminate_workers(workers)
        return 130
    except Exception:
        _terminate_workers(workers)
        raise
    finally:
        _close_logs(workers)


if __name__ == "__main__":
    raise SystemExit(main())
