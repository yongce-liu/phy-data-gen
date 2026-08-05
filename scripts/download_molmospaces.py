import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import tyro
from molmospaces_resources import HFRemoteStorage as OriginHFRemoteStorage
from molmospaces_resources import R2RemoteStorage, ResourceManager

logger = logging.getLogger("molmospaces_resources")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)


class HFRemoteStorage(OriginHFRemoteStorage):
    """HFRemoteStorage with support for a custom Hugging Face endpoint."""

    def _file_url(self, path_in_repo: str) -> str:
        endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

        return (
            f"{endpoint}/datasets/{self.repo_id}/resolve/{self.revision}/{path_in_repo}"
        )


DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".molmospaces"


SOURCE_TO_VERSION = {
    "objects": {
        "mjcf": {
            "thor": "20251117",
            "objaverse": "20260131",
        },
        "usd": {
            "thor": "20260128",
            "objaverse": "20260128",
        },
    },
    "scenes": {
        "mjcf": {
            "ithor": "20251217",
            "procthor-10k-train": "20251122",
            "procthor-10k-val": "20251217",
            "procthor-10k-test": "20251121",
            "holodeck-objaverse-train": "20251217",
            "holodeck-objaverse-val": "20251217",
            "procthor-objaverse-train": "20251205",
            "procthor-objaverse-val": "20251205",
        },
        "usd": {
            "ithor": "20260121",
            "procthor-10k-train": "20260128",
            "procthor-10k-val": "20260128",
            "procthor-10k-test": "20260128",
            "procthor-objaverse-train": "20260128",
            "procthor-objaverse-val": "20260128",
            "holodeck-objaverse-train": "20260128",
            "holodeck-objaverse-val": "20260128",
        },
    },
}


TYPE_TO_PREFIX: dict[str, str] = {
    "mjcf": "mujoco",
    "usd": "isaac",
}


AssetName = Literal["thor", "objaverse"]

SceneName = Literal[
    "ithor",
    "procthor-10k-train",
    "procthor-10k-val",
    "procthor-10k-test",
    "procthor-objaverse-train",
    "procthor-objaverse-val",
    "holodeck-objaverse-train",
    "holodeck-objaverse-val",
]


@dataclass
class DownloadArgs:
    """Download MolmoSpaces assets and scenes."""

    # `mjcf` for MuJoCo / ManiSkill, `usd` for Isaac Sim.
    type: Literal["mjcf", "usd"]

    # Directory containing symlinks to extracted resources.
    install_dir: Path

    # Object asset datasets to download.
    assets: list[AssetName] = field(default_factory=lambda: ["thor"])

    # Scene datasets to download.
    scenes: list[SceneName] = field(default_factory=list)

    # Download every object and scene available for the selected type.
    download_all: bool = False

    # Directory storing downloaded and extracted versioned data.
    cache_dir: Path = DEFAULT_CACHE_DIR

    # If omitted, read HF_TOKEN from the environment.
    hf_token: str | None = None

    # Use Cloudflare R2 instead of Hugging Face.
    use_r2: bool = False


def resolve_requested_sources(
    args: DownloadArgs,
) -> tuple[list[str], list[str]]:
    available_assets = SOURCE_TO_VERSION["objects"][args.type]
    available_scenes = SOURCE_TO_VERSION["scenes"][args.type]

    if args.download_all:
        assets = list(available_assets.keys())
        scenes = list(available_scenes.keys())
    else:
        assets = list(dict.fromkeys(args.assets))
        scenes = list(dict.fromkeys(args.scenes))

    invalid_assets = sorted(set(assets) - set(available_assets))
    invalid_scenes = sorted(set(scenes) - set(available_scenes))

    if invalid_assets:
        raise ValueError(
            f"Unsupported {args.type} object datasets: {invalid_assets}. "
            f"Available: {sorted(available_assets)}"
        )

    if invalid_scenes:
        raise ValueError(
            f"Unsupported {args.type} scene datasets: {invalid_scenes}. "
            f"Available: {sorted(available_scenes)}"
        )

    return assets, scenes


def build_sources_to_version(
    data_type: str,
    assets: list[str],
    scenes: list[str],
) -> dict[str, dict[str, str]]:
    return {
        "objects": {
            dataset_id: SOURCE_TO_VERSION["objects"][data_type][dataset_id]
            for dataset_id in assets
        },
        "scenes": {
            dataset_id: SOURCE_TO_VERSION["scenes"][data_type][dataset_id]
            for dataset_id in scenes
        },
    }


def main() -> int:
    args = tyro.cli(DownloadArgs)

    if args.type not in TYPE_TO_PREFIX:
        raise ValueError(
            f"Unsupported type: {args.type}. Expected one of {sorted(TYPE_TO_PREFIX)}."
        )

    assets, scenes = resolve_requested_sources(args)

    if not assets and not scenes:
        logger.warning("No objects or scenes were selected.")
        return 0

    args.install_dir.mkdir(parents=True, exist_ok=True)

    type_cache_dir = args.cache_dir / args.type
    type_cache_dir.mkdir(parents=True, exist_ok=True)

    sources_to_version = build_sources_to_version(
        data_type=args.type,
        assets=assets,
        scenes=scenes,
    )

    logger.info("Resource type: %s", args.type)
    logger.info("Install directory: %s", args.install_dir)
    logger.info("Cache directory: %s", type_cache_dir)
    logger.info("Object datasets: %s", assets or "none")
    logger.info("Scene datasets: %s", scenes or "none")

    if args.use_r2:
        remote_storage = R2RemoteStorage(f"{TYPE_TO_PREFIX[args.type]}-thor-resources")
    else:
        remote_storage = HFRemoteStorage(
            repo_id="allenai/molmospaces",
            repo_prefix=TYPE_TO_PREFIX[args.type],
            token=args.hf_token or os.getenv("HF_TOKEN"),
        )

    manager = ResourceManager(
        remote_storage=remote_storage,
        data_type_to_source_to_version=sources_to_version,
        symlink_dir=args.install_dir,
        cache_dir=type_cache_dir,
        force_install=True,
    )

    manager.setup()

    if assets:
        logger.info("Installing object assets...")
        manager.install_all_for_data_type(
            "objects",
            skip_linking=False,
        )

    if scenes:
        logger.info("Installing scenes...")
        manager.install_all_for_data_type(
            "scenes",
            skip_linking=False,
        )

    logger.info("Installation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
