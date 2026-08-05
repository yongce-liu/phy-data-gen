"""Writers for the NVIDIA PhysicalAI-inspired dataset layout."""

from __future__ import annotations

import json
import re
from pathlib import Path

from phy_data_gen.recording import StateRecorder
from phy_data_gen.schemas import EpisodeSpec


def default_run_id(scene_name: str, template_path: Path, seed: int) -> str:
    """Build ``{scene_name}_{scene_hash}_{seed}`` from a template name."""

    match = re.search(r"_([0-9a-fA-F]{8})_\d+$", template_path.stem)
    scene_hash = match.group(1).lower() if match else "00000000"
    return f"{scene_name}_{scene_hash}_{seed}"


def build_camera_metadata(
    stage,
    cameras: dict[str, str],
    frame_count: int,
    width: int,
    height: int,
    depth_scale_meters: float,
) -> dict[str, dict]:
    """Return fixed-camera metadata matching the reference dataset fields."""

    from pxr import Gf, Usd, UsdGeom

    metadata: dict[str, dict] = {}
    for camera_name, prim_path in cameras.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Camera prim not found: {prim_path}")

        camera = UsdGeom.Camera(prim)
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        fx = focal_length / horizontal_aperture
        fy = focal_length / vertical_aperture

        world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_to_camera = world.GetInverse()
        transform = Gf.Transform(world_to_camera)
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        translation = transform.GetTranslation()
        pose_world2cam = [
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ]
        usd_transform = [
            float(world[row][column]) for row in range(4) for column in range(4)
        ]

        metadata[camera_name] = {
            "camera_name": camera_name,
            "frame_count": frame_count,
            "resolution": [width, height],
            "depth_encoding": {
                "codec": "ffv1",
                "pixel_format": "gray16le",
                "scale_meters": depth_scale_meters,
                "invalid_value": 0,
                "max_depth_meters": 65535 * depth_scale_meters,
            },
            "camera": {
                "focal_length": [[fx, fy] for _ in range(frame_count)],
                "principal_point": [[0.5, 0.5] for _ in range(frame_count)],
                "skew": 0.0,
                "distortion": [0.0, 0.0, 0.0, 0.0],
                "pose_world2cam": [pose_world2cam for _ in range(frame_count)],
            },
            "usd_transform": [usd_transform for _ in range(frame_count)],
        }
    return metadata


def save_camera_metadata(
    metadata: dict[str, dict], output_root: Path, run_id: str
) -> None:
    output_dir = output_root / "cameras" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, payload in metadata.items():
        output_path = output_dir / f"{camera_name}.json"
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def save_physics_annotations(
    recorder: StateRecorder,
    episode_spec: EpisodeSpec,
    object_ids: list[str],
    object_paths: list[str],
    camera_names: list[str],
    output_root: Path,
    run_id: str,
) -> None:
    """Write velocity, spin, displacement and orientation NPZ annotations."""

    import numpy as np

    object_index = {object_id: index for index, object_id in enumerate(object_ids)}
    frame_count = max((record["frame"] for record in recorder.records), default=-1) + 1
    object_count = len(object_ids)

    position = np.zeros((object_count, frame_count, 3), dtype=np.float32)
    orientation = np.zeros((object_count, frame_count, 4), dtype=np.float32)
    velocity = np.zeros((object_count, frame_count, 3), dtype=np.float32)
    spin = np.zeros((object_count, frame_count, 3), dtype=np.float32)
    for record in recorder.records:
        object_id = record["object_id"]
        index = object_index[object_id]
        frame = record["frame"]
        position[index, frame] = record["position"]
        orientation[index, frame] = record["orientation_xyzw"]
        velocity[index, frame] = record["linear_velocity"]
        spin[index, frame] = np.degrees(record["angular_velocity"])

    displacement = position - position[:, :1, :]
    colors = np.asarray(
        [
            [
                (53 * index + 67) % 256,
                (97 * index + 131) % 256,
                (193 * index + 29) % 256,
                255,
            ]
            for index in range(object_count)
        ],
        dtype=np.uint8,
    )
    common = {
        "frame_count": np.asarray(frame_count, dtype=np.int64),
        "segmentation_colors": colors,
        "object_ids": np.asarray(object_ids),
        "prim_paths": np.asarray(object_paths),
    }

    output_dir = output_root / "physics" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_name in camera_names:
        _save_vector_npz(
            output_dir / f"{camera_name}_velocity.npz",
            velocity,
            common,
            velocities=velocity,
        )
        _save_vector_npz(output_dir / f"{camera_name}_spin.npz", spin, common)
        _save_vector_npz(output_dir / f"{camera_name}_com.npz", displacement, common)
        np.savez_compressed(
            output_dir / f"{camera_name}_rot.npz",
            **common,
            data=orientation,
        )
        _save_static_metadata(
            output_dir / f"{camera_name}_static.json",
            camera_name,
            episode_spec,
            object_ids,
            object_paths,
            colors,
            frame_count,
        )


def _save_vector_npz(output_path: Path, data, common: dict, **extra) -> None:
    import numpy as np

    global_min = data.min(axis=(0, 1)) if data.size else np.zeros(3, dtype=np.float32)
    global_max = data.max(axis=(0, 1)) if data.size else np.zeros(3, dtype=np.float32)
    np.savez_compressed(
        output_path,
        **common,
        data=data,
        global_min=global_min,
        global_max=global_max,
        **extra,
    )


def _save_static_metadata(
    output_path: Path,
    camera_name: str,
    episode_spec: EpisodeSpec,
    object_ids: list[str],
    object_paths: list[str],
    colors,
    frame_count: int,
) -> None:
    objects = []
    generated_specs = {spec.object_id: spec for spec in episode_spec.objects}
    replacement_specs = {
        spec.object_id: spec for spec in episode_spec.replacements
    }
    for index, (object_id, prim_path) in enumerate(
        zip(object_ids, object_paths, strict=True)
    ):
        spec = generated_specs.get(object_id)
        replacement = replacement_specs.get(object_id)
        if replacement is not None:
            properties = {
                "source": "replacement",
                "template_path": episode_spec.template_path,
                "asset_path": replacement.asset_path,
                "scale": replacement.scale,
                "target_prim_path": replacement.target_prim_path,
            }
        elif spec is None:
            properties = {
                "source": "template",
                "template_path": episode_spec.template_path,
            }
        else:
            properties = {
                "source": "generated",
                "asset_path": spec.asset_path,
                "mass": spec.mass,
                "scale": spec.scale,
                "static_friction": spec.material.static_friction,
                "dynamic_friction": spec.material.dynamic_friction,
                "restitution": spec.material.restitution,
            }
        objects.append(
            {
                "object_id": object_id,
                "prim_path": prim_path,
                "segmentation_color": colors[index].tolist(),
                "properties": properties,
            }
        )
    payload = {
        "camera_name": camera_name,
        "frame_count": frame_count,
        "gravity": {"world_gravity": [0.0, 0.0, -9.81], "gravity_magnitude": 9.81},
        "objects": objects,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
