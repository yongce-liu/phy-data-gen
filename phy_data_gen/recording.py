"""Stream synchronized RGB and metric depth frames directly into videos.

RGB and depth annotators share one render product per camera and all cameras
are captured after a single Kit update. Raw frames are piped to long-lived
FFmpeg processes, avoiding per-frame PNG encoding and intermediate image I/O.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class StateRecorder:
    """Accumulate per-frame object states and dump them to JSONL."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(
        self,
        frame: int,
        timestamp: float,
        object_id: str,
        position: Sequence[float],
        orientation_xyzw: Sequence[float],
        linear_velocity: Sequence[float],
        angular_velocity: Sequence[float],
    ) -> None:
        self.records.append(
            {
                "frame": int(frame),
                "timestamp": float(timestamp),
                "object_id": object_id,
                "position": [float(v) for v in position],
                "orientation_xyzw": [float(v) for v in orientation_xyzw],
                "linear_velocity": [float(v) for v in linear_velocity],
                "angular_velocity": [float(v) for v in angular_velocity],
            }
        )

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            for record in self.records:
                file.write(json.dumps(record, ensure_ascii=False))
                file.write("\n")


class _VideoPipe:
    """Write fixed-size raw frames into one persistent FFmpeg process."""

    def __init__(self, command: list[str], output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.output_path.unlink()

        # ffmpeg stderr goes to a sidecar log file instead of a pipe. A pipe
        # with no reader fills up (~64 KB) and blocks ffmpeg forever once it
        # starts writing, which deadlocks ``close()``/``wait()`` when many
        # encoders fail at once (e.g. NVENC concurrent-session exhaustion).
        self._stderr_path = self.output_path.with_suffix(
            self.output_path.suffix + ".stderr.log"
        )
        self._stderr_path.unlink(missing_ok=True)
        self._stderr_file = self._stderr_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=self._stderr_file,
                bufsize=16 * 1024 * 1024,
            )
        except BaseException:
            self._stderr_file.close()
            raise

    def write(self, frame_bytes: bytes) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"FFmpeg stdin is unavailable for {self.output_path}")
        try:
            self._process.stdin.write(frame_bytes)
        except BrokenPipeError as error:
            stderr = self._read_stderr()
            raise RuntimeError(
                f"FFmpeg stopped while writing {self.output_path}: {stderr}"
            ) from error

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait()
        self._stderr_file.close()
        stderr = self._read_stderr()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed for {self.output_path} with code {return_code}: {stderr}"
            )

    def terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait()
        self._stderr_file.close()

    def _read_stderr(self) -> str:
        try:
            content = self._stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return content.strip()


@dataclass
class _CameraCapture:
    render_product: object
    rgb_annotator: object
    depth_annotator: object
    rgb_pipe: _VideoPipe
    depth_pipe: _VideoPipe


class FrameRecorder:
    """Capture multiple cameras and modalities with one render per frame."""

    def __init__(
        self,
        output_root: Path,
        run_id: str,
        cameras: dict[str, str],
        width: int,
        height: int,
        fps: int,
        depth_scale_meters: float,
        rgb_encoder: str,
    ) -> None:
        if depth_scale_meters <= 0.0:
            raise ValueError("depth_scale_meters must be positive")
        if rgb_encoder not in {"h264_nvenc", "libx264"}:
            raise ValueError(f"Unsupported RGB encoder: {rgb_encoder}")

        self.output_root = output_root.resolve()
        self.run_id = run_id
        self.cameras = cameras
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_scale_meters = depth_scale_meters
        self.rgb_encoder = rgb_encoder
        self._captures: dict[str, _CameraCapture] = {}
        self.captured_frames = 0
        # Pre-allocate depth buffer to reduce allocation churn.
        self._depth_buffer = np.zeros((height, width), dtype=np.uint16)

    @property
    def rgb_paths(self) -> dict[str, Path]:
        return {
            name: self.output_root / "videos" / self.run_id / f"{name}.mp4"
            for name in self.cameras
        }

    @property
    def depth_paths(self) -> dict[str, Path]:
        return {
            name: self.output_root / "depths" / self.run_id / f"{name}.mkv"
            for name in self.cameras
        }

    def initialize(self) -> None:
        import omni.replicator.core as rep

        rep.orchestrator.set_capture_on_play(False)
        for camera_name, camera_prim_path in self.cameras.items():
            render_product = rep.create.render_product(
                camera_prim_path,
                resolution=(self.width, self.height),
            )
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            rgb_annotator.attach(render_product)
            depth_annotator.attach(render_product)

            rgb_path = self.rgb_paths[camera_name]
            depth_path = self.depth_paths[camera_name]
            self._captures[camera_name] = _CameraCapture(
                render_product=render_product,
                rgb_annotator=rgb_annotator,
                depth_annotator=depth_annotator,
                rgb_pipe=_VideoPipe(self._rgb_command(rgb_path), rgb_path),
                depth_pipe=_VideoPipe(self._depth_command(depth_path), depth_path),
            )

    def capture(self, simulation_app, update: bool = True) -> None:
        """Capture attached annotators, optionally advancing Kit first.

        Batch replay attaches several recorders to one stage, advances Kit
        once, then reads every recorder with ``update=False``.
        """

        # One Kit update renders every attached render product.
        if update:
            simulation_app.update()
        for camera_name, capture in self._captures.items():
            rgb = np.asarray(capture.rgb_annotator.get_data())
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise RuntimeError(
                    f"RGB annotator for {camera_name} returned shape {rgb.shape}"
                )
            rgb_frame = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
            capture.rgb_pipe.write(rgb_frame.tobytes())

            depth = np.asarray(capture.depth_annotator.get_data(), dtype=np.float32)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            if depth.shape != (self.height, self.width):
                raise RuntimeError(
                    f"Depth annotator for {camera_name} returned shape {depth.shape}"
                )
            # Reuse pre-allocated buffer: fill with 0 for invalid pixels,
            # scale valid (finite, positive) depths into the buffer.
            self._depth_buffer.fill(0)
            valid = np.isfinite(depth) & (depth > 0.0)
            scaled = np.rint(depth[valid] / self.depth_scale_meters).clip(1, 65535).astype(np.uint16)
            self._depth_buffer[valid] = scaled
            capture.depth_pipe.write(np.ascontiguousarray(self._depth_buffer).tobytes())

        self.captured_frames += 1

    def finalize(self) -> None:
        errors: list[Exception] = []
        for capture in self._captures.values():
            capture.rgb_annotator.detach()
            capture.depth_annotator.detach()
            capture.render_product.destroy()
            for pipe in (capture.rgb_pipe, capture.depth_pipe):
                try:
                    pipe.close()
                except Exception as error:  # noqa: BLE001 - close all encoders
                    errors.append(error)
        self._captures.clear()
        if errors:
            raise RuntimeError("; ".join(str(error) for error in errors))

    def abort(self) -> None:
        for capture in self._captures.values():
            capture.rgb_annotator.detach()
            capture.depth_annotator.detach()
            capture.render_product.destroy()
            capture.rgb_pipe.terminate()
            capture.depth_pipe.terminate()
        self._captures.clear()

    def _raw_input_args(self, pixel_format: str) -> list[str]:
        return [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-an",
        ]

    def _rgb_command(self, output_path: Path) -> list[str]:
        command = self._raw_input_args("rgb24")
        if self.rgb_encoder == "h264_nvenc":
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-tune",
                    "hq",
                    "-rc",
                    "vbr",
                    "-cq",
                    "18",
                    "-b:v",
                    "0",
                ]
            )
        else:
            command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
        command.extend(
            ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)]
        )
        return command

    def _depth_command(self, output_path: Path) -> list[str]:
        return [
            *self._raw_input_args("gray16le"),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-g",
            "1",
            "-slicecrc",
            "1",
            "-pix_fmt",
            "gray16le",
            str(output_path),
        ]
