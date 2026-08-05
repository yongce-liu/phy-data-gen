"""Minimal episode validation over recorded object states.

First-version checks only:

* every recorded value is finite,
* at least one object fell more than 0.2 m (Z axis is up),
* at least one object reached a speed above 0.1 m/s,
* the maximum observed speed stays below 50 m/s.

Contact and visual-visibility checks are intentionally out of scope for
now. This module has no runtime dependency.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

_MIN_FALL_METERS = 0.2
_MIN_SPEED = 0.1
_MAX_SPEED = 50.0


def _is_finite(values) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _speed(velocity) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in velocity))


def validate_episode(records: list[dict]) -> dict:
    """Validate per-frame state records and return a summary dict."""

    if not records:
        return {
            "finite": False,
            "moved": False,
            "fell": False,
            "max_speed_ok": False,
            "passed": False,
            "reason": "no records",
        }

    finite = True
    max_speed = 0.0
    moved = False
    fell = False

    # Track first/last Z per object to detect falling.
    first_z: dict[str, float] = {}
    last_z: dict[str, float] = defaultdict(float)

    for record in records:
        position = record["position"]
        velocity = record["linear_velocity"]

        if not (
            _is_finite(position)
            and _is_finite(velocity)
            and _is_finite(record["orientation_xyzw"])
            and _is_finite(record["angular_velocity"])
        ):
            finite = False

        speed = _speed(velocity)
        max_speed = max(max_speed, speed)
        if speed > _MIN_SPEED:
            moved = True

        object_id = record["object_id"]
        z = float(position[2])
        if object_id not in first_z:
            first_z[object_id] = z
        last_z[object_id] = z

    for object_id, start_z in first_z.items():
        if start_z - last_z[object_id] > _MIN_FALL_METERS:
            fell = True
            break

    max_speed_ok = max_speed < _MAX_SPEED
    passed = finite and moved and fell and max_speed_ok

    return {
        "finite": bool(finite),
        "moved": bool(moved),
        "fell": bool(fell),
        "max_speed_ok": bool(max_speed_ok),
        "max_speed": float(max_speed),
        "passed": bool(passed),
    }


def save_validation(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_states(states_path: Path) -> list[dict]:
    records: list[dict] = []
    with states_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
