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


def validate_episode(
    records: list[dict],
    require_fall: bool = True,
    require_contact: bool = False,
    min_approach_frames: int | None = None,
) -> dict:
    """Validate per-frame state records and return a summary dict.

    Basic checks are always run (finite, moved, fell, max speed).  When
    ``require_contact`` is set it also verifies that two recorded dynamic
    objects actually touch at some frame; ``min_approach_frames`` additionally
    requires the contact to happen at least that many frames after the start
    (the visible run-up phase).  Both are opt-in so other categories that
    legitimately never collide are not rejected.
    """

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

    # Contact check (opt-in): any two distinct recorded objects whose
    # centres come within 12 cm of each other.
    contact_frame = None
    if require_contact:
        by_obj: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_obj[record["object_id"]].append(record)
        dynamic = {oid: recs for oid, recs in by_obj.items()}
        items = list(dynamic.items())
        for i, (oid_a, recs_a) in enumerate(items):
            for oid_b, recs_b in items[i + 1 :]:
                n = min(len(recs_a), len(recs_b))
                for k in range(n):
                    pa, pb = recs_a[k]["position"], recs_b[k]["position"]
                    d2 = sum((x - y) ** 2 for x, y in zip(pa, pb))
                    if d2 <= (0.12 ** 2):
                        contact_frame = recs_a[k]["frame"]
                        break
                if contact_frame is not None:
                    break
            if contact_frame is not None:
                break

    has_contact = contact_frame is not None
    approach_ok = (
        contact_frame is None or min_approach_frames is None
        or contact_frame >= min_approach_frames
    )

    passed = (
        finite
        and moved
        and (fell or not require_fall)
        and max_speed_ok
        and (not require_contact or has_contact)
        and approach_ok
    )

    return {
        "finite": bool(finite),
        "moved": bool(moved),
        "fell": bool(fell),
        "max_speed_ok": bool(max_speed_ok),
        "max_speed": float(max_speed),
        "contact_frame": contact_frame,
        "has_contact": bool(has_contact),
        "approach_ok": bool(approach_ok),
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
