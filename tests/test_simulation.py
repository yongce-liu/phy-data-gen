from phy_data_gen.simulation import compute_step_counts
from phy_data_gen.validation import validate_episode


def test_reference_frame_cadence() -> None:
    steps, capture_every = compute_step_counts(5.0, 1.0 / 480.0, 30)

    assert steps == 2400
    assert capture_every == 16
    assert len(range(0, steps, capture_every)) == 150


def test_template_dynamics_validation_does_not_require_falling() -> None:
    records = [
        {
            "frame": 0,
            "object_id": "CueBall",
            "position": [0.0, 0.0, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [4.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 1.0],
        },
        {
            "frame": 1,
            "object_id": "CueBall",
            "position": [0.1, 0.0, 0.05],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [3.5, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 1.0],
        },
    ]

    summary = validate_episode(records, require_fall=False)

    assert summary["moved"] is True
    assert summary["fell"] is False
    assert summary["passed"] is True
