from phy_data_gen.simulation import compute_step_counts


def test_reference_frame_cadence() -> None:
    steps, capture_every = compute_step_counts(5.0, 1.0 / 480.0, 30)

    assert steps == 2400
    assert capture_every == 16
    assert len(range(0, steps, capture_every)) == 150
