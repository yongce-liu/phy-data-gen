from pathlib import Path

from phy_data_gen.dataset import default_run_id


def test_default_run_id_uses_template_hash_and_seed() -> None:
    run_id = default_run_id(
        "objects_falling",
        Path("objects_falling_c8917ca2_871.usda"),
        42,
    )

    assert run_id == "objects_falling_c8917ca2_42"


def test_default_run_id_handles_template_without_hash() -> None:
    run_id = default_run_id("objects_falling", Path("template.usda"), 7)

    assert run_id == "objects_falling_00000000_7"
