from pxr import Usd, UsdGeom, UsdPhysics

from phy_data_gen.config import SceneConfig
from phy_data_gen.scene import _repair_missing_template_primitives, build_scene
from phy_data_gen.simulation import (
    _find_replacement_rigid_body_paths,
    _find_template_rigid_body_paths,
)
from phy_data_gen.schemas import AssetReplacementSpec, EpisodeSpec


def _scene_config() -> SceneConfig:
    return SceneConfig(
        name="billiards",
        object_mode="template_dynamics",
        world_prim="/World",
        physics_scene_prim="/PhysicsScene",
        ground_prim="/World/TableSurface",
        cameras={},
        dynamic_prims=(),
    )


def test_repairs_missing_billiards_primitives() -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")

    cue_ball = stage.DefinePrim("/World/CueBall", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(cue_ball)
    UsdPhysics.CollisionAPI.Apply(cue_ball)
    stage.DefinePrim("/World/CueBall/sphere")

    bumper = stage.DefinePrim("/World/Bumper_N", "Xform")
    UsdPhysics.CollisionAPI.Apply(bumper)
    stage.DefinePrim("/World/Bumper_N/prism")

    repaired = _repair_missing_template_primitives(stage, _scene_config())

    sphere = UsdGeom.Sphere(stage.GetPrimAtPath("/World/CueBall/sphere"))
    cube = UsdGeom.Cube(stage.GetPrimAtPath("/World/Bumper_N/prism"))
    assert repaired == ["/World/Bumper_N/prism", "/World/CueBall/sphere"]
    assert sphere.GetRadiusAttr().Get() == 0.5
    assert cube.GetSizeAttr().Get() == 1.0
    assert sphere.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    assert cube.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_discovers_template_rigid_bodies() -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    ball = stage.DefinePrim("/World/Ball_00_rubber", "Xform")
    cue_ball = stage.DefinePrim("/World/CueBall", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(ball)
    UsdPhysics.RigidBodyAPI.Apply(cue_ball)

    paths = _find_template_rigid_body_paths(stage, "/World")

    assert paths == {
        "Ball_00_rubber": "/World/Ball_00_rubber",
        "CueBall": "/World/CueBall",
    }


def test_replaces_geometry_but_preserves_template_rigid_body(tmp_path) -> None:
    template_path = tmp_path / "template.usda"
    template = Usd.Stage.CreateNew(str(template_path))
    world = UsdGeom.Xform.Define(template, "/World").GetPrim()
    template.SetDefaultPrim(world)
    UsdPhysics.Scene.Define(template, "/PhysicsScene")
    body = UsdGeom.Xform.Define(template, "/World/Ball").GetPrim()
    body_api = UsdPhysics.RigidBodyAPI.Apply(body)
    body_api.CreateVelocityAttr((1.0, 2.0, 3.0))
    UsdGeom.Sphere.Define(template, "/World/Ball/Original")
    template.GetRootLayer().Save()

    asset_path = tmp_path / "asset.usda"
    asset = Usd.Stage.CreateNew(str(asset_path))
    asset_root = UsdGeom.Xform.Define(asset, "/Asset").GetPrim()
    asset.SetDefaultPrim(asset_root)
    nested_body = UsdGeom.Xform.Define(asset, "/Asset/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(nested_body)
    collider = UsdGeom.Cube.Define(asset, "/Asset/Body/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    asset.GetRootLayer().Save()

    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path=str(template_path),
        backend="physx",
        object_mode="replace_assets",
        duration_seconds=1.0,
        physics_dt=1.0 / 60.0,
        render_fps=30,
        objects=[],
        replacements=[
            AssetReplacementSpec(
                object_id="Ball",
                target_prim_path="/World/Ball",
                asset_path=str(asset_path),
                scale=0.5,
                translation=(0.0, 0.0, 0.0),
                asset_rigid_body_path="/Body",
            )
        ],
    )
    output_path = tmp_path / "compiled.usda"

    build_scene(spec, _scene_config(), output_path)
    compiled = Usd.Stage.Open(str(output_path))
    compiled_body = compiled.GetPrimAtPath("/World/Ball")
    rigid_paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(compiled_body)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]

    replacement_body_path = "/World/Ball/ReplacementAsset/Asset/Body"
    assert rigid_paths == [replacement_body_path]
    assert not compiled_body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert compiled.GetPrimAtPath(replacement_body_path).GetAttribute(
        "physics:velocity"
    ).Get() == (1.0, 2.0, 3.0)
    assert not compiled.GetPrimAtPath("/World/Ball/Original").IsActive()
    assert compiled.GetPrimAtPath("/World/Ball/ReplacementAsset/Asset").IsValid()


def test_replacement_recording_includes_non_replaced_template_body() -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    cue_ball = stage.DefinePrim("/World/CueBall", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(cue_ball)
    replacement_target = stage.DefinePrim("/World/TargetBall", "Xform")
    replacement_body = stage.DefinePrim(
        "/World/TargetBall/ReplacementAsset/Body", "Xform"
    )
    UsdPhysics.RigidBodyAPI.Apply(replacement_body)
    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path="template.usda",
        backend="physx",
        object_mode="replace_assets",
        duration_seconds=1.0,
        physics_dt=1.0 / 60.0,
        render_fps=30,
        objects=[],
        replacements=[
            AssetReplacementSpec(
                object_id="TargetBall",
                target_prim_path=str(replacement_target.GetPath()),
                asset_path="asset.usda",
                scale=1.0,
                translation=(0.0, 0.0, 0.0),
            )
        ],
    )

    paths = _find_replacement_rigid_body_paths(stage, spec, "/World")

    assert paths == {
        "TargetBall": "/World/TargetBall/ReplacementAsset/Body",
        "CueBall": "/World/CueBall",
    }
