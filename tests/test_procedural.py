"""Tests for procedural scenes, primitive objects, velocities and cameras."""

import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from phy_data_gen.config import ProceduralConfig, SceneConfig
from phy_data_gen.schemas import CameraSpec, EpisodeSpec, ObjectSpec, PhysicsMaterialSpec
from phy_data_gen.scene import build_scene, _object_prim_path
from phy_data_gen.simulation import _find_generated_rigid_body_paths


def _scene_config(**overrides) -> SceneConfig:
    defaults = dict(
        name="procedural",
        object_mode="procedural",
        world_prim="/World",
        physics_scene_prim="/PhysicsScene",
        ground_prim="/World/Ground",
        cameras={},
        dynamic_prims=(),
        procedural=ProceduralConfig(build_ground=True, ground_size=4.0),
    )
    defaults.update(overrides)
    return SceneConfig(**defaults)


def _ball(
    object_id: str,
    position=(0.0, 0.0, 0.1),
    radius=0.05,
    mass=0.1,
    velocity=(0.0, 0.0, 0.0),
    **overrides,
) -> ObjectSpec:
    defaults = dict(
        object_id=object_id,
        asset_path="",
        position=position,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        scale=1.0,
        mass=mass,
        material=PhysicsMaterialSpec(
            static_friction=0.5, dynamic_friction=0.4, restitution=0.5
        ),
        kind="sphere",
        radius=radius,
        initial_linear_velocity=velocity,
    )
    defaults.update(overrides)
    return ObjectSpec(**defaults)


def test_primitive_object_authors_single_rigid_body(tmp_path) -> None:
    scene = _scene_config()
    output = tmp_path / "episode.usda"
    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path=None,
        backend="physx",
        object_mode="procedural",
        duration_seconds=1.0,
        physics_dt=1.0 / 120.0,
        render_fps=30,
        objects=[
            _ball(
                "ball_0",
                position=(0.0, 0.0, 0.1),
                radius=0.05,
                velocity=(1.0, 2.0, 0.0),
            ),
            _ball(
                "ball_1",
                position=(0.3, 0.0, 0.1),
                radius=0.05,
                mass=0.5,
            ),
        ],
    )
    build_scene(spec, scene, output)

    compiled = Usd.Stage.Open(str(output))
    paths = _find_generated_rigid_body_paths(compiled, spec)

    assert set(paths) == {"ball_0", "ball_1"}
    assert paths["ball_0"] == "/World/GeneratedObjects/ball_0"

    ball_0 = compiled.GetPrimAtPath(paths["ball_0"])
    assert ball_0.HasAPI(UsdPhysics.RigidBodyAPI)
    assert ball_0.GetAttribute("physics:velocity").Get() == (1.0, 2.0, 0.0)
    assert ball_0.GetAttribute("physics:angularVelocity").Get() == (0.0, 0.0, 0.0)
    assert ball_0.GetAttribute("physxRigidBody:enableCCD").Get() is True
    mass_api = UsdPhysics.MassAPI(ball_0)
    assert mass_api.GetMassAttr().Get() == pytest.approx(0.1)

    sphere = UsdGeom.Sphere(compiled.GetPrimAtPath("/World/GeneratedObjects/ball_0/Geometry"))
    assert sphere.GetRadiusAttr().Get() == 0.05
    assert sphere.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_record_false_objects_are_excluded_from_views(tmp_path) -> None:
    scene = _scene_config()
    output = tmp_path / "episode.usda"
    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path=None,
        backend="physx",
        object_mode="procedural",
        duration_seconds=1.0,
        physics_dt=1.0 / 120.0,
        render_fps=30,
        objects=[
            _ball("ball_0", position=(0.0, 0.0, 0.2), radius=0.05),
            _ball("grain_0", position=(0.1, 0.1, 0.02), radius=0.015, record=False),
            _ball("grain_1", position=(0.12, 0.1, 0.02), radius=0.015, record=False),
        ],
    )
    build_scene(spec, scene, output)
    compiled = Usd.Stage.Open(str(output))

    assert _object_prim_path(spec.objects[0]) == "/World/GeneratedObjects/ball_0"
    assert _object_prim_path(spec.objects[1]) == "/World/GeneratedObjects/Bulk/grain_0"

    paths = _find_generated_rigid_body_paths(compiled, spec)
    assert set(paths) == {"ball_0"}
    # Bulk grains still carry a rigid body so PhysX simulates them.
    grain = compiled.GetPrimAtPath("/World/GeneratedObjects/Bulk/grain_0")
    assert grain.HasAPI(UsdPhysics.RigidBodyAPI)


def test_static_object_has_no_rigid_body(tmp_path) -> None:
    scene = _scene_config()
    output = tmp_path / "episode.usda"
    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path=None,
        backend="physx",
        object_mode="procedural",
        duration_seconds=1.0,
        physics_dt=1.0 / 120.0,
        render_fps=30,
        objects=[
            _ball("ball_0", position=(0.0, 0.0, 0.2), radius=0.05),
            _ball("obstacle", position=(0.5, 0.0, 0.5), radius=0.2, dynamic=False),
        ],
    )
    build_scene(spec, scene, output)
    compiled = Usd.Stage.Open(str(output))
    obstacle = compiled.GetPrimAtPath("/World/GeneratedObjects/obstacle")
    assert not obstacle.HasAPI(UsdPhysics.RigidBodyAPI)
    geometry = compiled.GetPrimAtPath("/World/GeneratedObjects/obstacle/Geometry")
    assert geometry.HasAPI(UsdPhysics.CollisionAPI)

    paths = _find_generated_rigid_body_paths(compiled, spec)
    assert set(paths) == {"ball_0"}


def test_cameras_are_defined_and_metadata_readable(tmp_path) -> None:
    scene = _scene_config(cameras={"Side": "/World/Camera_Side"})
    output = tmp_path / "episode.usda"
    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path=None,
        backend="physx",
        object_mode="procedural",
        duration_seconds=1.0,
        physics_dt=1.0 / 120.0,
        render_fps=30,
        objects=[_ball("ball_0", position=(0.0, 0.0, 0.1), radius=0.05)],
        cameras={
            "Side": CameraSpec(
                prim_path="/World/Camera_Side",
                position=(2.0, 0.0, 1.0),
                look_at=(0.0, 0.0, 0.1),
            )
        },
    )
    build_scene(spec, scene, output)
    compiled = Usd.Stage.Open(str(output))
    camera_prim = compiled.GetPrimAtPath("/World/Camera_Side")
    assert camera_prim.IsValid()
    assert camera_prim.IsA(UsdGeom.Camera)
    camera = UsdGeom.Camera(camera_prim)
    assert camera.GetFocalLengthAttr().Get() == 24.0
    world = camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert world.ExtractTranslation() is not None


def test_template_procedural_disables_template_dynamics(tmp_path) -> None:
    template = tmp_path / "template.usda"
    stage = Usd.Stage.CreateNew(str(template))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    cue = UsdGeom.Xform.Define(stage, "/World/CueBall").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(cue)
    stage.GetRootLayer().Save()

    scene = _scene_config()
    output = tmp_path / "episode.usda"
    spec = EpisodeSpec(
        episode_id="test",
        seed=1,
        template_path=str(template),
        backend="physx",
        object_mode="procedural",
        duration_seconds=1.0,
        physics_dt=1.0 / 120.0,
        render_fps=30,
        objects=[_ball("ball_0", position=(0.0, 0.0, 0.1), radius=0.05)],
    )
    build_scene(spec, scene, output)
    compiled = Usd.Stage.Open(str(output))
    cue = compiled.GetPrimAtPath("/World/CueBall")
    assert not cue.IsActive()
    ball = compiled.GetPrimAtPath("/World/GeneratedObjects/ball_0")
    assert ball.IsValid()
