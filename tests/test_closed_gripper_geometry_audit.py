import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from scripts.analyze_closed_gripper_geometry import convex_distance, signed_surface_distance
from scripts.audit_closed_gripper_proxy_contacts import proxy_center_in_target_frame


@pytest.mark.parametrize("shift,expected", [([.03, 0, 0], .01), ([.03, .04, 0], np.sqrt(.0005)), ([.005, 0, 0], 0)])
def test_convex_distance_separated_diagonal_and_overlapping(shift, expected):
    box = trimesh.creation.box(extents=[.02] * 3).vertices
    # The same answer must hold after a common rotation and translation.
    rotation = Rotation.from_euler("xyz", [.7, .2, -.4])
    a = rotation.apply(box) + [.4, -.2, .3]
    b = rotation.apply(box + shift) + [.4, -.2, .3]
    distance, points = convex_distance(a, b)
    assert distance == pytest.approx(expected, abs=1e-7)
    assert np.linalg.norm(points[0] - points[1]) == pytest.approx(distance)


def test_signed_surface_distance_inside_and_outside():
    mesh = trimesh.creation.box(extents=[.02] * 3)
    assert signed_surface_distance(mesh, [[0, 0, 0], [.03, 0, 0]]) == pytest.approx([-.01, .02])


def test_proxy_transform_respects_rotated_base_env_origin_and_support():
    base = Rotation.from_euler("z", .8)
    target = Rotation.from_euler("xyz", [.2, .3, -.4])
    support = Rotation.from_euler("xyz", [-.3, .5, .7])
    root_xyz, origin = np.array([1., 2., .03]), np.array([.5, 1., 0.])
    target_xyz, expected = np.array([.4, .2, .01]), np.array([.03, -.01, .02])
    tip_w = target.apply(support.inv().apply(expected)) + target_xyz + origin
    row = {"target_position_m": target_xyz.tolist(), "target_quaternion_wxyz": np.roll(target.as_quat(), 1).tolist(),
           "planner_tip_position_m": base.inv().apply(tip_w - root_xyz).tolist(),
           "robot_root_pose_w": root_xyz.tolist() + np.roll(base.as_quat(), 1).tolist(), "env_origin_w": origin.tolist()}
    assert proxy_center_in_target_frame(row, {}, support) == pytest.approx(expected)


def test_legacy_frame_matches_explicit_recorded_base():
    row = {"target_position_m": [.4, .2, .01], "target_quaternion_wxyz": [1, 0, 0, 0],
           "planner_tip_position_m": [.42, .2, -.005]}
    legacy = proxy_center_in_target_frame(row, {"franka_base_height_m": .029}, Rotation.identity())
    row.update(robot_root_pose_w=[0, 0, .029, 1, 0, 0, 0], env_origin_w=[0, 0, 0])
    assert proxy_center_in_target_frame(row, {}, Rotation.identity()) == pytest.approx(legacy)
