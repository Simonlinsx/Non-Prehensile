"""Run with AppLauncher active; ordinary CPU pytest skips this integration test."""

import pytest
import torch

pytest.importorskip("omni.log")

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from dapl.contact_planner.isaac_trajectory_action import TrajectoryOperationalSpaceController


def make_controller(cls):
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs", "wrench_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        motion_stiffness_task=100.0,
        motion_damping_ratio_task=1.0,
        contact_wrench_control_axes_task=(1, 1, 1, 0, 0, 0),
        nullspace_control="none",
    )
    controller = cls(cfg, num_envs=1, device="cpu")
    pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    wrench = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]])
    controller.set_command(torch.cat((pose, wrench), dim=1), current_ee_pose_b=pose)
    return controller, dict(
        jacobian_b=torch.eye(6).unsqueeze(0),
        mass_matrix=torch.eye(6).unsqueeze(0),
        current_ee_pose_b=pose,
        current_ee_vel_b=torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0]]),
    )


def test_perfect_moving_reference_preserves_feedforward_wrench():
    controller, state = make_controller(TrajectoryOperationalSpaceController)
    measured = state["current_ee_vel_b"].clone()
    controller.set_desired_velocity(measured)
    torque = controller.compute(**state)
    torch.testing.assert_close(torque, torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(state["current_ee_vel_b"], measured)


def test_zero_velocity_reference_exactly_matches_stock_controller():
    controller, state = make_controller(TrajectoryOperationalSpaceController)
    stock, _ = make_controller(OperationalSpaceController)
    torch.testing.assert_close(controller.compute(**state), stock.compute(**state))


def test_clear_velocity_after_motion_restores_hold_damping():
    controller, state = make_controller(TrajectoryOperationalSpaceController)
    controller.set_desired_velocity(state["current_ee_vel_b"])
    controller.set_desired_velocity(torch.zeros(1, 6))
    stock, _ = make_controller(OperationalSpaceController)
    torch.testing.assert_close(controller.compute(**state), stock.compute(**state))


def test_invalid_velocity_is_rejected_without_changing_reference():
    controller, _ = make_controller(TrajectoryOperationalSpaceController)
    with pytest.raises(ValueError, match="shape"):
        controller.set_desired_velocity(torch.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        controller.set_desired_velocity(torch.full((1, 6), float("nan")))
    torch.testing.assert_close(controller.desired_velocity_b, torch.zeros(1, 6))
