"""Track a moving C3 reference with Isaac Lab 2.2's robot-local OSC.

Import after AppLauncher, like Isaac Lab's other action terms. The upstream
OSC assumes a stationary reference; its velocity argument is used only in
the motion damping term. Passing measured minus desired velocity there gives
Kd * (v_des - v) without changing inertia, wrench, or null-space control.
"""

from __future__ import annotations

import torch

from isaaclab.controllers import OperationalSpaceController
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import math as math_utils


class TrajectoryOperationalSpaceController(OperationalSpaceController):
    def __init__(self, cfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        self.desired_velocity_b = torch.zeros(num_envs, 6, device=device)

    def set_desired_velocity(self, velocity_b: torch.Tensor) -> None:
        if velocity_b.shape != self.desired_velocity_b.shape:
            raise ValueError("desired velocity must have shape (num_envs, 6)")
        if not torch.isfinite(velocity_b).all():
            raise ValueError("desired velocity must be finite")
        self.desired_velocity_b.copy_(velocity_b)

    def compute(self, jacobian_b, current_ee_pose_b=None, current_ee_vel_b=None, **kwargs):
        relative_velocity_b = (
            None if current_ee_vel_b is None
            else current_ee_vel_b - self.desired_velocity_b
        )
        return super().compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=current_ee_pose_b,
            current_ee_vel_b=relative_velocity_b,
            **kwargs,
        )


class TrajectoryOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._osc = TrajectoryOperationalSpaceController(
            cfg=self.cfg.controller_cfg, num_envs=self.num_envs, device=self.device
        )

    def set_desired_velocity(self, velocity_b: torch.Tensor) -> None:
        self._osc.set_desired_velocity(velocity_b)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._osc.desired_velocity_b.zero_()
        else:
            self._osc.desired_velocity_b[env_ids] = 0.0


class TaskPointOperationalSpaceControllerAction(TrajectoryOperationalSpaceControllerAction):
    """Use one offset point for pose, twist, Jacobian and applied wrench.

    Isaac Lab 2.2's stock offset Jacobian uses a body-axis offset in a
    root-axis Jacobian. Rotate the offset before shifting, and obtain twist
    from J*qdot for this fixed-base arm rather than mixing link pose with
    the body's COM velocity. J_tip.T automatically maps a tip force to its
    corresponding wrist moment.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not self._asset.is_fixed_base:
            raise ValueError("Task-point OSC currently requires a fixed-base articulation")

    def dynamics_audit_snapshot(self):
        """Read cached inputs/inertia from the last OSC call; no control changes."""
        return {
            "frame": "robot_root_axes_at_task_point",
            "state_timing": "cached last OSC call before the final physics substep of this servo interval",
            "joint_position_rad": self._joint_pos[0].detach().cpu().tolist(),
            "joint_velocity_rad_s": self._joint_vel[0].detach().cpu().tolist(),
            "ee_pose_b_wxyz": self._ee_pose_b[0].detach().cpu().tolist(),
            "jacobian_b": self._jacobian_b[0].detach().cpu().tolist(),
            "joint_mass_matrix_kg_m2": self._mass_matrix[0].detach().cpu().tolist(),
            "osc_operational_inertia_b": self._osc._os_mass_matrix_b[0].detach().cpu().tolist(),
            "inertial_dynamics_decoupling": self._osc.cfg.inertial_dynamics_decoupling,
            "partial_inertial_dynamics_decoupling": self._osc.cfg.partial_inertial_dynamics_decoupling,
        }

    def _compute_ee_jacobian(self):
        # Clone before rotation: never mutate a cached PhysX Jacobian view.
        jacobian = self.jacobian_w.clone()
        base_inv = math_utils.quat_inv(self._asset.data.root_quat_w)
        R = math_utils.matrix_from_quat(base_inv)
        jacobian[:, :3] = torch.bmm(R, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(R, jacobian[:, 3:])
        if self.cfg.body_offset is not None:
            hand_q_b = math_utils.quat_mul(
                base_inv, self._asset.data.body_quat_w[:, self._ee_body_idx])
            r_b = math_utils.quat_apply(hand_q_b, self._offset_pos)
            jacobian[:, :3] -= torch.bmm(math_utils.skew_symmetric_matrix(r_b), jacobian[:, 3:])
        self._jacobian_b[:] = jacobian

    def _compute_ee_velocity(self):
        qdot = self._asset.data.joint_vel[:, self._joint_ids]
        self._ee_vel_b[:] = torch.bmm(self._jacobian_b, qdot.unsqueeze(-1)).squeeze(-1)
