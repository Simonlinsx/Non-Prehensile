"""Franka TCP kinematics used by Isaac Lab contact-planner executors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares


class FrankaTcpIK:
    """Pinocchio IK with a bridge calibrated from a live simulator pose."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        hand_to_tcp_m: float = 0.1034,
        max_evaluations: int = 100,
    ) -> None:
        source = Path(urdf_path)
        if not source.is_file():
            raise FileNotFoundError(f"Franka IK URDF is missing: {source}")
        self.model = pin.buildModelFromUrdf(str(source))
        self.frame_id = self.model.getFrameId("panda_hand")
        self.hand_to_tcp = pin.SE3(
            np.eye(3), np.array((0.0, 0.0, float(hand_to_tcp_m)))
        )
        self.lower = self.model.lowerPositionLimit[:7] + 1.0e-4
        self.upper = self.model.upperPositionLimit[:7] - 1.0e-4
        self.max_evaluations = int(max_evaluations)

    def forward(self, q_arm: np.ndarray) -> pin.SE3:
        data = self.model.createData()
        q_full = np.concatenate((np.asarray(q_arm, dtype=float), np.array((0.04, 0.04))))
        pin.forwardKinematics(self.model, data, q_full)
        pin.updateFramePlacements(self.model, data)
        return data.oMf[self.frame_id] * self.hand_to_tcp

    def bridge(
        self,
        q_reference: np.ndarray,
        tcp_position_env: np.ndarray,
        tcp_rotation_env: np.ndarray,
    ) -> tuple[pin.SE3, np.ndarray, np.ndarray]:
        """Calibrate the URDF FK frame to a live Isaac Lab TCP pose."""

        pin_reference = self.forward(q_reference)
        env_from_pin_rotation = tcp_rotation_env @ pin_reference.rotation.T
        return pin_reference, env_from_pin_rotation, tcp_position_env

    def solve(
        self,
        *,
        seed: np.ndarray,
        regularization_reference: np.ndarray,
        bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
        desired_tcp_env: np.ndarray,
        desired_rotation_env: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        pin_reference, env_from_pin_rotation, tcp_reference_env = bridge
        desired_translation_pin = pin_reference.translation + (
            env_from_pin_rotation.T @ (desired_tcp_env - tcp_reference_env)
        )
        desired_rotation_pin = env_from_pin_rotation.T @ desired_rotation_env

        def residual(q_arm: np.ndarray) -> np.ndarray:
            pose = self.forward(q_arm)
            return np.concatenate(
                (
                    20.0 * (pose.translation - desired_translation_pin),
                    pin.log3(pose.rotation.T @ desired_rotation_pin),
                    5.0e-4 * (q_arm - regularization_reference),
                )
            )

        result = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            max_nfev=self.max_evaluations,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        solved = self.forward(result.x)
        position_error = float(np.linalg.norm(solved.translation - desired_translation_pin))
        rotation_error = float(
            np.linalg.norm(pin.log3(solved.rotation.T @ desired_rotation_pin))
        )
        return result.x, position_error, rotation_error

    def solve_translation(
        self,
        *,
        seed: np.ndarray,
        regularization_reference: np.ndarray,
        desired_translation: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Solve a translation-only target in the URDF base frame.

        This is the correct local-control boundary for Push Anything because
        its point end effector does not track orientation.  The small joint
        regularizer resolves redundancy continuously from the previous target.
        """

        desired = np.asarray(desired_translation, dtype=float)
        if desired.shape != (3,) or not np.all(np.isfinite(desired)):
            raise ValueError("desired_translation must contain three finite values")
        reference = np.asarray(regularization_reference, dtype=float)

        def residual(q_arm: np.ndarray) -> np.ndarray:
            pose = self.forward(q_arm)
            return np.concatenate(
                (
                    20.0 * (pose.translation - desired),
                    5.0e-4 * (q_arm - reference),
                )
            )

        result = least_squares(
            residual,
            seed,
            bounds=(self.lower, self.upper),
            max_nfev=self.max_evaluations,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        position_error = float(
            np.linalg.norm(self.forward(result.x).translation - desired)
        )
        return result.x, position_error

    def tcp_pose_in_env(
        self,
        q_arm: np.ndarray,
        bridge: tuple[pin.SE3, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        pin_reference, env_from_pin_rotation, tcp_reference_env = bridge
        pose = self.forward(q_arm)
        position = tcp_reference_env + env_from_pin_rotation @ (
            pose.translation - pin_reference.translation
        )
        rotation = env_from_pin_rotation @ pose.rotation
        return position, rotation
