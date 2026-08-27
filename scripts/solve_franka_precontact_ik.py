#!/usr/bin/env python3
"""Solve one reproducible Franka hand pose for task reset configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


DEFAULT_URDF = Path(
    "/data1/linsixu/IsaacLab-2.2.0/source/isaaclab/isaaclab/"
    "controllers/config/data/lula_franka_gen.urdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--hand-position",
        type=float,
        nargs=3,
        default=(0.252, -0.061, 0.102),
        metavar=("X", "Y", "Z"),
        help="Desired panda_hand origin in the robot base frame.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--restarts", type=int, default=32)
    parser.add_argument(
        "--orientation",
        choices=("horizontal-x", "vertical-down"),
        default="horizontal-x",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = pin.buildModelFromUrdf(str(args.urdf))
    frame_id = model.getFrameId("panda_hand")
    target_position = np.asarray(args.hand_position, dtype=np.float64)
    # Local hand +Z points along world +X; local +Y remains world +Y.  The
    # average fingertip/TCP is therefore approximately 0.1034 m ahead in +X.
    if args.orientation == "horizontal-x":
        target_rotation = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    else:
        target_rotation = np.diag((1.0, -1.0, -1.0))
    lower = model.lowerPositionLimit[:7] + 1.0e-4
    upper = model.upperPositionLimit[:7] - 1.0e-4
    reference = np.array(
        [0.0, 0.0398, 0.0, -2.13345, 0.0, 2.05065, 0.0], dtype=np.float64
    )
    rng = np.random.default_rng(args.seed)
    starts = [np.clip(reference, lower, upper)]
    starts.extend(rng.uniform(lower, upper) for _ in range(args.restarts - 1))

    def forward(q_arm: np.ndarray) -> pin.SE3:
        q = np.concatenate((q_arm, np.array((0.04, 0.04))))
        data = model.createData()
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return data.oMf[frame_id]

    def residual(q_arm: np.ndarray) -> np.ndarray:
        pose = forward(q_arm)
        position_error = 10.0 * (pose.translation - target_position)
        rotation_error = pin.log3(pose.rotation.T @ target_rotation)
        regularizer = 1.0e-4 * (q_arm - reference)
        return np.concatenate((position_error, rotation_error, regularizer))

    candidates = []
    for start in starts:
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=2_000,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        pose = forward(result.x)
        position_error = np.linalg.norm(pose.translation - target_position)
        rotation_error = np.linalg.norm(pin.log3(pose.rotation.T @ target_rotation))
        candidates.append((position_error + rotation_error, position_error, rotation_error, result.x, pose))

    _, position_error, rotation_error, q_arm, pose = min(candidates, key=lambda item: item[0])
    quat_xyzw = Rotation.from_matrix(pose.rotation).as_quat()
    print("joint_pos:", q_arm.tolist())
    print("hand_position:", pose.translation.tolist())
    print("hand_quaternion_wxyz:", [quat_xyzw[3], *quat_xyzw[:3]])
    print("position_error_m:", position_error)
    print("rotation_error_rad:", rotation_error)
    print("tcp_position:", (pose.translation + pose.rotation[:, 2] * 0.1034).tolist())


if __name__ == "__main__":
    main()
