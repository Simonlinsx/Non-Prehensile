"""Simulator-independent DAPL task metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def planar_pose_success(
    current_position: torch.Tensor,
    current_quaternion: torch.Tensor,
    goal_pose: torch.Tensor,
    *,
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
) -> torch.Tensor:
    """Evaluate DAPL's planar-position and full-orientation success rule.

    Quaternions use the Isaac ``[w, x, y, z]`` convention.  Normalizing both
    inputs makes the metric robust to small simulator drift, and taking the
    absolute inner product handles the equivalent ``q`` and ``-q`` forms.
    """

    if current_position.shape[-1] != 3:
        raise ValueError("current_position must have three coordinates")
    if current_quaternion.shape[-1] != 4:
        raise ValueError("current_quaternion must have four coordinates")
    if goal_pose.shape[-1] != 7:
        raise ValueError("goal_pose must contain position and wxyz quaternion")
    if current_position.shape[:-1] != current_quaternion.shape[:-1]:
        raise ValueError("current position and quaternion batch shapes must match")
    if current_position.shape[:-1] != goal_pose.shape[:-1]:
        raise ValueError("current and goal batch shapes must match")
    if position_threshold <= 0.0 or rotation_threshold <= 0.0:
        raise ValueError("success thresholds must be positive")

    planar_distance = torch.linalg.vector_norm(
        goal_pose[..., :2] - current_position[..., :2], dim=-1
    )
    current_quaternion = functional.normalize(current_quaternion, dim=-1)
    goal_quaternion = functional.normalize(goal_pose[..., 3:7], dim=-1)
    quaternion_dot = torch.sum(current_quaternion * goal_quaternion, dim=-1)
    angular_distance = 2.0 * torch.acos(
        torch.clamp(torch.abs(quaternion_dot), max=1.0)
    )
    return (planar_distance < position_threshold) & (
        angular_distance < rotation_threshold
    )
