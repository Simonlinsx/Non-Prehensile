# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from scipy.spatial.transform import Rotation as R

from dapl.metrics import planar_pose_success

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.01,  # Position threshold in meters
    rotation_threshold: float = 0.1,  # Rotation threshold in radians
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    planar: bool = False,  # If True, only consider x,y coordinates for position distance
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position and orientation.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object position to reach the goal. Defaults to 0.01.
        rotation_threshold: The threshold for the object orientation in radians. Defaults to 0.1.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        planar: If True, only consider x,y coordinates for position distance (ignore z). Defaults to False.

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # command contains 7D pose: [x, y, z, qw, qx, qy, qz] in environment coordinates
    des_pos_env = command[:, :3]  # target position in environment coordinates
    des_rot_env = command[:, 3:7]  # target orientation (quaternion) in environment coordinates
    
    # get current object position in environment coordinates
    object_pos_w = object.data.root_pos_w[:, :3]  # world coordinates
    object_pos_env = object_pos_w - env.scene.env_origins  # convert to environment coordinates
    
    # get current object orientation
    from scipy.spatial.transform import Rotation as R
    object_quat_w = object.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    # Convert quaternion to euler angles for comparison
    # Isaac uses [w,x,y,z] format, scipy expects [x,y,z,w]
    quat_scipy = object_quat_w[:, [1, 2, 3, 0]]  # Convert from [w,x,y,z] to [x,y,z,w]
    quat_np = quat_scipy.cpu().numpy()
    
    # calculate position distance (only consider x,y if planar=True)
    if planar:
        position_distance = torch.norm(des_pos_env[:, :2] - object_pos_env[:, :2], dim=1)
        return position_distance < threshold
       
    position_distance = torch.norm(des_pos_env - object_pos_env, dim=1)
    
    # Calculate angular distance between quaternions
    dot_product = torch.sum(object_quat_w * des_rot_env, dim=1)  # (num_envs,)
    dot_product = torch.clamp(torch.abs(dot_product), max=1.0)  # Clamp to avoid numerical errors
    angular_distance = 2 * torch.acos(dot_product)  # Angular distance in radians
    
    # terminate if both position and orientation are within thresholds
    position_reached = position_distance < threshold
    rotation_reached = angular_distance < rotation_threshold

    return position_reached & rotation_reached


def clutter_target_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Terminate on the planar-position and full-orientation success criterion.

    DAPL evaluates target translation in the table plane while still requiring
    the full quaternion angular error to be below the rotation threshold.  The
    legacy ``object_reached_goal(planar=True)`` intentionally ignores rotation,
    so the clutter task needs a separate termination term.
    """

    target: RigidObject = env.scene[target_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_env = target.data.root_pos_w[:, :3] - env.scene.env_origins
    return planar_pose_success(
        target_pos_env,
        target.data.root_quat_w,
        command,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
    )


def object_dropped_off_table(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.02,  # Minimum height above table surface
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object dropping off the table.

    Args:
        env: The environment.
        minimum_height: The minimum height above the table surface. Defaults to 0.02m.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate due to object falling.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    # get current object position in world coordinates
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_height = object_pos_w[:, 2]  # z-coordinate
    # terminate if object is below minimum height (fell off table)
    return object_height < minimum_height


def clutter_object_dropped_off_table(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.15,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
) -> torch.Tensor:
    """Terminate when the target or any non-target object falls off the table."""

    target: RigidObject = env.scene[target_cfg.name]
    obstacles: RigidObjectCollection = env.scene[obstacles_cfg.name]
    target_dropped = target.data.root_pos_w[:, 2] < minimum_height
    active_obstacle_count = int(
        getattr(env, "_clutter_active_obstacle_count", obstacles.num_objects)
    )
    if active_obstacle_count == 0:
        obstacle_dropped = torch.zeros_like(target_dropped)
    else:
        obstacle_dropped = torch.any(
            obstacles.data.object_pos_w[:, :active_obstacle_count, 2]
            < minimum_height,
            dim=1,
        )
    return target_dropped | obstacle_dropped
