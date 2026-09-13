# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from dapl.metrics import (
    dapl_combined_pose_error,
    dapl_tanh_proximity_reward,
    planar_pose_success,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_left = ee_frame.data.target_pos_w[..., 1, :]
    ee_right = ee_frame.data.target_pos_w[..., 2, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance_left = torch.norm(obj_pos_w - ee_left, dim=1)
    object_ee_distance_right = torch.norm(obj_pos_w - ee_right, dim=1)
    object_ee_distance = torch.minimum(object_ee_distance_left, object_ee_distance_right)

    return dapl_tanh_proximity_reward(
        object_ee_distance, standard_deviation=std
    )


def object_goal_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    obj_ee_distance_threshold: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    obj_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_left = ee_frame.data.target_pos_w[..., 1, :]
    ee_right = ee_frame.data.target_pos_w[..., 2, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance_left = torch.norm(obj_pos_w - ee_left, dim=1)
    object_ee_distance_right = torch.norm(obj_pos_w - ee_right, dim=1)
    object_ee_distance = torch.minimum(object_ee_distance_left, object_ee_distance_right)
    obj_ee_dist_cond = object_ee_distance < obj_ee_distance_threshold
    
    # Get target position and orientation in environment coordinates
    des_pos_env = command[:, :3]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_env = object_pos_w - env.scene.env_origins

    # Position distance
    pos_distance = torch.norm(des_pos_env - object_pos_env, dim=1)

    des_rot_env = command[:, 3:7]
    object_quat_w = object.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    dot_product = torch.sum(object_quat_w * des_rot_env, dim=1)
    dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
    ang_distance = 2 * torch.acos(dot_product)
    ang_distance = torch.clamp(ang_distance, max=torch.pi)
    pose_distance = dapl_combined_pose_error(pos_distance, ang_distance)

    return obj_ee_dist_cond * dapl_tanh_proximity_reward(
        pose_distance, standard_deviation=std
    )

def joint_power_penalty(
    env: ManagerBasedRLEnv,
    k_e: float = 0.0001,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Energy penalty
    
    Penalty form: c_energy = k_e * Σ(τ_i * q̇_i)
    where τ_i is joint torque and q̇_i is joint velocity.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    
    # Get joint torques and velocities
    joint_torques = robot.data.applied_torque  # (num_envs, num_joints)
    joint_velocities = robot.data.joint_vel    # (num_envs, num_joints)
    
    # Calculate power for each joint: τ * q̇
    joint_power = joint_torques * joint_velocities  # (num_envs, num_joints)
    
    # Sum over all joints and apply scaling
    total_power = torch.sum(torch.abs(joint_power), dim=1)  # (num_envs,)
    
    penalty = k_e * total_power
    
    return penalty


def task_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    planar: bool = False,  # If True, only consider x and y position
    base_reward: float = 1.0,  # Base reward for success
) -> torch.Tensor:
    """Task success reward: gives base_reward when object reaches target pose within thresholds."""
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Get target pose
    des_pos_env = command[:, :3]  # target position in environment coordinates
    des_rot_env = command[:, 3:7]  # target orientation (quaternion) in environment coordinates
    
    # Get current object position in environment coordinates
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_env = object_pos_w - env.scene.env_origins
    
    # Get current object orientation and convert to euler
    object_quat_w = object.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    position_distance = torch.norm(des_pos_env[:, :2] - object_pos_env[:, :2], dim=1)
    
    if planar:
        # position_distance = torch.norm(des_pos_env[:, :2] - object_pos_env[:, :2], dim=1)
        position_reached = position_distance < threshold
        return position_reached.float()

    # position_distance = torch.norm(des_pos_env - object_pos_env, dim=1)
    
    dot_product = torch.sum(object_quat_w * des_rot_env, dim=1)
    dot_product = torch.clamp(torch.abs(dot_product), max=1.0)
    angular_distance = 2 * torch.acos(dot_product)
    
    # Check if within thresholds
    position_reached = position_distance < threshold
    rotation_reached = angular_distance < rotation_threshold
    
    # Calculate success mask
    success_mask = position_reached & rotation_reached
    
    # Final reward: base_reward for success, 0.0 for failure
    reward = torch.where(
        success_mask,
        base_reward,
        torch.zeros_like(success_mask, dtype=torch.float)
    )

    return reward


def clutter_task_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
    position_threshold: float = 0.05,
    rotation_threshold: float = 0.1,
    maximum_obstacle_translation: float = 0.2,
    maximum_obstacle_rotation: float = torch.pi,
    base_reward: float = 1.0,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    obstacles_cfg: SceneEntityCfg = SceneEntityCfg("obstacles"),
) -> torch.Tensor:
    """Paper success reward discounted by non-target object motion.

    DAPL specifies planar target success and averages the translational and
    angular offsets of all non-target objects.  The paper does not report the
    two normalization constants, so they remain explicit environment
    parameters instead of being hidden inside this term.
    """

    if maximum_obstacle_translation <= 0.0 or maximum_obstacle_rotation <= 0.0:
        raise ValueError("obstacle-motion normalization constants must be positive")
    if not hasattr(env, "_clutter_initial_obstacle_pose"):
        raise RuntimeError(
            "initial clutter poses are missing; configure reset_clutter_from_manifest"
        )

    target: RigidObject = env.scene[target_cfg.name]
    obstacles: RigidObjectCollection = env.scene[obstacles_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_pos_env = target.data.root_pos_w[:, :3] - env.scene.env_origins
    success = planar_pose_success(
        target_pos_env,
        target.data.root_quat_w,
        command,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
    )

    initial_pose = env._clutter_initial_obstacle_pose
    obstacle_pos_env = obstacles.data.object_pos_w - env.scene.env_origins.unsqueeze(1)
    translation = torch.linalg.vector_norm(
        obstacle_pos_env - initial_pose[..., :3], dim=-1
    ).mean(dim=1)
    obstacle_dot = torch.sum(
        obstacles.data.object_quat_w * initial_pose[..., 3:7], dim=-1
    )
    rotation = (
        2.0 * torch.acos(torch.clamp(torch.abs(obstacle_dot), max=1.0))
    ).mean(dim=1)

    normalized_translation = torch.clamp(
        translation / maximum_obstacle_translation, min=0.0, max=1.0
    )
    normalized_rotation = torch.clamp(
        rotation / maximum_obstacle_rotation, min=0.0, max=1.0
    )
    motion = 0.5 * (normalized_translation + normalized_rotation)
    scale = torch.clamp(1.0 - 0.5 * motion, min=0.5, max=1.0)
    return success.to(dtype=scale.dtype) * scale * base_reward
