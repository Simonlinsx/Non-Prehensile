#!/usr/bin/env python3
"""Audit legal safe-contact IK reachability for direction-endpoint scenes.

This is a diagnostic, not a controller.  It searches rigid hand-cloud poses
whose sampled hand geometry touches the oracle safe set while maintaining the
audited C1 margin to every non-safe target point, then checks whether the
Franka can realize one of those poses within its joint limits.  Proximal-link
PhysX contact is intentionally left for a later simulator replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-AffordanceTeacher-T0-Soft-Franka-v0")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--settle-steps", type=int, default=5)
parser.add_argument("--endpoint-angle-deg", type=float, default=70.0)
parser.add_argument("--contact-distance-m", type=float, default=0.010)
parser.add_argument("--forbidden-clearance-m", type=float, default=0.010)
parser.add_argument("--safe-point-count", type=int, default=20)
parser.add_argument("--hand-point-count", type=int, default=24)
parser.add_argument("--candidate-chunk-size", type=int, default=48)
parser.add_argument("--ik-candidate-count", type=int, default=12)
parser.add_argument("--ik-max-evaluations", type=int, default=600)
parser.add_argument("--path-steps", type=int, default=41)
parser.add_argument(
    "--wrist-yaw-deg",
    type=float,
    nargs="+",
    default=(-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0),
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import pinocchio as pin
import torch
from scipy.optimize import least_squares

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
    get_end_effector_pointcloud_in_env_frame,
    get_object_pointcloud_in_env_frame,
)


FRANKA_URDF = (
    "/data1/linsixu/IsaacLab-2.2.0/source/isaaclab/isaaclab/"
    "controllers/config/data/lula_franka_gen.urdf"
)


def _yaw_quaternion(
    yaw_rad: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    half = 0.5 * yaw_rad
    return torch.tensor(
        (math.cos(half), 0.0, 0.0, math.sin(half)),
        device=device,
        dtype=dtype,
    )


def _direction_bin(angle_deg: float) -> str:
    if angle_deg < -70.0:
        return "[-90,-70)"
    if angle_deg < -35.0:
        return "[-70,-35)"
    if angle_deg < 0.0:
        return "[-35,0)"
    if angle_deg < 35.0:
        return "[0,35)"
    if angle_deg < 70.0:
        return "[35,70)"
    return "[70,90]"


def main() -> None:
    if args_cli.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    if args_cli.safe_point_count <= 0 or args_cli.hand_point_count <= 0:
        raise ValueError("point counts must be positive")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.use_torch_compile = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base = env.unwrapped
        for _ in range(args_cli.settle_steps):
            env.step(torch.zeros((base.num_envs, 7), device=base.device))

        target_cfg = SceneEntityCfg("target")
        target_points = get_object_pointcloud_in_env_frame(base, target_cfg).reshape(
            base.num_envs, 512, 3
        )
        features = mdp.domino_target_affordance(base, target_cfg).reshape(
            base.num_envs, 512, 2
        )
        safe_mask = features[..., 0] >= 0.25
        if not torch.all(torch.any(safe_mask, dim=1)):
            raise RuntimeError("every scene must contain at least one safe target point")

        hand_points = get_end_effector_pointcloud_in_env_frame(base)
        ee_frame = base.scene["ee_frame"]
        tcp_position = ee_frame.data.target_pos_w[:, 0] - base.scene.env_origins
        tcp_quaternion = ee_frame.data.target_quat_w[:, 0]
        local_hand = quat_apply_inverse(
            tcp_quaternion[:, None, :]
            .expand(-1, hand_points.shape[1], -1)
            .reshape(-1, 4),
            (hand_points - tcp_position[:, None, :]).reshape(-1, 3),
        ).reshape_as(hand_points)
        # The local cache is identical across replicated environments.  Points
        # furthest from the TCP cover both finger tips and their outer faces.
        hand_candidate_count = min(args_cli.hand_point_count, local_hand.shape[1])
        hand_contact_indices = torch.topk(
            torch.linalg.vector_norm(local_hand[0], dim=1),
            k=hand_candidate_count,
        ).indices

        goal = base.command_manager.get_command("target_object_pose")
        target_position = base.scene["target"].data.root_pos_w[:, :3] - base.scene.env_origins
        displacement = goal[:, :2] - target_position[:, :2]
        direction = displacement / torch.clamp(
            torch.linalg.vector_norm(displacement, dim=1, keepdim=True), min=1.0e-8
        )
        direction_deg = torch.rad2deg(torch.atan2(direction[:, 1], direction[:, 0]))

        model = pin.buildModelFromUrdf(FRANKA_URDF)
        frame_id = model.getFrameId("panda_hand")
        lower = model.lowerPositionLimit[:7] + 1.0e-4
        upper = model.upperPositionLimit[:7] - 1.0e-4
        robot = base.scene["robot"]
        joint_ids, _ = robot.find_joints("panda_joint.*", preserve_order=True)
        q_initial_np = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()

        def forward(q_arm: np.ndarray) -> pin.SE3:
            data = model.createData()
            q_full = np.concatenate((q_arm, np.array((0.04, 0.04))))
            pin.forwardKinematics(model, data, q_full)
            pin.updateFramePlacements(model, data)
            return data.oMf[frame_id]

        initial_pin_pose = forward(q_initial_np)
        scene_rows: list[dict[str, object]] = []
        endpoint_mask = torch.abs(direction_deg) >= args_cli.endpoint_angle_deg
        endpoint_ids = torch.nonzero(endpoint_mask, as_tuple=False).flatten().tolist()

        for env_id in endpoint_ids:
            points = target_points[env_id]
            is_safe = safe_mask[env_id]
            safe_points = points[is_safe]
            safe_center = safe_points.mean(dim=0)
            trailing_score = torch.sum(
                (safe_points[:, :2] - safe_center[:2]) * direction[env_id], dim=1
            )
            safe_count = min(args_cli.safe_point_count, safe_points.shape[0])
            trailing_points = safe_points[
                torch.topk(-trailing_score, k=safe_count).indices
            ]
            forbidden_points = points[~is_safe]
            geometry_candidates: list[dict[str, object]] = []

            for yaw_deg in args_cli.wrist_yaw_deg:
                yaw_quat = _yaw_quaternion(
                    math.radians(yaw_deg),
                    device=points.device,
                    dtype=points.dtype,
                )
                candidate_quaternion = quat_mul(yaw_quat, tcp_quaternion[env_id])
                rotated_hand = quat_apply(
                    candidate_quaternion[None, :].expand(local_hand.shape[1], -1),
                    local_hand[env_id],
                )
                contact_offsets = rotated_hand[hand_contact_indices]
                candidate_tcp = (
                    trailing_points[:, None, :] - contact_offsets[None, :, :]
                ).reshape(-1, 3)

                for start in range(0, candidate_tcp.shape[0], args_cli.candidate_chunk_size):
                    candidate_chunk = candidate_tcp[
                        start : start + args_cli.candidate_chunk_size
                    ]
                    posed_hand = (
                        rotated_hand[None, :, :] + candidate_chunk[:, None, :]
                    )
                    safe_distance = torch.cdist(
                        posed_hand,
                        safe_points[None, :, :].expand(posed_hand.shape[0], -1, -1),
                    ).amin(dim=(1, 2))
                    if forbidden_points.numel() == 0:
                        forbidden_distance = torch.full_like(safe_distance, torch.inf)
                    else:
                        forbidden_distance = torch.cdist(
                            posed_hand,
                            forbidden_points[None, :, :].expand(
                                posed_hand.shape[0], -1, -1
                            ),
                        ).amin(dim=(1, 2))
                    legal = (
                        safe_distance <= args_cli.contact_distance_m
                    ) & (forbidden_distance > args_cli.forbidden_clearance_m)
                    for local_index in torch.nonzero(legal, as_tuple=False).flatten().tolist():
                        tcp = candidate_chunk[local_index]
                        score = float(
                            torch.linalg.vector_norm(tcp - tcp_position[env_id]).item()
                            + 0.0002 * abs(float(yaw_deg))
                        )
                        geometry_candidates.append(
                            {
                                "score": score,
                                "yaw_deg": float(yaw_deg),
                                "tcp": tcp.detach().cpu().numpy(),
                                "safe_distance_m": float(safe_distance[local_index].item()),
                                "forbidden_clearance_m": float(
                                    forbidden_distance[local_index].item()
                                ),
                            }
                        )

            geometry_candidates.sort(key=lambda item: float(item["score"]))
            ik_success = False
            best_ik_position_error = math.inf
            best_ik_rotation_error = math.inf
            best_yaw_deg: float | None = None
            best_q_arm: np.ndarray | None = None
            best_clearance = (
                max(
                    (float(item["forbidden_clearance_m"]) for item in geometry_candidates),
                    default=-math.inf,
                )
            )
            for candidate in geometry_candidates[: args_cli.ik_candidate_count]:
                desired_tcp = np.asarray(candidate["tcp"], dtype=np.float64)
                desired_translation = initial_pin_pose.translation + (
                    desired_tcp - tcp_position[env_id].detach().cpu().numpy()
                )
                yaw_rad = math.radians(float(candidate["yaw_deg"]))
                desired_rotation = pin.rpy.rpyToMatrix(0.0, 0.0, yaw_rad) @ initial_pin_pose.rotation

                def residual(q_arm: np.ndarray) -> np.ndarray:
                    pose = forward(q_arm)
                    return np.concatenate(
                        (
                            20.0 * (pose.translation - desired_translation),
                            pin.log3(pose.rotation.T @ desired_rotation),
                            1.0e-3 * (q_arm - q_initial_np),
                        )
                    )

                result = least_squares(
                    residual,
                    q_initial_np,
                    bounds=(lower, upper),
                    max_nfev=args_cli.ik_max_evaluations,
                    ftol=1.0e-10,
                    xtol=1.0e-10,
                    gtol=1.0e-10,
                )
                solved_pose = forward(result.x)
                position_error = float(
                    np.linalg.norm(solved_pose.translation - desired_translation)
                )
                rotation_error = float(
                    np.linalg.norm(
                        pin.log3(solved_pose.rotation.T @ desired_rotation)
                    )
                )
                if position_error < best_ik_position_error:
                    best_ik_position_error = position_error
                    best_ik_rotation_error = rotation_error
                    best_yaw_deg = float(candidate["yaw_deg"])
                if position_error <= 1.0e-3 and rotation_error <= 1.0e-2:
                    ik_success = True
                    best_q_arm = result.x.copy()
                    break

            direct_path_min_forbidden_clearance = math.inf
            direct_path_semantic_c1_free = False
            if best_q_arm is not None:
                initial_hand_np = hand_points[env_id].detach().cpu().numpy()
                initial_tcp_np = tcp_position[env_id].detach().cpu().numpy()
                forbidden_np = forbidden_points.detach().cpu().numpy()
                safe_np = safe_points.detach().cpu().numpy()
                final_safe_distance = math.inf
                for alpha in np.linspace(0.0, 1.0, args_cli.path_steps):
                    q_arm = (1.0 - alpha) * q_initial_np + alpha * best_q_arm
                    path_pose = forward(q_arm)
                    relative_rotation = (
                        path_pose.rotation @ initial_pin_pose.rotation.T
                    )
                    tcp_path = initial_tcp_np + (
                        path_pose.translation - initial_pin_pose.translation
                    )
                    posed_hand = (
                        relative_rotation
                        @ (initial_hand_np - initial_tcp_np).T
                    ).T + tcp_path
                    if forbidden_np.size:
                        step_clearance = float(
                            np.linalg.norm(
                                posed_hand[:, None, :] - forbidden_np[None, :, :],
                                axis=2,
                            ).min()
                        )
                        direct_path_min_forbidden_clearance = min(
                            direct_path_min_forbidden_clearance, step_clearance
                        )
                    if alpha == 1.0:
                        final_safe_distance = float(
                            np.linalg.norm(
                                posed_hand[:, None, :] - safe_np[None, :, :],
                                axis=2,
                            ).min()
                        )
                direct_path_semantic_c1_free = bool(
                    direct_path_min_forbidden_clearance
                    > args_cli.forbidden_clearance_m
                    and final_safe_distance <= args_cli.contact_distance_m
                )

            angle = float(direction_deg[env_id].item())
            scene_rows.append(
                {
                    "environment_index": env_id,
                    "goal_direction_deg": angle,
                    "direction_bin": _direction_bin(angle),
                    "geometry_legal_candidate_count": len(geometry_candidates),
                    "geometry_legal": bool(geometry_candidates),
                    "ik_reachable": ik_success,
                    "direct_joint_path_semantic_c1_free": direct_path_semantic_c1_free,
                    "best_candidate_yaw_deg": best_yaw_deg,
                    "best_forbidden_clearance_m": (
                        best_clearance if math.isfinite(best_clearance) else None
                    ),
                    "best_ik_position_error_m": (
                        best_ik_position_error
                        if math.isfinite(best_ik_position_error)
                        else None
                    ),
                    "best_ik_rotation_error_rad": (
                        best_ik_rotation_error
                        if math.isfinite(best_ik_rotation_error)
                        else None
                    ),
                    "direct_joint_path_min_forbidden_clearance_m": (
                        direct_path_min_forbidden_clearance
                        if math.isfinite(direct_path_min_forbidden_clearance)
                        else None
                    ),
                }
            )

        bin_rows: dict[str, dict[str, int]] = {}
        for row in scene_rows:
            stats = bin_rows.setdefault(
                str(row["direction_bin"]),
                {
                    "scenes": 0,
                    "geometry_legal": 0,
                    "ik_reachable": 0,
                    "direct_joint_path_semantic_c1_free": 0,
                },
            )
            stats["scenes"] += 1
            stats["geometry_legal"] += int(bool(row["geometry_legal"]))
            stats["ik_reachable"] += int(bool(row["ik_reachable"]))
            stats["direct_joint_path_semantic_c1_free"] += int(
                bool(row["direct_joint_path_semantic_c1_free"])
            )

        payload = {
            "task": args_cli.task,
            "endpoint_angle_deg": args_cli.endpoint_angle_deg,
            "contact_distance_m": args_cli.contact_distance_m,
            "forbidden_clearance_m": args_cli.forbidden_clearance_m,
            "wrist_yaw_deg": list(args_cli.wrist_yaw_deg),
            "scope_note": (
                "Necessary-condition semantic hand-cloud and joint-limit IK audit; "
                "proximal-link PhysX collision is not certified here."
            ),
            "endpoint_scenes": len(scene_rows),
            "geometry_legal_scenes": sum(
                int(bool(row["geometry_legal"])) for row in scene_rows
            ),
            "ik_reachable_scenes": sum(
                int(bool(row["ik_reachable"])) for row in scene_rows
            ),
            "direct_joint_path_semantic_c1_free_scenes": sum(
                int(bool(row["direct_joint_path_semantic_c1_free"]))
                for row in scene_rows
            ),
            "per_direction_bin": bin_rows,
            "per_scene": scene_rows,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({key: payload[key] for key in (
            "endpoint_scenes", "geometry_legal_scenes", "ik_reachable_scenes",
            "direct_joint_path_semantic_c1_free_scenes",
            "per_direction_bin",
        )}, indent=2), flush=True)
        print(f"Saved: {args_cli.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
