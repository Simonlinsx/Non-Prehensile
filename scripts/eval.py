# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RSL-RL agent and report success rate."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL and report success rate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1248, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=2000, help="Number of episodes to evaluate.")
parser.add_argument("--max_episode_steps", type=int, default=300, help="Safety cap on episode length (steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation (num_envs must be 1).")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=1_000_000, help="Interval between videos (unused in eval, kept for parity).")
parser.add_argument("--video_folder", type=str, default=None, help="Optional directory for evaluation videos.")
parser.add_argument("--video_name_prefix", type=str, default="eval", help="Filename prefix for evaluation videos.")
parser.add_argument("--output_dir", type=str, default=None, help="Optional directory for JSON/CSV evaluation results.")
parser.add_argument(
    "--camera_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional world-frame viewer camera position for video evaluation.",
)
parser.add_argument(
    "--camera_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Optional world-frame viewer camera target for video evaluation.",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use the policy mean action instead of sampling training-time action noise.",
)
parser.add_argument(
    "--visualize_goal",
    action="store_true",
    default=False,
    help="Overlay a goal frame and a green marker on the current target object in videos.",
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible (single env).")
# checkpoint selection
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
# append RSL-RL cli arguments (includes --checkpoint)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime
import json
import csv
from tqdm import tqdm

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import IsaacLab_nonPrehensile.tasks  # noqa: F401


def _extract_asset_names_from_env(env) -> list[str]:
    """Extract asset base names from env config's MultiAssetSpawnerCfg.

    Assumes env.unwrapped.cfg.scene.object.spawn.assets_cfg exists and each has a usd_path like .../<name>/<name>.usd.
    """
    names: list[str] = []
    cfg = getattr(env, "unwrapped", env)
    scene = getattr(cfg, "cfg", None)
    if scene is None:
        scene = getattr(env, "cfg", None)
    if scene is None:
        return names
    scene_cfg = getattr(scene, "scene", None)
    if scene_cfg is None:
        return names
    object_cfg = getattr(scene_cfg, "object", None)
    spawn_cfg = getattr(object_cfg, "spawn", None) if object_cfg is not None else None
    assets_cfg = getattr(spawn_cfg, "assets_cfg", None) if spawn_cfg is not None else None
    if assets_cfg is None:
        return names
    for usd_cfg in assets_cfg:
        usd_path = getattr(usd_cfg, "usd_path", None)
        if isinstance(usd_path, str) and len(usd_path) > 0:
            base = os.path.basename(os.path.dirname(usd_path))
            names.append(base)
        else:
            names.append("unknown")
    return names


def _build_env_to_object_index(num_envs: int, num_assets: int) -> torch.Tensor:
    """Deterministic mapping from env_id to asset index when random_choice=False."""
    if num_assets <= 0:
        return torch.full((num_envs,), -1, dtype=torch.long)
    idx = torch.arange(num_envs, dtype=torch.long)
    return idx % num_assets


def _clutter_diagnostic_snapshot(env, actions, rewards, step: int, initial_target_pose: torch.Tensor) -> dict:
    """Capture privileged single-environment diagnostics for a video frame."""
    base_env = env.unwrapped
    target = base_env.scene["target"]
    ee_frame = base_env.scene["ee_frame"]
    command = base_env.command_manager.get_command("target_object_pose")

    target_pos_env = target.data.root_pos_w[0, :3] - base_env.scene.env_origins[0]
    target_quat = target.data.root_quat_w[0]
    goal_pos = command[0, :3]
    goal_quat = command[0, 3:7]

    planar_error = torch.linalg.vector_norm(goal_pos[:2] - target_pos_env[:2])
    goal_dot = torch.clamp(torch.abs(torch.sum(target_quat * goal_quat)), max=1.0)
    rotation_error = 2.0 * torch.acos(goal_dot)

    ee_targets = ee_frame.data.target_pos_w[0]
    finger_distance = torch.minimum(
        torch.linalg.vector_norm(target.data.root_pos_w[0, :3] - ee_targets[1]),
        torch.linalg.vector_norm(target.data.root_pos_w[0, :3] - ee_targets[2]),
    )

    initial_pos = initial_target_pose[:3]
    initial_quat = initial_target_pose[3:7]
    target_translation = torch.linalg.vector_norm(target_pos_env[:2] - initial_pos[:2])
    initial_dot = torch.clamp(torch.abs(torch.sum(target_quat * initial_quat)), max=1.0)
    target_rotation = 2.0 * torch.acos(initial_dot)

    obstacle_translation = torch.tensor(0.0, device=base_env.device)
    obstacle_rotation = torch.tensor(0.0, device=base_env.device)
    if hasattr(base_env, "_clutter_initial_obstacle_pose"):
        obstacles = base_env.scene["obstacles"]
        obstacle_pos_env = obstacles.data.object_pos_w[0] - base_env.scene.env_origins[0].unsqueeze(0)
        initial_obstacles = base_env._clutter_initial_obstacle_pose[0]
        obstacle_translation = torch.linalg.vector_norm(
            obstacle_pos_env - initial_obstacles[:, :3], dim=-1
        ).mean()
        obstacle_dot = torch.sum(obstacles.data.object_quat_w[0] * initial_obstacles[:, 3:7], dim=-1)
        obstacle_rotation = (
            2.0 * torch.acos(torch.clamp(torch.abs(obstacle_dot), max=1.0))
        ).mean()

    return {
        "step": int(step),
        "reward": float(rewards[0].item()),
        "action_l2": float(torch.linalg.vector_norm(actions[0]).item()),
        "ee_target_distance_m": float(finger_distance.item()),
        "target_goal_planar_error_m": float(planar_error.item()),
        "target_goal_rotation_error_rad": float(rotation_error.item()),
        "target_translation_from_start_m": float(target_translation.item()),
        "target_rotation_from_start_rad": float(target_rotation.item()),
        "obstacle_mean_translation_m": float(obstacle_translation.item()),
        "obstacle_mean_rotation_rad": float(obstacle_rotation.item()),
    }


def _create_clutter_video_markers(env):
    """Create non-physical target and goal overlays for evaluation videos."""
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG

    goal_cfg = FRAME_MARKER_CFG.copy()
    goal_cfg.prim_path = "/Visuals/Eval/GoalPose"
    goal_cfg.markers["frame"].scale = (0.16, 0.16, 0.16)
    goal_marker = VisualizationMarkers(goal_cfg)

    target_cfg = SPHERE_MARKER_CFG.copy()
    target_cfg.prim_path = "/Visuals/Eval/CurrentTarget"
    target_cfg.markers["sphere"].radius = 0.025
    target_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
    target_marker = VisualizationMarkers(target_cfg)

    _update_clutter_video_markers(env, goal_marker, target_marker)
    return goal_marker, target_marker


def _update_clutter_video_markers(env, goal_marker, target_marker) -> None:
    base_env = env.unwrapped
    command = base_env.command_manager.get_command("target_object_pose")
    goal_pos_w = command[:, :3] + base_env.scene.env_origins
    goal_marker.visualize(translations=goal_pos_w, orientations=command[:, 3:7])
    target_marker.visualize(translations=base_env.scene["target"].data.root_pos_w[:, :3])


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate RSL-RL agent and report success rate over multiple episodes."""
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed

    if args_cli.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    if args_cli.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
    if args_cli.camera_eye is not None or args_cli.camera_lookat is not None:
        env_cfg.viewer.origin_type = "world"

    # Disable observation noise for evaluation
    env_cfg.disable_obs_noise = True

    # Limitations for video recording in evaluation
    if args_cli.video and env_cfg.scene.num_envs != 1:
        print("[WARN] Video recording in eval supports only num_envs=1. Overriding num_envs to 1.")
        env_cfg.scene.num_envs = 1

    # specify directory for loading experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # resolve resume path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    video_markers = _create_clutter_video_markers(env) if args_cli.visualize_goal else None

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_folder = args_cli.video_folder or os.path.join(log_dir, "videos", "eval")
        video_folder = os.path.abspath(video_folder)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,  # record first episode by default
            "video_length": min(args_cli.video_length, args_cli.max_episode_steps),
            "name_prefix": args_cli.video_name_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy_func = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy_obj = ppo_runner.alg.policy  # Get the actual policy object for act() method

    # evaluation loop (vectorized)
    num_envs = env.unwrapped.num_envs
    episodes_completed = 0
    total_successes = 0
    
    # Get initial stats from env - require them to be available
    if not hasattr(env.unwrapped, 'total_episodes') or not hasattr(env.unwrapped, 'total_successes'):
        raise AttributeError("Environment does not have total_episodes or total_successes. This eval script requires the NonPrehensileEnv with success tracking.")
    
    episodes_completed = env.unwrapped.total_episodes
    total_successes = env.unwrapped.total_successes
    

    # per-object accounting
    asset_names = _extract_asset_names_from_env(env)
    num_assets = len(asset_names)
    env_to_obj_idx = _build_env_to_object_index(num_envs, num_assets)
    # counters stored on CPU for printing simplicity
    obj_episodes = {name: 0 for name in asset_names}
    obj_successes = {name: 0 for name in asset_names}

    # reset environment
    obs, _ = env.get_observations()
    if video_markers is not None:
        _update_clutter_video_markers(env, *video_markers)
    target = env.unwrapped.scene["target"]
    initial_target_pose = torch.cat(
        (
            target.data.root_pos_w[0, :3] - env.unwrapped.scene.env_origins[0],
            target.data.root_quat_w[0],
        )
    ).clone()
    diagnostic_trace: list[dict] = []

    # timing
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None

    # Initialize progress bar
    pbar = tqdm(total=args_cli.num_episodes, desc="Evaluating", unit="episodes")
    pbar.set_postfix({
        "Success Rate": "0.00%",
        "Episodes": 0,
        "Successes": 0
    })

    # Run until desired number of episodes are completed
    step_count = 0
    while episodes_completed < args_cli.num_episodes and simulation_app.is_running():
        start_time = time.time()
        step_count += 1
        
        with torch.inference_mode():
            if args_cli.deterministic:
                actions = policy_func(obs)
            else:
                # Match training-time exploration by sampling the action distribution.
                actions = policy_obj.act(obs)
            obs, rewards, dones, _ = env.step(actions)

        if video_markers is not None:
            _update_clutter_video_markers(env, *video_markers)

        sample_diagnostic = (
            step_count == 1
            or step_count % 30 == 0
            or (step_count + 1) % args_cli.max_episode_steps == 0
        )
        # ManagerBasedRLEnv auto-resets ended environments inside step(), so a
        # post-done snapshot would describe the next episode rather than the
        # just-recorded video.
        if sample_diagnostic and not torch.any(dones):
            diagnostic_trace.append(
                _clutter_diagnostic_snapshot(env, actions, rewards, step_count, initial_target_pose)
            )

        # Use env's built-in success tracking - require it to be available
        if not hasattr(env.unwrapped, 'episode_success_buf'):
            raise AttributeError("Environment does not have episode_success_buf. This eval script requires the NonPrehensileEnv with success tracking.")

        # Use environment's episode ending signals directly
        # RslRlVecEnvWrapper already combines terminated | truncated into dones
        ended = dones.bool()
        

        if torch.any(ended):
            ended_ids = torch.where(ended)[0]
            # Use env's built-in statistics - require them to be available
            if not hasattr(env.unwrapped, 'total_episodes') or not hasattr(env.unwrapped, 'total_successes'):
                raise AttributeError("Environment does not have total_episodes or total_successes. This eval script requires the NonPrehensileEnv with success tracking.")
            
            episodes_completed = env.unwrapped.total_episodes
            total_successes = env.unwrapped.total_successes
            
            # per-object accumulation
            for env_id in ended_ids.tolist():
                if 0 <= num_assets and num_assets > 0:
                    obj_idx = int(env_to_obj_idx[env_id].item()) if env_to_obj_idx.numel() == num_envs else -1
                    if 0 <= obj_idx < num_assets:
                        obj_name = asset_names[obj_idx]
                        obj_episodes[obj_name] = obj_episodes.get(obj_name, 0) + 1
                        # Use env's success status before reset
                        if hasattr(env.unwrapped, '_episode_success_before_reset'):
                            env_success = bool(env.unwrapped._episode_success_before_reset[env_id].item())
                        else:
                            # Fallback: use current episode_success_buf (may be reset)
                            env_success = bool(env.unwrapped.episode_success_buf[env_id].item())
                        
                        if env_success:
                            obj_successes[obj_name] = obj_successes.get(obj_name, 0) + 1
                        # Debug: print first few per-object stats
                        if episodes_completed <= 5:
                            print(f"[DEBUG] Env {env_id} (obj {obj_name}): success={env_success}")
            
            # Update progress bar
            current_success_rate = (total_successes / episodes_completed) * 100 if episodes_completed > 0 else 0.0
            pbar.update(ended_ids.numel())
            pbar.set_postfix({
                "Success Rate": f"{current_success_rate:.2f}%",
                "Episodes": episodes_completed,
                "Successes": total_successes
            })


        # optional real-time pacing
        if args_cli.real_time and (dt is not None):
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    success_rate = (total_successes / episodes_completed) if episodes_completed > 0 else 0.0

    # Close progress bar
    pbar.close()

    # persist results
    results_dir = os.path.abspath(args_cli.output_dir) if args_cli.output_dir else log_dir
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "eval_summary.json")
    per_object_path = os.path.join(results_dir, "eval_per_object.csv")

    results_payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "seed": args_cli.seed,
        "deterministic": bool(args_cli.deterministic),
        "episodes": int(episodes_completed),
        "successes": int(total_successes),
        "success_rate": float(success_rate),
        "diagnostic_trace": diagnostic_trace,
        "diagnostic_summary": {
            "min_ee_target_distance_m": min(
                (sample["ee_target_distance_m"] for sample in diagnostic_trace), default=None
            ),
            "min_target_goal_planar_error_m": min(
                (sample["target_goal_planar_error_m"] for sample in diagnostic_trace), default=None
            ),
            "min_target_goal_rotation_error_rad": min(
                (sample["target_goal_rotation_error_rad"] for sample in diagnostic_trace), default=None
            ),
            "max_target_translation_from_start_m": max(
                (sample["target_translation_from_start_m"] for sample in diagnostic_trace), default=None
            ),
            "max_obstacle_mean_translation_m": max(
                (sample["obstacle_mean_translation_m"] for sample in diagnostic_trace), default=None
            ),
            "final": diagnostic_trace[-1] if diagnostic_trace else None,
        },
        "per_object": [
            {
                "name": name,
                "episodes": int(obj_episodes.get(name, 0)),
                "successes": int(obj_successes.get(name, 0)),
                "success_rate": (float(obj_successes.get(name, 0)) / float(obj_episodes.get(name, 0))) if obj_episodes.get(name, 0) > 0 else 0.0,
            }
            for name in sorted(obj_episodes.keys())
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    # write CSV for per-object breakdown
    with open(per_object_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["name", "episodes", "successes", "success_rate"])
        for name in sorted(obj_episodes.keys()):
            ep = int(obj_episodes.get(name, 0))
            sc = int(obj_successes.get(name, 0))
            rate = (sc / ep) if ep > 0 else 0.0
            writer.writerow([name, ep, sc, rate])

    # print summary
    print("\n========== Evaluation Summary ==========")
    print(f"Task: {args_cli.task}")
    print(f"Checkpoint: {resume_path}")
    print(f"Episodes: {episodes_completed}")
    print(f"Successes: {total_successes}")
    print(f"Success Rate: {success_rate * 100:.2f}%")
    if diagnostic_trace:
        diagnostic_summary = results_payload["diagnostic_summary"]
        print(
            "Diagnostics: "
            f"min finger-target={diagnostic_summary['min_ee_target_distance_m']:.4f} m, "
            f"min planar goal error={diagnostic_summary['min_target_goal_planar_error_m']:.4f} m, "
            f"min rotation goal error={diagnostic_summary['min_target_goal_rotation_error_rad']:.4f} rad"
        )
    print(f"Saved: {summary_path}")
    if len(obj_episodes) > 0:
        print(f"Saved: {per_object_path}")
    if num_assets > 0 and len(obj_episodes) > 0:
        print("\nPer-object success rates:")
        for name in sorted(obj_episodes.keys()):
            ep = obj_episodes[name]
            sc = obj_successes.get(name, 0)
            rate = (sc / ep) * 100.0 if ep > 0 else 0.0
            print(f"  - {name}: {sc}/{ep} ({rate:.2f}%)")
    print("=======================================\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
