"""Collect aligned DAPL world-model transitions from the Clutter6D task."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Clutter6D-Franka-v0")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--seed", type=int, default=17)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/dapl_transitions"))
parser.add_argument("--shard-transitions", type=int, default=512)
parser.add_argument(
    "--action-mode",
    choices=("random", "zero", "policy"),
    default="random",
    help="Use correlated random, stationary, or checkpoint-policy actions.",
)
parser.add_argument("--random-action-std", type=float, default=0.15)
parser.add_argument("--random-action-decay", type=float, default=0.85)
parser.add_argument(
    "--checkpoint",
    type=Path,
    help="RSL-RL checkpoint required by --action-mode=policy.",
)
parser.add_argument(
    "--policy-action",
    choices=("sample", "mean"),
    default="sample",
    help="Sample the PPO action distribution or use its deterministic mean.",
)
parser.add_argument(
    "--initial-task-mode",
    choices=("distributed", "first"),
    default="distributed",
    help="Distribute parallel environments across manifest tasks or start all at task 0.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Transition collection requires the non-concatenated physical scene group.
os.environ["DAPL_ENABLE_WORLD_MODEL_OBSERVATION"] = "1"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import IsaacLab_nonPrehensile.tasks  # noqa: F401
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp.observations import (
    build_dapl_physical_scene,
)


def _validate_args() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")
    if args_cli.shard_transitions <= 0:
        raise ValueError("--shard-transitions must be positive")
    if args_cli.random_action_std < 0.0:
        raise ValueError("--random-action-std must be non-negative")
    if not 0.0 <= args_cli.random_action_decay < 1.0:
        raise ValueError("--random-action-decay must be in [0, 1)")
    if args_cli.action_mode == "policy" and args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required by --action-mode=policy")
    if args_cli.checkpoint is not None and not args_cli.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"policy checkpoint does not exist: {args_cli.checkpoint}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TransitionShardWriter:
    """Write bounded CPU shards without retaining a full rollout in memory."""

    def __init__(
        self,
        output_dir: Path,
        shard_transitions: int,
        control_dt_s: float,
        rollout_metadata: dict[str, str | int | float | None],
    ):
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = tuple(self.output_dir.glob("transitions_*.pt"))
        if existing:
            raise FileExistsError(
                f"refusing to mix transition runs in {self.output_dir}; "
                "choose a new --output-dir"
            )
        self.shard_transitions = shard_transitions
        self.control_dt_s = control_dt_s
        self.rollout_metadata = dict(rollout_metadata)
        self.buffers: dict[str, list[torch.Tensor]] = {}
        self.buffered = 0
        self.total = 0
        self.shard_index = 0

    def append(self, batch: dict[str, torch.Tensor]) -> None:
        batch_size = int(batch["scene_t"].shape[0])
        if batch_size == 0:
            return
        if any(value.shape[0] != batch_size for value in batch.values()):
            raise ValueError("all transition fields must have the same batch dimension")
        for key, value in batch.items():
            self.buffers.setdefault(key, []).append(value.detach().cpu())
        self.buffered += batch_size
        if self.buffered >= self.shard_transitions:
            self.flush()

    def flush(self) -> None:
        if self.buffered == 0:
            return
        transitions = {
            key: torch.cat(parts, dim=0) for key, parts in self.buffers.items()
        }
        payload = {
            "schema_version": 1,
            "control_dt_s": self.control_dt_s,
            "feature_order": ("x", "y", "z", "point_mass", "vx", "vy", "vz"),
            "component_ranges": {
                "target": (0, 512),
                "obstacles": (512, 1024),
                "end_effector": (1024, 1280),
            },
            "rollout": self.rollout_metadata,
            "transitions": transitions,
        }
        path = self.output_dir / f"transitions_{self.shard_index:05d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        self.total += self.buffered
        print(
            "DAPL_TRANSITION_SHARD",
            f"path={path}",
            f"transitions={self.buffered}",
            flush=True,
        )
        self.buffers.clear()
        self.buffered = 0
        self.shard_index += 1


def main() -> None:
    _validate_args()
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        policy_runner = None
        wrapped_env = None
        if args_cli.action_mode == "policy":
            agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
            agent_cfg.seed = args_cli.seed
            agent_cfg.device = args_cli.device
            wrapped_env = RslRlVecEnvWrapper(
                raw_env, clip_actions=agent_cfg.clip_actions
            )
            policy_runner = OnPolicyRunner(
                wrapped_env,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
            checkpoint = args_cli.checkpoint.expanduser().resolve()
            policy_runner.load(str(checkpoint), load_optimizer=False)
            policy_runner.eval_mode()
            policy_observation, observation_extras = wrapped_env.get_observations()
            observations = observation_extras["observations"]
            if args_cli.policy_action == "mean":
                policy = policy_runner.get_inference_policy(device=wrapped_env.device)
            else:
                policy = lambda value: policy_runner.alg.policy.act(  # noqa: E731
                    policy_runner.obs_normalizer(value)
                )
        else:
            observations, _ = raw_env.reset(seed=args_cli.seed)
            policy_observation = observations["policy"]
            policy = None

        base = raw_env.unwrapped
        if args_cli.initial_task_mode == "distributed":
            episode_offsets = torch.tensor(
                [
                    env_id
                    % len(
                        base._clutter_scenes_runtime[
                            env_id % len(base._clutter_scenes_runtime)
                        ].tasks
                    )
                    for env_id in range(base.num_envs)
                ],
                device=base.device,
                dtype=torch.long,
            )
            base._clutter_episode_counts.copy_(episode_offsets)
            if wrapped_env is None:
                observations, _ = raw_env.reset(seed=args_cli.seed)
                policy_observation = observations["policy"]
            else:
                policy_observation, observation_extras = wrapped_env.reset()
                observations = observation_extras["observations"]

        control_dt_s = float(base.step_dt)
        if abs(control_dt_s - 0.1) > 1.0e-9:
            raise RuntimeError(
                f"DAPL transitions require 0.1 s control steps, got {control_dt_s}"
            )
        writer = TransitionShardWriter(
            args_cli.output_dir,
            args_cli.shard_transitions,
            control_dt_s,
            rollout_metadata={
                "task": args_cli.task,
                "seed": args_cli.seed,
                "num_envs": base.num_envs,
                "requested_control_steps": args_cli.steps,
                "action_mode": args_cli.action_mode,
                "initial_task_mode": args_cli.initial_task_mode,
                "policy_action": (
                    args_cli.policy_action if args_cli.action_mode == "policy" else None
                ),
                "checkpoint": (
                    str(args_cli.checkpoint.expanduser().resolve())
                    if args_cli.checkpoint is not None
                    else None
                ),
                "hand_point_source": base._dapl_hand_point_source,
                "manifest": base.cfg.clutter_manifest_path,
                "manifest_sha256": _sha256(Path(base.cfg.clutter_manifest_path)),
                "scene_count": len(base._clutter_scenes_runtime),
                "asset_source": base.cfg.clutter_asset_source,
            },
        )
        scene_t = observations["world_model"]["scene"]
        if scene_t.shape != (args_cli.num_envs, 1280, 7):
            raise RuntimeError(f"unexpected physical scene shape: {tuple(scene_t.shape)}")

        action_state = torch.zeros(
            (base.num_envs, base.action_manager.total_action_dim), device=base.device
        )
        generator = torch.Generator(device=base.device)
        generator.manual_seed(args_cli.seed)
        target_cfg = SceneEntityCfg("target")
        obstacles_cfg = SceneEntityCfg("obstacles")
        ee_frame_cfg = SceneEntityCfg("ee_frame")

        with torch.inference_mode():
            for step_index in range(args_cli.steps):
                source_indices = base._dapl_obstacle_source_indices.clone()
                scene_indices = base._clutter_scene_indices.clone()
                task_indices = base._clutter_task_indices.clone()
                episode_indices = base._clutter_episode_counts.clone() - 1
                goals = base.command_manager.get_command("target_object_pose").clone()

                if args_cli.action_mode == "policy":
                    actions = policy(policy_observation)
                elif args_cli.action_mode == "zero":
                    actions = torch.zeros_like(action_state)
                else:
                    noise = torch.randn(
                        action_state.shape,
                        device=base.device,
                        generator=generator,
                    )
                    action_state.mul_(args_cli.random_action_decay).add_(
                        noise, alpha=args_cli.random_action_std
                    )
                    actions = torch.clamp(action_state, min=-1.0, max=1.0)

                if actions.shape != action_state.shape or not torch.isfinite(actions).all():
                    raise RuntimeError(
                        f"invalid action batch from {args_cli.action_mode}: "
                        f"shape={tuple(actions.shape)}"
                    )

                if wrapped_env is None:
                    next_observations, rewards, terminated, truncated, _ = raw_env.step(
                        actions
                    )
                    done = terminated | truncated
                    next_policy_observation = next_observations["policy"]
                else:
                    (
                        next_policy_observation,
                        rewards,
                        dones,
                        step_extras,
                    ) = wrapped_env.step(actions)
                    done = dones.bool()
                    next_observations = step_extras["observations"]
                valid = ~done

                # Rebuild t+1 with the exact obstacle canonical-point indices
                # selected at t.  This is the correspondence contract used by
                # the DAPL position and velocity prediction losses.
                aligned_future = build_dapl_physical_scene(
                    base,
                    target_cfg=target_cfg,
                    obstacles_cfg=obstacles_cfg,
                    ee_frame_cfg=ee_frame_cfg,
                    obstacle_source_indices=source_indices,
                ).features
                end_effector_flow = (
                    aligned_future[:, 1024:, :3].mean(dim=1)
                    - scene_t[:, 1024:, :3].mean(dim=1)
                )
                env_ids = torch.arange(base.num_envs, device=base.device)
                sim_steps = torch.full(
                    (base.num_envs,), step_index, device=base.device, dtype=torch.long
                )
                writer.append(
                    {
                        "scene_t": scene_t[valid],
                        "action": actions[valid],
                        "end_effector_flow": end_effector_flow[valid],
                        "scene_tp1": aligned_future[valid],
                        "obstacle_source_indices": source_indices[valid],
                        "goal_pose": goals[valid],
                        "reward": rewards[valid].unsqueeze(-1),
                        "scene_index": scene_indices[valid].unsqueeze(-1),
                        "task_index": task_indices[valid].unsqueeze(-1),
                        "episode_index": episode_indices[valid].unsqueeze(-1),
                        "env_id": env_ids[valid].unsqueeze(-1),
                        "sim_step": sim_steps[valid].unsqueeze(-1),
                    }
                )
                action_state[done] = 0.0
                scene_t = next_observations["world_model"]["scene"]
                policy_observation = next_policy_observation

        writer.flush()
        print(
            "DAPL_TRANSITION_COLLECTION_OK",
            f"transitions={writer.total}",
            f"shards={writer.shard_index}",
            f"control_dt_s={control_dt_s:.6f}",
            f"output_dir={writer.output_dir}",
            flush=True,
        )
    finally:
        raw_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
