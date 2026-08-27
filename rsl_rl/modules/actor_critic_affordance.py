# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Part-aware PointNet/attention actor-critic for DOMINO manipulation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), resolve_nn_activation(activation)))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ActorCriticAffordance(nn.Module):
    """Jointly encode point geometry and affordance before spatial pooling.

    Observation layout is deliberately strict and shared by every curriculum
    stage::

        target:   512 x [x, y, z, safe, protected]
        obstacles: 512 x [x, y, z]
        state:     a task-configured vector (45 values for the deployable
                   teacher contract, including noisy target twist)

    Separate PointNets retain target semantics and clutter geometry.  The
    current robot/goal state supplies the query for attention over both sets;
    max pooling supplies a state-independent global summary.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        target_num_points: int = 512,
        target_point_dim: int = 5,
        obstacle_num_points: int = 512,
        obstacle_point_dim: int = 3,
        environment_state_dim: int = 50,
        critic_environment_state_dim: int | None = None,
        point_feature_dim: int = 64,
        attention_heads: int = 4,
        attention_queries: int = 1,
        fusion_hidden_dims: list[int] | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        max_noise_std: float | None = None,
        use_relation_features: bool = False,
        use_wrench_relation_features: bool = False,
        separate_wrench_relation_features: bool = False,
        use_protected_obstacle_relation_features: bool = False,
        freeze_base_actor_for_protected_obstacle_transfer: bool = False,
        zero_initialize_relation_output: bool = True,
        zero_initialize_protected_obstacle_relation_output: bool = True,
        wrench_relation_yaw_moment_weight: float = 0.5,
        wrench_relation_yaw_activation_rad: float = 0.10,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticAffordance.__init__ got unexpected arguments, which will be ignored: "
                + str(sorted(kwargs))
            )
        super().__init__()
        if point_feature_dim % attention_heads != 0:
            raise ValueError("point_feature_dim must be divisible by attention_heads")
        if int(attention_queries) <= 0:
            raise ValueError("attention_queries must be positive")

        self.target_num_points = int(target_num_points)
        self.target_point_dim = int(target_point_dim)
        self.obstacle_num_points = int(obstacle_num_points)
        self.obstacle_point_dim = int(obstacle_point_dim)
        self.environment_state_dim = int(environment_state_dim)
        self.attention_queries = int(attention_queries)
        self.use_relation_features = bool(use_relation_features)
        self.use_wrench_relation_features = bool(use_wrench_relation_features)
        self.separate_wrench_relation_features = bool(
            separate_wrench_relation_features
        )
        self.use_protected_obstacle_relation_features = bool(
            use_protected_obstacle_relation_features
        )
        self.freeze_base_actor_for_protected_obstacle_transfer = bool(
            freeze_base_actor_for_protected_obstacle_transfer
        )
        self.zero_initialize_relation_output = bool(
            zero_initialize_relation_output
        )
        self.zero_initialize_protected_obstacle_relation_output = bool(
            zero_initialize_protected_obstacle_relation_output
        )
        self.wrench_relation_yaw_moment_weight = float(
            wrench_relation_yaw_moment_weight
        )
        self.wrench_relation_yaw_activation_rad = float(
            wrench_relation_yaw_activation_rad
        )
        if self.use_wrench_relation_features and not self.use_relation_features:
            raise ValueError("wrench relation features require relation features")
        if (
            self.separate_wrench_relation_features
            and not self.use_wrench_relation_features
        ):
            raise ValueError(
                "separate wrench relation features require wrench relations"
            )
        if self.wrench_relation_yaw_moment_weight < 0.0:
            raise ValueError("wrench relation yaw moment weight must be non-negative")
        if self.wrench_relation_yaw_activation_rad <= 0.0:
            raise ValueError("wrench relation yaw activation must be positive")
        self.critic_environment_state_dim = (
            self.environment_state_dim
            if critic_environment_state_dim is None
            else int(critic_environment_state_dim)
        )
        if self.environment_state_dim <= 0 or self.critic_environment_state_dim <= 0:
            raise ValueError("actor and critic environment state dimensions must be positive")
        self.target_flat_dim = self.target_num_points * self.target_point_dim
        self.obstacle_flat_dim = self.obstacle_num_points * self.obstacle_point_dim
        expected_actor = (
            self.target_flat_dim + self.obstacle_flat_dim + self.environment_state_dim
        )
        expected_critic = (
            self.target_flat_dim
            + self.obstacle_flat_dim
            + self.critic_environment_state_dim
        )
        if num_actor_obs != expected_actor or num_critic_obs != expected_critic:
            raise ValueError(
                "affordance observation dimensions do not match the configured "
                f"contracts: actor expected {expected_actor}, got {num_actor_obs}; "
                f"critic expected {expected_critic}, got {num_critic_obs}"
            )

        self.target_pointnet = nn.Sequential(
            nn.Linear(self.target_point_dim, 32),
            resolve_nn_activation(activation),
            nn.Linear(32, point_feature_dim),
            resolve_nn_activation(activation),
        )
        self.obstacle_pointnet = nn.Sequential(
            nn.Linear(self.obstacle_point_dim, 32),
            resolve_nn_activation(activation),
            nn.Linear(32, point_feature_dim),
            resolve_nn_activation(activation),
        )
        if self.use_relation_features:
            # These channels are deterministic functions of the existing
            # recoverable observation: point-to-hand, object-local position,
            # and a goal-conditioned contact support score.  The optional
            # wrench-aware form adds signed yaw moment compatibility to the
            # old trailing-side projection.  Both are deterministic functions
            # of the existing recoverable observation and do not change the
            # external [xyz,safe,protected] point contract.
            target_relation_dim = (
                8 if self.separate_wrench_relation_features else 7
            )
            self.target_relation_pointnet = _mlp(
                target_relation_dim, [32], point_feature_dim, activation
            )
            self.obstacle_relation_pointnet = _mlp(
                6, [32], point_feature_dim, activation
            )
            if self.zero_initialize_relation_output:
                self._zero_relation_output(self.target_relation_pointnet)
                self._zero_relation_output(self.obstacle_relation_pointnet)
        else:
            self.target_relation_pointnet = None
            self.obstacle_relation_pointnet = None
        if self.use_protected_obstacle_relation_features:
            # This residual exposes the relation that defines C2 directly:
            # blocker position relative to the target's protected geometry.
            # It is derived only from the deployable [xyz,safe,protected] and
            # obstacle XYZ observations; no simulator-only label is added.
            self.protected_obstacle_relation_pointnet = _mlp(
                4, [32], point_feature_dim, activation
            )
            if self.zero_initialize_protected_obstacle_relation_output:
                self._zero_relation_output(
                    self.protected_obstacle_relation_pointnet
                )
        else:
            self.protected_obstacle_relation_pointnet = None
        self.state_encoder = _mlp(
            self.environment_state_dim,
            [128],
            self.attention_queries * point_feature_dim,
            activation,
        )
        self.target_attention = nn.MultiheadAttention(
            point_feature_dim, attention_heads, batch_first=True
        )
        self.obstacle_attention = nn.MultiheadAttention(
            point_feature_dim, attention_heads, batch_first=True
        )
        self.target_norm = nn.LayerNorm(point_feature_dim)
        self.obstacle_norm = nn.LayerNorm(point_feature_dim)

        fusion_hidden_dims = fusion_hidden_dims or [256, 128]
        actor_hidden_dims = actor_hidden_dims or [128, 64]
        critic_hidden_dims = critic_hidden_dims or [128, 64]
        # One state summary, one global summary for each point set, and all
        # task-conditioned query outputs from both target and obstacle tokens.
        # attention_queries=1 is exactly the legacy five-token contract.
        fusion_input_dim = (3 + 2 * self.attention_queries) * point_feature_dim
        self.feature_fusion = _mlp(
            fusion_input_dim,
            fusion_hidden_dims[:-1],
            fusion_hidden_dims[-1],
            activation,
        )
        self._has_asymmetric_critic = (
            self.critic_environment_state_dim != self.environment_state_dim
        )
        if self._has_asymmetric_critic:
            # Keep every critic encoder independent.  Privileged physics may
            # improve the value baseline, but must not update a shared feature
            # pathway that the actor later uses without those inputs.
            self.critic_target_pointnet = nn.Sequential(
                nn.Linear(self.target_point_dim, 32),
                resolve_nn_activation(activation),
                nn.Linear(32, point_feature_dim),
                resolve_nn_activation(activation),
            )
            self.critic_obstacle_pointnet = nn.Sequential(
                nn.Linear(self.obstacle_point_dim, 32),
                resolve_nn_activation(activation),
                nn.Linear(32, point_feature_dim),
                resolve_nn_activation(activation),
            )
            if self.use_relation_features:
                self.critic_target_relation_pointnet = _mlp(
                    target_relation_dim, [32], point_feature_dim, activation
                )
                self.critic_obstacle_relation_pointnet = _mlp(
                    6, [32], point_feature_dim, activation
                )
                if self.zero_initialize_relation_output:
                    self._zero_relation_output(
                        self.critic_target_relation_pointnet
                    )
                    self._zero_relation_output(
                        self.critic_obstacle_relation_pointnet
                    )
            else:
                self.critic_target_relation_pointnet = None
                self.critic_obstacle_relation_pointnet = None
            if self.use_protected_obstacle_relation_features:
                self.critic_protected_obstacle_relation_pointnet = _mlp(
                    4, [32], point_feature_dim, activation
                )
                if self.zero_initialize_protected_obstacle_relation_output:
                    self._zero_relation_output(
                        self.critic_protected_obstacle_relation_pointnet
                    )
            else:
                self.critic_protected_obstacle_relation_pointnet = None
            self.critic_state_encoder = _mlp(
                self.critic_environment_state_dim,
                [128],
                self.attention_queries * point_feature_dim,
                activation,
            )
            self.critic_target_attention = nn.MultiheadAttention(
                point_feature_dim, attention_heads, batch_first=True
            )
            self.critic_obstacle_attention = nn.MultiheadAttention(
                point_feature_dim, attention_heads, batch_first=True
            )
            self.critic_target_norm = nn.LayerNorm(point_feature_dim)
            self.critic_obstacle_norm = nn.LayerNorm(point_feature_dim)
            self.critic_feature_fusion = _mlp(
                fusion_input_dim,
                fusion_hidden_dims[:-1],
                fusion_hidden_dims[-1],
                activation,
            )
        else:
            self.critic_target_pointnet = None
            self.critic_obstacle_pointnet = None
            self.critic_target_relation_pointnet = None
            self.critic_obstacle_relation_pointnet = None
            self.critic_protected_obstacle_relation_pointnet = None
            self.critic_state_encoder = None
            self.critic_target_attention = None
            self.critic_obstacle_attention = None
            self.critic_target_norm = None
            self.critic_obstacle_norm = None
            self.critic_feature_fusion = None
        self.actor = _mlp(
            fusion_hidden_dims[-1], actor_hidden_dims, num_actions, activation
        )
        self.critic = _mlp(
            fusion_hidden_dims[-1], critic_hidden_dims, 1, activation
        )

        self.noise_std_type = noise_std_type
        if max_noise_std is not None and float(max_noise_std) <= 0.0:
            raise ValueError("max_noise_std must be positive when provided")
        self.max_noise_std = (
            None if max_noise_std is None else float(max_noise_std)
        )
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)
        if self.freeze_base_actor_for_protected_obstacle_transfer:
            if not self.use_protected_obstacle_relation_features:
                raise ValueError(
                    "protected-obstacle adapter transfer requires its relation features"
                )
            self._freeze_base_actor_for_protected_obstacle_transfer()

    @staticmethod
    def _zero_relation_output(module: nn.Sequential) -> None:
        """Start a relation branch as an exact behavior-preserving residual."""

        final_linear = next(
            layer for layer in reversed(module) if isinstance(layer, nn.Linear)
        )
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def _freeze_base_actor_for_protected_obstacle_transfer(self) -> None:
        """Train only the new actor residual while retaining a trainable critic.

        The asymmetric critic has its own encoders, so all ``critic*``
        parameters can adapt to C2 returns without changing the frozen source
        policy.  The action noise is frozen as part of that source contract.
        """

        for name, parameter in self.named_parameters():
            trainable = name.startswith(
                "protected_obstacle_relation_pointnet."
            ) or name.startswith("critic")
            parameter.requires_grad_(trainable)

    @staticmethod
    def _relation_inputs(
        target: torch.Tensor,
        obstacles: torch.Tensor,
        state: torch.Tensor,
        *,
        use_wrench_relation_features: bool = False,
        separate_wrench_relation_features: bool = False,
        yaw_moment_weight: float = 0.5,
        yaw_activation_rad: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct recoverable hand/object/goal relations for every point."""

        # State layout starts with normalized hand position.  These constants
        # exactly invert mdp.observations.hand_state.
        hand_mean = state.new_tensor((0.5, 0.0, 0.15))
        hand_std = state.new_tensor((0.4, 0.4, 0.4))
        hand_position = state[:, :3] * hand_std + hand_mean
        target_xyz = target[..., :3]
        target_center = target_xyz.mean(dim=1)

        # rel_goal starts after hand(9), robot(14), and previous action(7).
        goal_scale = state.new_tensor((0.10, 0.10, 0.02))
        goal_displacement = state[:, 30:33] * goal_scale
        goal_direction_xy = goal_displacement[:, :2] / torch.clamp(
            torch.linalg.vector_norm(goal_displacement[:, :2], dim=1, keepdim=True),
            min=1.0e-6,
        )
        target_local = target_xyz - target_center[:, None, :]
        trailing_projection = torch.sum(
            target_local[..., :2] * -goal_direction_xy[:, None, :], dim=-1
        )
        contact_support_score = trailing_projection
        signed_yaw_moment = None
        if use_wrench_relation_features:
            if yaw_moment_weight < 0.0 or yaw_activation_rad <= 0.0:
                raise ValueError(
                    "yaw moment weight must be non-negative and activation positive"
                )
            # rel_goal rotation is stored as the first two rows of its
            # relative rotation matrix.  atan2(R10, R00) is the signed yaw
            # error for this support-preserving planar task.
            yaw_error = torch.atan2(state[:, 36], state[:, 33])
            signed_moment_arm = (
                target_local[..., 0] * goal_direction_xy[:, None, 1]
                - target_local[..., 1] * goal_direction_xy[:, None, 0]
            )
            yaw_gate = torch.tanh(yaw_error / float(yaw_activation_rad))
            signed_yaw_moment = (
                float(yaw_moment_weight)
                * yaw_gate[:, None]
                * signed_moment_arm
            )
            contact_support_score = trailing_projection + signed_yaw_moment
        if separate_wrench_relation_features:
            if signed_yaw_moment is None:
                raise ValueError(
                    "separate wrench channels require wrench relations"
                )
            task_relation = torch.stack(
                (trailing_projection, signed_yaw_moment), dim=-1
            ) / 0.2
        else:
            task_relation = contact_support_score[..., None] / 0.2
        target_relation = torch.cat(
            (
                (target_xyz - hand_position[:, None, :]) / 0.4,
                target_local / 0.2,
                task_relation,
            ),
            dim=-1,
        )
        obstacle_relation = torch.cat(
            (
                (obstacles - hand_position[:, None, :]) / 0.4,
                (obstacles - target_center[:, None, :]) / 0.4,
            ),
            dim=-1,
        )
        # The no-clutter curriculum uses an all-zero 512x3 token.  Do not turn
        # it into a fictitious obstacle merely by subtracting hand/target pose.
        obstacle_valid = torch.any(obstacles != 0.0, dim=(1, 2))
        return target_relation, obstacle_relation, obstacle_valid

    @staticmethod
    def _protected_obstacle_relation_inputs(
        target: torch.Tensor,
        obstacles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode blocker points relative to recoverable protected geometry.

        The protected channel is a per-point soft mask, so a weighted centroid
        works for both binary oracle labels and future predicted probabilities.
        A compact single-hammer protected endpoint is the current controlled
        C2 setting.  Invalid all-zero obstacle tokens remain exactly masked.
        """

        target_xyz = target[..., :3]
        protected_weights = torch.clamp(target[..., 4], min=0.0, max=1.0)
        protected_mass = protected_weights.sum(dim=1, keepdim=True)
        protected_centroid = torch.sum(
            target_xyz * protected_weights[..., None], dim=1
        ) / torch.clamp(protected_mass, min=1.0e-6)
        relative_xyz = obstacles - protected_centroid[:, None, :]
        scale_m = 0.20
        relation = torch.cat(
            (
                relative_xyz / scale_m,
                torch.linalg.vector_norm(
                    relative_xyz, dim=-1, keepdim=True
                ) / scale_m,
            ),
            dim=-1,
        )
        protected_valid = protected_mass[:, 0] > 1.0e-6
        obstacle_valid = torch.any(obstacles != 0.0, dim=(1, 2))
        return relation, protected_valid & obstacle_valid

    def _split_observations(
        self,
        observations: torch.Tensor,
        environment_state_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_dim = (
            self.environment_state_dim
            if environment_state_dim is None
            else int(environment_state_dim)
        )
        expected = self.target_flat_dim + self.obstacle_flat_dim + state_dim
        if observations.ndim != 2 or observations.shape[1] != expected:
            raise ValueError(
                f"affordance observations must have shape [batch, {expected}], "
                f"got {tuple(observations.shape)}"
            )
        target_end = self.target_flat_dim
        obstacle_end = target_end + self.obstacle_flat_dim
        target = observations[:, :target_end].reshape(
            -1, self.target_num_points, self.target_point_dim
        )
        obstacles = observations[:, target_end:obstacle_end].reshape(
            -1, self.obstacle_num_points, self.obstacle_point_dim
        )
        return target, obstacles, observations[:, obstacle_end:]

    def _encode_fused_features(
        self,
        observations: torch.Tensor,
        *,
        environment_state_dim: int,
        target_pointnet: nn.Module,
        obstacle_pointnet: nn.Module,
        state_encoder: nn.Module,
        target_attention: nn.MultiheadAttention,
        obstacle_attention: nn.MultiheadAttention,
        target_norm: nn.LayerNorm,
        obstacle_norm: nn.LayerNorm,
        feature_fusion: nn.Module,
        target_relation_pointnet: nn.Module | None = None,
        obstacle_relation_pointnet: nn.Module | None = None,
        protected_obstacle_relation_pointnet: nn.Module | None = None,
    ) -> torch.Tensor:
        target, obstacles, state = self._split_observations(
            observations, environment_state_dim
        )
        target_tokens = target_pointnet(target)
        obstacle_tokens = obstacle_pointnet(obstacles)
        if target_relation_pointnet is not None:
            target_relation, obstacle_relation, obstacle_valid = self._relation_inputs(
                target,
                obstacles,
                state,
                use_wrench_relation_features=self.use_wrench_relation_features,
                separate_wrench_relation_features=(
                    self.separate_wrench_relation_features
                ),
                yaw_moment_weight=self.wrench_relation_yaw_moment_weight,
                yaw_activation_rad=self.wrench_relation_yaw_activation_rad,
            )
            target_tokens = target_tokens + target_relation_pointnet(target_relation)
            obstacle_tokens = obstacle_tokens + obstacle_relation_pointnet(
                obstacle_relation
            ) * obstacle_valid[:, None, None]
        if protected_obstacle_relation_pointnet is not None:
            protected_obstacle_relation, relation_valid = (
                self._protected_obstacle_relation_inputs(target, obstacles)
            )
            obstacle_tokens = obstacle_tokens + (
                protected_obstacle_relation_pointnet(
                    protected_obstacle_relation
                )
                * relation_valid[:, None, None]
            )
        state_tokens = state_encoder(state).reshape(
            -1, self.attention_queries, target_tokens.shape[-1]
        )
        target_attended, _ = target_attention(
            state_tokens, target_tokens, target_tokens, need_weights=False
        )
        obstacle_attended, _ = obstacle_attention(
            state_tokens, obstacle_tokens, obstacle_tokens, need_weights=False
        )
        target_task = target_norm(state_tokens + target_attended).flatten(1)
        obstacle_task = obstacle_norm(state_tokens + obstacle_attended).flatten(1)
        state_summary = state_tokens.mean(dim=1)
        target_global = torch.amax(target_tokens, dim=1)
        obstacle_global = torch.amax(obstacle_tokens, dim=1)
        return feature_fusion(
            torch.cat(
                (
                    state_summary,
                    target_task,
                    target_global,
                    obstacle_task,
                    obstacle_global,
                ),
                dim=-1,
            )
        )

    def _fused_features(self, observations: torch.Tensor) -> torch.Tensor:
        return self._encode_fused_features(
            observations,
            environment_state_dim=self.environment_state_dim,
            target_pointnet=self.target_pointnet,
            obstacle_pointnet=self.obstacle_pointnet,
            state_encoder=self.state_encoder,
            target_attention=self.target_attention,
            obstacle_attention=self.obstacle_attention,
            target_norm=self.target_norm,
            obstacle_norm=self.obstacle_norm,
            feature_fusion=self.feature_fusion,
            target_relation_pointnet=self.target_relation_pointnet,
            obstacle_relation_pointnet=self.obstacle_relation_pointnet,
            protected_obstacle_relation_pointnet=(
                self.protected_obstacle_relation_pointnet
            ),
        )

    def _critic_fused_features(self, observations: torch.Tensor) -> torch.Tensor:
        if not self._has_asymmetric_critic:
            return self._fused_features(observations)
        return self._encode_fused_features(
            observations,
            environment_state_dim=self.critic_environment_state_dim,
            target_pointnet=self.critic_target_pointnet,
            obstacle_pointnet=self.critic_obstacle_pointnet,
            state_encoder=self.critic_state_encoder,
            target_attention=self.critic_target_attention,
            obstacle_attention=self.critic_obstacle_attention,
            target_norm=self.critic_target_norm,
            obstacle_norm=self.critic_obstacle_norm,
            feature_fusion=self.critic_feature_fusion,
            target_relation_pointnet=self.critic_target_relation_pointnet,
            obstacle_relation_pointnet=self.critic_obstacle_relation_pointnet,
            protected_obstacle_relation_pointnet=(
                self.critic_protected_obstacle_relation_pointnet
            ),
        )

    def reset(self, dones=None) -> None:
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(self._fused_features(observations))
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        if self.max_noise_std is not None:
            # A resumed checkpoint also restores its learned exploration
            # parameter.  Hard-constraint fine-tuning needs an independent
            # upper bound so old stochasticity cannot cause immediate unsafe
            # terminations before PPO sees useful trajectories.
            std = torch.clamp(std, max=self.max_noise_std)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self._fused_features(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(self._critic_fused_features(critic_observations))

    def load_state_dict(self, state_dict, strict: bool = True) -> bool:
        missing_legacy_critic = self._has_asymmetric_critic and not any(
            key.startswith("critic_target_pointnet.") for key in state_dict
        )
        missing_relation = self.use_relation_features and not any(
            key.startswith("target_relation_pointnet.") for key in state_dict
        )
        missing_protected_obstacle_relation = (
            self.use_protected_obstacle_relation_features
            and not any(
                key.startswith("protected_obstacle_relation_pointnet.")
                for key in state_dict
            )
        )
        if (
            missing_legacy_critic
            or missing_relation
            or missing_protected_obstacle_relation
        ):
            if missing_relation and not self.zero_initialize_relation_output:
                raise RuntimeError(
                    "cannot load a checkpoint without relation branches into "
                    "a non-zero-initialized relation policy"
                )
            if (
                missing_protected_obstacle_relation
                and not self.zero_initialize_protected_obstacle_relation_output
            ):
                raise RuntimeError(
                    "cannot load a checkpoint without protected-obstacle "
                    "relations into a non-zero-initialized relation policy"
                )
            # Evaluation and controlled curriculum transfer may load a legacy
            # actor checkpoint produced before the independent critic/relation
            # branches existed.  The relation residual starts exactly at zero,
            # so loading it missing is behavior preserving; every pre-existing
            # action-path key remains strict.
            incompatible = super().load_state_dict(state_dict, strict=False)
            allowed_missing_prefixes = (
                "critic_target_pointnet.",
                "critic_obstacle_pointnet.",
                "critic_state_encoder.",
                "critic_target_attention.",
                "critic_obstacle_attention.",
                "critic_target_norm.",
                "critic_obstacle_norm.",
                "critic_feature_fusion.",
                "target_relation_pointnet.",
                "obstacle_relation_pointnet.",
                "critic_target_relation_pointnet.",
                "critic_obstacle_relation_pointnet.",
                "protected_obstacle_relation_pointnet.",
                "critic_protected_obstacle_relation_pointnet.",
            )
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_missing_prefixes)
            ]
            if strict and (invalid_missing or incompatible.unexpected_keys):
                raise RuntimeError(
                    "legacy affordance checkpoint has incompatible actor keys: "
                    f"missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        else:
            super().load_state_dict(state_dict, strict=strict)
        return True
