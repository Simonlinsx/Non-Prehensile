# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DAPL dynamics-conditioned actor-critic for the in-tree RSL-RL fork."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from dapl.models import (
    DAPLFeatureNormalizer,
    DAPLDynamicsEncoder,
    DAPLSemanticPatchTokenizerConfig,
    DAPLWorldModelConfig,
)
from dapl.representation import DAPLSceneTensorConfig
from rsl_rl.utils import resolve_nn_activation


def _world_model_config(payload: dict) -> DAPLWorldModelConfig:
    values = dict(payload)
    tokenizer_values = dict(values.pop("tokenizer"))
    scene = DAPLSceneTensorConfig(**tokenizer_values.pop("scene"))
    tokenizer = DAPLSemanticPatchTokenizerConfig(scene=scene, **tokenizer_values)
    config = DAPLWorldModelConfig(tokenizer=tokenizer, **values)
    if asdict(config) != payload:
        raise ValueError("world-model configuration did not round-trip")
    return config


def _mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation_name: str,
) -> nn.Sequential:
    if not hidden_dims:
        raise ValueError("MLP hidden dimensions must not be empty")
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        if hidden <= 0:
            raise ValueError("MLP hidden dimensions must be positive")
        layers.extend((nn.Linear(previous, hidden), resolve_nn_activation(activation_name)))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ActorCriticDAPL(nn.Module):
    """Frozen DAPL encoder with environment-state cross-attention and PPO heads.

    Flattened observations must be laid out as the paper's physical scene first
    (1,280 points with 7 features), followed by its 44-D environment state.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        world_model_checkpoint_path: str | None = None,
        scene_num_points: int = 1280,
        scene_point_dim: int = 7,
        environment_state_dim: int = 44,
        policy_attention_heads: int = 8,
        fusion_hidden_dims: list[int] | None = None,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "ActorCriticDAPL.__init__ got unexpected arguments, which will be ignored: "
                + str(sorted(kwargs))
            )
        super().__init__()
        if world_model_checkpoint_path is None:
            raise ValueError(
                "DAPL policy requires world_model_checkpoint_path; set "
                "DAPL_WORLD_MODEL_CHECKPOINT before launching training"
            )
        if min(scene_num_points, scene_point_dim, environment_state_dim) <= 0:
            raise ValueError("DAPL observation dimensions must be positive")
        if num_actions != 7:
            raise ValueError(f"DAPL paper policy requires 7 actions, got {num_actions}")

        self.scene_num_points = scene_num_points
        self.scene_point_dim = scene_point_dim
        self.scene_flat_dim = scene_num_points * scene_point_dim
        self.environment_state_dim = environment_state_dim
        expected_obs = self.scene_flat_dim + environment_state_dim
        if num_actor_obs != expected_obs or num_critic_obs != expected_obs:
            raise ValueError(
                "DAPL actor and critic observations must both have dimension "
                f"{expected_obs}, got {num_actor_obs} and {num_critic_obs}"
            )

        checkpoint_path = Path(world_model_checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DAPL world-model checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != 1:
            raise ValueError(f"unsupported DAPL world-model checkpoint: {checkpoint_path}")
        world_config = _world_model_config(checkpoint["model_config"])
        scene_config = world_config.tokenizer.scene
        if (
            scene_config.total_points != scene_num_points
            or scene_config.feature_dim != scene_point_dim
        ):
            raise ValueError("policy scene dimensions do not match the world-model checkpoint")

        self.dynamics_encoder = DAPLDynamicsEncoder(world_config)
        prefix = "dynamics_encoder."
        encoder_state = {
            key[len(prefix) :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith(prefix)
        }
        self.dynamics_encoder.load_state_dict(encoder_state, strict=True)
        for parameter in self.dynamics_encoder.parameters():
            parameter.requires_grad_(False)
        self.dynamics_encoder.eval()

        self.feature_normalizer = DAPLFeatureNormalizer()
        self.feature_normalizer.load_state_dict(checkpoint["normalizer"], strict=True)
        self.feature_normalizer.eval()
        self.register_buffer(
            "pretrained_world_model_step",
            torch.tensor(int(checkpoint["step"]), dtype=torch.int64),
        )
        self.world_model_checkpoint_path = str(checkpoint_path)

        token_dim = world_config.tokenizer.token_dim
        if policy_attention_heads <= 0 or token_dim % policy_attention_heads != 0:
            raise ValueError("policy attention heads must divide the dynamics token dimension")
        self.environment_query = nn.Linear(environment_state_dim, token_dim)
        self.cross_attention = nn.MultiheadAttention(
            token_dim, policy_attention_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(token_dim)

        fusion_hidden_dims = [512, 256, 128] if fusion_hidden_dims is None else fusion_hidden_dims
        actor_hidden_dims = [64] if actor_hidden_dims is None else actor_hidden_dims
        critic_hidden_dims = [64] if critic_hidden_dims is None else critic_hidden_dims
        self.feature_fusion = _mlp(
            token_dim + environment_state_dim,
            fusion_hidden_dims[:-1],
            fusion_hidden_dims[-1],
            activation,
        )
        self.actor = _mlp(
            fusion_hidden_dims[-1], actor_hidden_dims, num_actions, activation
        )
        self.critic = _mlp(
            fusion_hidden_dims[-1], critic_hidden_dims, 1, activation
        )

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError("noise_std_type must be 'scalar' or 'log'")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)
        print(
            "Loaded frozen DAPL dynamics encoder",
            f"step={int(self.pretrained_world_model_step.item())}",
            f"checkpoint={checkpoint_path}",
        )

    def _split_observations(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.scene_flat_dim + self.environment_state_dim
        if observations.ndim != 2 or observations.shape[1] != expected:
            raise ValueError(
                f"DAPL observations must have shape [batch, {expected}], "
                f"got {tuple(observations.shape)}"
            )
        scene = observations[:, : self.scene_flat_dim].reshape(
            -1, self.scene_num_points, self.scene_point_dim
        )
        environment_state = observations[:, self.scene_flat_dim :]
        return scene, environment_state

    def _fused_features(self, observations: torch.Tensor) -> torch.Tensor:
        scene, environment_state = self._split_observations(observations)
        with torch.no_grad():
            normalized_scene = self.feature_normalizer.normalize(scene)
            dynamics_tokens, _ = self.dynamics_encoder(normalized_scene)
        query = self.environment_query(environment_state).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=query,
            key=dynamics_tokens,
            value=dynamics_tokens,
            need_weights=False,
        )
        task_feature = self.attention_norm(query + attended).squeeze(1)
        return self.feature_fusion(torch.cat((task_feature, environment_state), dim=-1))

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
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self._fused_features(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(self._fused_features(critic_observations))

    def train(self, mode: bool = True):
        super().train(mode)
        self.dynamics_encoder.eval()
        self.feature_normalizer.eval()
        return self

    def load_state_dict(self, state_dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
