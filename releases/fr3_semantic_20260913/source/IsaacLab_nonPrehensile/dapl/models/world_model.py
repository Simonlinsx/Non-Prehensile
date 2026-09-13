"""Paper-aligned DAPL dynamics encoder, action decoder, and losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tokenizer import (
    DAPLSemanticPatchTokenizer,
    DAPLSemanticPatchTokenizerConfig,
    SemanticPatchTokens,
)


@dataclass(frozen=True)
class DAPLWorldModelConfig:
    """Architecture values reported in the DAPL supplementary material."""

    tokenizer: DAPLSemanticPatchTokenizerConfig = DAPLSemanticPatchTokenizerConfig()
    encoder_depth: int = 12
    attention_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    action_dim: int = 3

    def __post_init__(self) -> None:
        if self.encoder_depth <= 0 or self.attention_heads <= 0:
            raise ValueError("encoder depth and attention heads must be positive")
        if self.tokenizer.token_dim % self.attention_heads != 0:
            raise ValueError("token_dim must be divisible by attention_heads")
        if self.mlp_ratio <= 0.0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid transformer MLP ratio or dropout")
        if self.action_dim != 3:
            raise ValueError("DAPL conditions the world model on 3-D end-effector flow")


@dataclass(frozen=True)
class DAPLWorldModelPrediction:
    """Future point state and intermediate dynamics representations."""

    position: torch.Tensor
    velocity: torch.Tensor
    dynamics_tokens: torch.Tensor
    point_features: torch.Tensor
    tokenization: SemanticPatchTokens


class DAPLDynamicsEncoder(nn.Module):
    """Semantic patch tokenizer followed by the 12-block, 8-head ViT."""

    def __init__(self, config: DAPLWorldModelConfig | None = None):
        super().__init__()
        self.config = DAPLWorldModelConfig() if config is None else config
        self.tokenizer = DAPLSemanticPatchTokenizer(self.config.tokenizer)
        token_dim = self.config.tokenizer.token_dim
        block = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=self.config.attention_heads,
            dim_feedforward=int(round(token_dim * self.config.mlp_ratio)),
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block,
            num_layers=self.config.encoder_depth,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )

    def forward(self, scene: torch.Tensor) -> tuple[torch.Tensor, SemanticPatchTokens]:
        tokenization = self.tokenizer(scene)
        return self.transformer(tokenization.tokens), tokenization


class DAPLActionConditionedDecoder(nn.Module):
    """Single cross-attention layer and scatter-based point decoder."""

    def __init__(self, config: DAPLWorldModelConfig | None = None):
        super().__init__()
        self.config = DAPLWorldModelConfig() if config is None else config
        token_dim = self.config.tokenizer.token_dim
        self.action_projection = nn.Sequential(
            nn.Linear(self.config.action_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            token_dim,
            self.config.attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, int(round(token_dim * self.config.mlp_ratio))),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(int(round(token_dim * self.config.mlp_ratio)), token_dim),
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.local_position_encoder = nn.Sequential(
            nn.Linear(3, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.position_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 3)
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, 3)
        )

    def forward(
        self,
        scene: torch.Tensor,
        dynamics_tokens: torch.Tensor,
        tokenization: SemanticPatchTokens,
        end_effector_flow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, point_count, _ = scene.shape
        expected_action_shape = (batch_size, self.config.action_dim)
        if end_effector_flow.shape != expected_action_shape:
            raise ValueError(
                f"end_effector_flow must have shape {expected_action_shape}, "
                f"got {tuple(end_effector_flow.shape)}"
            )
        action_token = self.action_projection(end_effector_flow).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=dynamics_tokens,
            key=action_token,
            value=action_token,
            need_weights=False,
        )
        conditioned_tokens = self.attention_norm(dynamics_tokens + attended)
        conditioned_tokens = self.output_norm(
            conditioned_tokens + self.feed_forward(conditioned_tokens)
        )
        point_features = self._unpatchify(
            scene,
            conditioned_tokens,
            tokenization.centers,
            tokenization.neighbor_indices,
            point_count,
        )
        return (
            self.position_head(point_features),
            self.velocity_head(point_features),
            point_features,
        )

    def _unpatchify(
        self,
        scene: torch.Tensor,
        patch_tokens: torch.Tensor,
        patch_centers: torch.Tensor,
        neighbor_indices: torch.Tensor,
        point_count: int,
    ) -> torch.Tensor:
        """Scatter patch memberships to points, averaging overlapping patches."""

        batch_size, patch_count, neighbors = neighbor_indices.shape
        token_dim = patch_tokens.shape[-1]
        batch_indices = torch.arange(scene.shape[0], device=scene.device)[:, None, None]
        member_xyz = scene[batch_indices, neighbor_indices, :3]
        local_offsets = member_xyz - patch_centers[..., None, :3]
        member_features = patch_tokens[..., None, :] + self.local_position_encoder(
            local_offsets
        )
        flat_indices = neighbor_indices.reshape(batch_size, patch_count * neighbors)
        flat_features = member_features.reshape(
            batch_size, patch_count * neighbors, token_dim
        )
        point_sums = patch_tokens.new_zeros((batch_size, point_count, token_dim))
        point_sums.scatter_add_(
            1, flat_indices.unsqueeze(-1).expand(-1, -1, token_dim), flat_features
        )
        counts = patch_tokens.new_zeros((batch_size, point_count, 1))
        counts.scatter_add_(
            1,
            flat_indices.unsqueeze(-1),
            torch.ones(
                (batch_size, patch_count * neighbors, 1),
                device=scene.device,
                dtype=scene.dtype,
            ),
        )
        averaged = point_sums / counts.clamp_min(1.0)

        # FPS+kNN patches may overlap, leaving a small number of points without
        # membership.  Assign those points to their nearest semantic patch so
        # the parallel heads still produce dense 1,280-point supervision.
        squared_distance = torch.sum(
            (
                scene[:, :, None, :3]
                - patch_centers[:, None, :, :3]
            ).square(),
            dim=-1,
        )
        nearest_patch = torch.argmin(squared_distance, dim=-1)
        point_batch = torch.arange(batch_size, device=scene.device)[:, None]
        nearest_tokens = patch_tokens[point_batch, nearest_patch]
        nearest_centers = patch_centers[point_batch, nearest_patch, :3]
        fallback = nearest_tokens + self.local_position_encoder(
            scene[..., :3] - nearest_centers
        )
        return torch.where(counts > 0.0, averaged, fallback)


class DAPLWorldModel(nn.Module):
    """End-to-end point dynamics predictor for one 0.1-second transition."""

    def __init__(self, config: DAPLWorldModelConfig | None = None):
        super().__init__()
        self.config = DAPLWorldModelConfig() if config is None else config
        self.dynamics_encoder = DAPLDynamicsEncoder(self.config)
        self.decoder = DAPLActionConditionedDecoder(self.config)

    def forward(
        self, scene: torch.Tensor, end_effector_flow: torch.Tensor
    ) -> DAPLWorldModelPrediction:
        dynamics_tokens, tokenization = self.dynamics_encoder(scene)
        position, velocity, point_features = self.decoder(
            scene, dynamics_tokens, tokenization, end_effector_flow
        )
        return DAPLWorldModelPrediction(
            position=position,
            velocity=velocity,
            dynamics_tokens=dynamics_tokens,
            point_features=point_features,
            tokenization=tokenization,
        )


@dataclass(frozen=True)
class DAPLWorldModelLossConfig:
    position_weight: float = 1.0
    velocity_weight: float = 1.0
    variance_weight: float = 100.0

    def __post_init__(self) -> None:
        if min(self.position_weight, self.velocity_weight, self.variance_weight) < 0.0:
            raise ValueError("world-model loss weights must be non-negative")


@dataclass(frozen=True)
class DAPLWorldModelLossOutput:
    total: torch.Tensor
    position: torch.Tensor
    velocity: torch.Tensor
    variance: torch.Tensor


class DAPLWorldModelLoss(nn.Module):
    """Position, velocity, and global velocity-variance objective."""

    def __init__(self, config: DAPLWorldModelLossConfig | None = None):
        super().__init__()
        self.config = DAPLWorldModelLossConfig() if config is None else config

    def forward(
        self, prediction: DAPLWorldModelPrediction, future_scene: torch.Tensor
    ) -> DAPLWorldModelLossOutput:
        expected = prediction.position.shape[:-1] + (7,)
        if future_scene.shape != expected:
            raise ValueError(
                f"future_scene must have shape {expected}, got {tuple(future_scene.shape)}"
            )
        position_loss = torch.mean(
            (prediction.position - future_scene[..., :3]).square()
        )
        velocity_loss = torch.mean(
            (prediction.velocity - future_scene[..., 4:7]).square()
        )
        predicted_variance = prediction.velocity.reshape(-1, 3).var(
            dim=0, unbiased=False
        )
        target_variance = future_scene[..., 4:7].reshape(-1, 3).var(
            dim=0, unbiased=False
        )
        variance_loss = torch.mean((predicted_variance - target_variance).square())
        total = (
            self.config.position_weight * position_loss
            + self.config.velocity_weight * velocity_loss
            + self.config.variance_weight * variance_loss
        )
        return DAPLWorldModelLossOutput(
            total=total,
            position=position_loss,
            velocity=velocity_loss,
            variance=variance_loss,
        )
