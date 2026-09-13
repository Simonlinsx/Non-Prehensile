"""Semantic FPS/kNN patch tokenization for DAPL physical scenes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dapl.representation import DAPLSceneTensorConfig, SceneComponent


@dataclass(frozen=True)
class DAPLSemanticPatchTokenizerConfig:
    """Paper-shaped semantic point grouping and token dimensions."""

    input_dim: int = 7
    token_dim: int = 128
    target_patches: int = 16
    obstacle_patches: int = 16
    end_effector_patches: int = 8
    neighbors: int = 32
    scene: DAPLSceneTensorConfig = DAPLSceneTensorConfig()

    def __post_init__(self) -> None:
        if self.input_dim != self.scene.feature_dim:
            raise ValueError("tokenizer input_dim must match the DAPL scene feature_dim")
        for name in (
            "token_dim",
            "target_patches",
            "obstacle_patches",
            "end_effector_patches",
            "neighbors",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.neighbors > min(
            self.scene.target_points,
            self.scene.obstacle_points,
            self.scene.end_effector_points,
        ):
            raise ValueError("neighbors exceeds a semantic component point count")
        if self.target_patches > self.scene.target_points:
            raise ValueError("target_patches exceeds target point count")
        if self.obstacle_patches > self.scene.obstacle_points:
            raise ValueError("obstacle_patches exceeds obstacle point count")
        if self.end_effector_patches > self.scene.end_effector_points:
            raise ValueError("end_effector_patches exceeds end-effector point count")

    @property
    def total_patches(self) -> int:
        return self.target_patches + self.obstacle_patches + self.end_effector_patches


@dataclass(frozen=True)
class SemanticPatchTokens:
    """Tokens and grouping provenance for reconstruction and diagnostics."""

    tokens: torch.Tensor
    centers: torch.Tensor
    center_indices: torch.Tensor
    neighbor_indices: torch.Tensor
    component_ids: torch.Tensor


def farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministic batched farthest-point sampling over XYZ coordinates.

    The canonical first point is used as the initial center, matching the
    deterministic ``random_start_point=False`` route used by the DyWA FPS
    grouping implementation.  Selection is non-differentiable; gathered patch
    features remain differentiable.
    """

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [batch, points, 3]")
    if count <= 0 or count > points.shape[1]:
        raise ValueError("FPS count must be in [1, number of points]")
    if not points.is_floating_point():
        raise ValueError("FPS points must use a floating-point dtype")

    batch_size, point_count, _ = points.shape
    selected = torch.empty(
        (batch_size, count), device=points.device, dtype=torch.long
    )
    minimum_distance = torch.full(
        (batch_size, point_count), torch.inf, device=points.device, dtype=points.dtype
    )
    farthest = torch.zeros(batch_size, device=points.device, dtype=torch.long)
    batch_indices = torch.arange(batch_size, device=points.device)
    for center_index in range(count):
        selected[:, center_index] = farthest
        center = points[batch_indices, farthest].unsqueeze(1)
        squared_distance = torch.sum((points - center).square(), dim=-1)
        minimum_distance = torch.minimum(minimum_distance, squared_distance)
        farthest = torch.argmax(minimum_distance, dim=1)
    return selected


class DAPLSemanticPatchTokenizer(nn.Module):
    """Encode 16 target, 16 obstacle, and 8 end-effector point patches.

    FPS and kNN operate on XYZ only.  Each gathered point is represented as
    center-relative XYZ plus its physical ``[point_mass, vx, vy, vz]`` values.
    A shared two-layer PointNet-style patch encoder is followed by a fixed
    sinusoidal embedding of the absolute patch-center XYZ coordinates.
    """

    def __init__(self, config: DAPLSemanticPatchTokenizerConfig | None = None):
        super().__init__()
        self.config = DAPLSemanticPatchTokenizerConfig() if config is None else config
        input_dim = self.config.input_dim
        token_dim = self.config.token_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        frequency_count = (token_dim + 5) // 6
        frequencies = torch.exp(
            torch.linspace(0.0, -torch.log(torch.tensor(10000.0)), frequency_count)
        )
        self.register_buffer("position_frequencies", frequencies, persistent=False)
        component_ids = torch.cat(
            (
                torch.full(
                    (self.config.target_patches,), SceneComponent.TARGET, dtype=torch.long
                ),
                torch.full(
                    (self.config.obstacle_patches,), SceneComponent.OBSTACLE, dtype=torch.long
                ),
                torch.full(
                    (self.config.end_effector_patches,),
                    SceneComponent.END_EFFECTOR,
                    dtype=torch.long,
                ),
            )
        )
        self.register_buffer("component_ids", component_ids, persistent=False)

    def forward(self, scene: torch.Tensor) -> SemanticPatchTokens:
        cfg = self.config
        expected = (cfg.scene.total_points, cfg.input_dim)
        if scene.ndim != 3 or scene.shape[1:] != expected:
            raise ValueError(
                f"scene must have shape [batch, {expected[0]}, {expected[1]}], "
                f"got {tuple(scene.shape)}"
            )
        if not scene.is_floating_point():
            raise ValueError("scene must use a floating-point dtype")

        target_end = cfg.scene.target_points
        obstacle_end = target_end + cfg.scene.obstacle_points
        groups = (
            self._tokenize_component(scene[:, :target_end], cfg.target_patches, 0),
            self._tokenize_component(
                scene[:, target_end:obstacle_end], cfg.obstacle_patches, target_end
            ),
            self._tokenize_component(
                scene[:, obstacle_end:], cfg.end_effector_patches, obstacle_end
            ),
        )
        patch_features = torch.cat([group[0] for group in groups], dim=1)
        centers = torch.cat([group[1] for group in groups], dim=1)
        center_indices = torch.cat([group[2] for group in groups], dim=1)
        neighbor_indices = torch.cat([group[3] for group in groups], dim=1)
        component_ids = self.component_ids.to(device=scene.device)
        tokens = patch_features + self._sinusoidal_position_embedding(centers[..., :3])
        return SemanticPatchTokens(
            tokens=tokens,
            centers=centers,
            center_indices=center_indices,
            neighbor_indices=neighbor_indices,
            component_ids=component_ids,
        )

    def _tokenize_component(
        self,
        features: torch.Tensor,
        patch_count: int,
        global_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = features[..., :3]
        center_indices = farthest_point_indices(xyz, patch_count)
        batch_indices = torch.arange(features.shape[0], device=features.device)[:, None]
        centers = features[batch_indices, center_indices]
        squared_distance = torch.sum(
            (centers[..., None, :3] - xyz[:, None, :, :]).square(), dim=-1
        )
        neighbor_indices = torch.topk(
            squared_distance,
            k=self.config.neighbors,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        gather_batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
        patches = features[gather_batch, neighbor_indices]
        local_patches = torch.cat(
            (patches[..., :3] - centers[..., None, :3], patches[..., 3:]), dim=-1
        )
        point_features = self.point_encoder(local_patches)
        patch_features = torch.amax(point_features, dim=2)
        return (
            patch_features,
            centers,
            center_indices + global_offset,
            neighbor_indices + global_offset,
        )

    def _sinusoidal_position_embedding(self, xyz: torch.Tensor) -> torch.Tensor:
        angles = xyz.unsqueeze(-1) * self.position_frequencies
        encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        encoded = encoded.flatten(start_dim=-3)
        return encoded[..., : self.config.token_dim]
