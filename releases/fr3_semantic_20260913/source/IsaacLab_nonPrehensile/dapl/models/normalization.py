"""Running normalization for the seven DAPL physical point features."""

from __future__ import annotations

import torch
from torch import nn


class DAPLFeatureNormalizer(nn.Module):
    """Numerically stable running mean/variance over batch and point axes."""

    def __init__(self, feature_dim: int = 7, epsilon: float = 1.0e-5):
        super().__init__()
        if feature_dim <= 0 or epsilon <= 0.0:
            raise ValueError("feature_dim and epsilon must be positive")
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("variance", torch.ones(feature_dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        self._validate(features)
        flat = features.detach().reshape(-1, self.feature_dim)
        batch_count = torch.as_tensor(
            flat.shape[0], device=flat.device, dtype=flat.dtype
        )
        batch_mean = flat.mean(dim=0)
        batch_variance = flat.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        updated_mean = self.mean + delta * batch_count / total_count
        current_m2 = self.variance * self.count
        batch_m2 = batch_variance * batch_count
        correction = delta.square() * self.count * batch_count / total_count
        self.mean.copy_(updated_mean)
        self.variance.copy_((current_m2 + batch_m2 + correction) / total_count)
        self.count.copy_(total_count)

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        self._validate(features)
        return (features - self.mean) / torch.sqrt(self.variance + self.epsilon)

    def denormalize(self, features: torch.Tensor) -> torch.Tensor:
        self._validate(features)
        return features * torch.sqrt(self.variance + self.epsilon) + self.mean

    def forward(self, features: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        if update:
            self.update(features)
        return self.normalize(features)

    def _validate(self, features: torch.Tensor) -> None:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must end in dimension {self.feature_dim}, "
                f"got {tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise ValueError("features must use a floating-point dtype")
