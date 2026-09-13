"""Construction of the paper-defined DAPL physical scene tensor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class SceneComponent(IntEnum):
    """Point provenance kept outside the seven physical input features."""

    TARGET = 0
    OBSTACLE = 1
    END_EFFECTOR = 2


@dataclass(frozen=True)
class DAPLSceneTensorConfig:
    """Fixed scene representation sizes reported by DAPL."""

    target_points: int = 512
    obstacle_points: int = 512
    end_effector_points: int = 256
    canonical_object_points: int = 512
    feature_dim: int = 7

    def __post_init__(self) -> None:
        for name in (
            "target_points",
            "obstacle_points",
            "end_effector_points",
            "canonical_object_points",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.feature_dim != 7:
            raise ValueError("DAPL features are fixed to [x, y, z, mass, vx, vy, vz]")

    @property
    def total_points(self) -> int:
        return self.target_points + self.obstacle_points + self.end_effector_points


@dataclass(frozen=True)
class PhysicalSceneBatch:
    """Batched physical state already transformed into environment coordinates.

    Object and end-effector velocities may be rigid-body linear velocities
    (without a point axis) or explicit per-point velocities.  ``obstacle_mask``
    marks real objects when the object dimension is padded across environments.
    """

    target_points: torch.Tensor
    target_mass: torch.Tensor
    target_velocity: torch.Tensor
    obstacle_points: torch.Tensor
    obstacle_masses: torch.Tensor
    obstacle_velocities: torch.Tensor
    end_effector_points: torch.Tensor
    end_effector_mass: torch.Tensor
    end_effector_velocity: torch.Tensor
    obstacle_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class PhysicalSceneTensor:
    """DAPL input features plus provenance needed for transition alignment."""

    features: torch.Tensor
    component_ids: torch.Tensor
    obstacle_source_indices: torch.Tensor
    obstacle_object_indices: torch.Tensor
    obstacle_point_indices: torch.Tensor
    config: DAPLSceneTensorConfig

    @property
    def target(self) -> torch.Tensor:
        return self.features[:, : self.config.target_points]

    @property
    def obstacles(self) -> torch.Tensor:
        start = self.config.target_points
        return self.features[:, start : start + self.config.obstacle_points]

    @property
    def end_effector(self) -> torch.Tensor:
        return self.features[:, -self.config.end_effector_points :]


class DAPLSceneTensorBuilder:
    """Build ``[B, 1280, 7]`` tensors without an Isaac Sim dependency.

    The returned ``obstacle_source_indices`` must be passed when constructing
    the corresponding future frame.  This preserves point identity across a
    world-model transition even if the nearest-point ordering changes.
    """

    def __init__(
        self,
        config: DAPLSceneTensorConfig | None = None,
        *,
        validate_values: bool = False,
    ) -> None:
        self.config = DAPLSceneTensorConfig() if config is None else config
        # Value validation synchronizes CUDA tensors with the CPU.  Keep it
        # available for fixtures/debugging, but off in the RL observation path.
        self.validate_values = validate_values

    def __call__(
        self,
        batch: PhysicalSceneBatch,
        *,
        obstacle_source_indices: torch.Tensor | None = None,
    ) -> PhysicalSceneTensor:
        return self.build(batch, obstacle_source_indices=obstacle_source_indices)

    def build(
        self,
        batch: PhysicalSceneBatch,
        *,
        obstacle_source_indices: torch.Tensor | None = None,
    ) -> PhysicalSceneTensor:
        cfg = self.config
        target = self._points(batch.target_points, "target_points", cfg.target_points)
        end_effector = self._points(
            batch.end_effector_points, "end_effector_points", cfg.end_effector_points
        )
        obstacles = batch.obstacle_points
        if obstacles.ndim != 4 or obstacles.shape[-1] != 3:
            raise ValueError(
                "obstacle_points must have shape [batch, objects, canonical_points, 3]"
            )
        batch_size, object_count, canonical_points, _ = obstacles.shape
        if target.shape[0] != batch_size or end_effector.shape[0] != batch_size:
            raise ValueError("all point tensors must have the same batch size")
        if canonical_points != cfg.canonical_object_points:
            raise ValueError(
                f"obstacle_points has {canonical_points} canonical points per object; "
                f"expected {cfg.canonical_object_points}"
            )
        if obstacles.device != target.device or end_effector.device != target.device:
            raise ValueError("all point tensors must be on the same device")
        if obstacles.dtype != target.dtype or end_effector.dtype != target.dtype:
            raise ValueError("all point tensors must have the same dtype")
        if not target.is_floating_point():
            raise ValueError("point tensors must use a floating-point dtype")

        obstacle_mask = batch.obstacle_mask
        if obstacle_mask is None:
            obstacle_mask = torch.ones(
                (batch_size, object_count), dtype=torch.bool, device=target.device
            )
        else:
            if obstacle_mask.shape != (batch_size, object_count):
                raise ValueError(
                    f"obstacle_mask must have shape {(batch_size, object_count)}, "
                    f"got {tuple(obstacle_mask.shape)}"
                )
            obstacle_mask = obstacle_mask.to(device=target.device, dtype=torch.bool)

        flat_obstacles = obstacles.reshape(batch_size, object_count * canonical_points, 3)
        if obstacle_source_indices is None:
            obstacle_source_indices = self.select_nearest_obstacles(
                target, flat_obstacles, obstacle_mask, canonical_points
            )
        else:
            obstacle_source_indices = self._validate_selection(
                obstacle_source_indices,
                obstacle_mask,
                canonical_points,
                flat_obstacles.shape[1],
                target.device,
            )

        gather_xyz = obstacle_source_indices.unsqueeze(-1).expand(-1, -1, 3)
        selected_obstacles = torch.gather(flat_obstacles, 1, gather_xyz)

        target_mass = self._scalar(batch.target_mass, batch_size, target, "target_mass")
        end_effector_mass = self._scalar(
            batch.end_effector_mass, batch_size, target, "end_effector_mass"
        )
        obstacle_masses = self._object_scalars(
            batch.obstacle_masses, batch_size, object_count, target, "obstacle_masses"
        )
        if self.validate_values:
            all_masses = torch.cat(
                (target_mass, end_effector_mass, obstacle_masses), dim=1
            )
            if torch.any(~torch.isfinite(all_masses) | (all_masses < 0)).item():
                raise ValueError("scene masses must be finite and non-negative")

        target_velocity = self._point_velocity(
            batch.target_velocity,
            batch_size,
            cfg.target_points,
            target,
            "target_velocity",
        )
        end_effector_velocity = self._point_velocity(
            batch.end_effector_velocity,
            batch_size,
            cfg.end_effector_points,
            target,
            "end_effector_velocity",
        )
        obstacle_velocities = self._obstacle_velocity(
            batch.obstacle_velocities,
            batch_size,
            object_count,
            canonical_points,
            target,
        ).reshape(batch_size, object_count * canonical_points, 3)
        selected_velocities = torch.gather(obstacle_velocities, 1, gather_xyz)

        target_point_mass = (target_mass / cfg.target_points).expand(-1, cfg.target_points)
        end_effector_point_mass = (end_effector_mass / cfg.end_effector_points).expand(
            -1, cfg.end_effector_points
        )
        all_obstacle_point_mass = (
            obstacle_masses.unsqueeze(-1)
            .expand(-1, -1, canonical_points)
            .reshape(batch_size, object_count * canonical_points)
            / canonical_points
        )
        selected_mass = torch.gather(all_obstacle_point_mass, 1, obstacle_source_indices)

        target_features = torch.cat(
            (target, target_point_mass.unsqueeze(-1), target_velocity), dim=-1
        )
        obstacle_features = torch.cat(
            (selected_obstacles, selected_mass.unsqueeze(-1), selected_velocities), dim=-1
        )
        end_effector_features = torch.cat(
            (
                end_effector,
                end_effector_point_mass.unsqueeze(-1),
                end_effector_velocity,
            ),
            dim=-1,
        )
        features = torch.cat(
            (target_features, obstacle_features, end_effector_features), dim=1
        )
        if features.shape != (batch_size, cfg.total_points, cfg.feature_dim):
            raise RuntimeError(f"internal DAPL feature shape error: {tuple(features.shape)}")

        component_ids = torch.cat(
            (
                torch.full(
                    (cfg.target_points,), SceneComponent.TARGET, dtype=torch.int64, device=target.device
                ),
                torch.full(
                    (cfg.obstacle_points,),
                    SceneComponent.OBSTACLE,
                    dtype=torch.int64,
                    device=target.device,
                ),
                torch.full(
                    (cfg.end_effector_points,),
                    SceneComponent.END_EFFECTOR,
                    dtype=torch.int64,
                    device=target.device,
                ),
            )
        )
        return PhysicalSceneTensor(
            features=features,
            component_ids=component_ids,
            obstacle_source_indices=obstacle_source_indices,
            obstacle_object_indices=torch.div(
                obstacle_source_indices, canonical_points, rounding_mode="floor"
            ),
            obstacle_point_indices=torch.remainder(obstacle_source_indices, canonical_points),
            config=cfg,
        )

    def select_nearest_obstacles(
        self,
        target_points: torch.Tensor,
        flat_obstacle_points: torch.Tensor,
        obstacle_mask: torch.Tensor,
        canonical_points: int,
    ) -> torch.Tensor:
        """Select obstacle points nearest to the target centroid."""

        valid_counts = obstacle_mask.sum(dim=1) * canonical_points
        if self.validate_values and torch.any(
            valid_counts < self.config.obstacle_points
        ).item():
            raise ValueError(
                f"each environment needs at least {self.config.obstacle_points} valid obstacle points"
            )
        target_centroid = target_points.mean(dim=1, keepdim=True)
        squared_distance = (flat_obstacle_points - target_centroid).square().sum(dim=-1)
        point_mask = obstacle_mask.unsqueeze(-1).expand(-1, -1, canonical_points).reshape(
            obstacle_mask.shape[0], -1
        )
        squared_distance = squared_distance.masked_fill(~point_mask, torch.inf)
        return torch.topk(
            squared_distance,
            k=self.config.obstacle_points,
            dim=1,
            largest=False,
            sorted=True,
        ).indices

    def _validate_selection(
        self,
        indices: torch.Tensor,
        obstacle_mask: torch.Tensor,
        canonical_points: int,
        source_point_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected = (obstacle_mask.shape[0], self.config.obstacle_points)
        if indices.shape != expected:
            raise ValueError(f"obstacle_source_indices must have shape {expected}")
        indices = indices.to(device=device, dtype=torch.int64)
        if self.validate_values:
            if torch.any((indices < 0) | (indices >= source_point_count)).item():
                raise ValueError("obstacle_source_indices contains an out-of-range index")
            object_indices = torch.div(indices, canonical_points, rounding_mode="floor")
            if torch.any(~torch.gather(obstacle_mask, 1, object_indices)).item():
                raise ValueError("obstacle_source_indices references a padded obstacle")
        return indices

    @staticmethod
    def _points(value: torch.Tensor, name: str, count: int) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (count, 3):
            raise ValueError(f"{name} must have shape [batch, {count}, 3]")
        return value

    @staticmethod
    def _scalar(
        value: torch.Tensor,
        batch_size: int,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.shape == (batch_size,):
            value = value.unsqueeze(-1)
        if value.shape != (batch_size, 1):
            raise ValueError(f"{name} must have shape [batch] or [batch, 1]")
        return value

    @staticmethod
    def _object_scalars(
        value: torch.Tensor,
        batch_size: int,
        object_count: int,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.shape != (batch_size, object_count):
            raise ValueError(f"{name} must have shape [batch, objects]")
        return value

    @staticmethod
    def _point_velocity(
        value: torch.Tensor,
        batch_size: int,
        point_count: int,
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.shape == (batch_size, 3):
            return value.unsqueeze(1).expand(-1, point_count, -1)
        if value.shape == (batch_size, point_count, 3):
            return value
        raise ValueError(
            f"{name} must have shape [batch, 3] or [batch, {point_count}, 3]"
        )

    @staticmethod
    def _obstacle_velocity(
        value: torch.Tensor,
        batch_size: int,
        object_count: int,
        point_count: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        if value.shape == (batch_size, object_count, 3):
            return value.unsqueeze(2).expand(-1, -1, point_count, -1)
        if value.shape == (batch_size, object_count, point_count, 3):
            return value
        raise ValueError(
            "obstacle_velocities must have shape [batch, objects, 3] or "
            f"[batch, objects, {point_count}, 3]"
        )
