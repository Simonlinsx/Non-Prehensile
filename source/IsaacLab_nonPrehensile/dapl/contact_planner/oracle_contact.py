"""Oracle-affordance contact selection for the M1 single-object planner.

This module deliberately contains no Isaac Lab or policy dependency.  It turns
metric target/hand geometry and oracle ``safe``/``protected`` scores into a
ranked set of short-horizon planar push candidates.  Robot IK and trajectory
execution remain downstream so that a motion generator can reject an otherwise
legal contact without silently weakening the semantic C1 contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OracleContactPlannerConfig:
    """Numerical contract for safe-contact candidate generation."""

    safe_threshold: float = 0.25
    protected_threshold: float = 0.25
    contact_distance_m: float = 0.010
    forbidden_clearance_m: float = 0.020
    approach_clearance_m: float = 0.015
    support_clearance_m: float = 0.002
    precontact_standoff_m: float = 0.050
    contact_penetration_m: float = 0.0
    minimum_push_distance_m: float = 0.008
    maximum_push_distance_m: float = 0.015
    push_distance_samples: int = 1
    push_overshoot_m: float = 0.004
    translation_efficiency: float = 0.35
    rotation_efficiency: float = 0.40
    safe_point_candidates: int = 32
    hand_point_candidates: int = 12
    push_direction_samples: int = 7
    push_direction_span_deg: float = 60.0
    hand_yaw_samples: int = 5
    hand_yaw_span_deg: float = 60.0
    path_samples: int = 11
    output_candidates: int = 16
    trailing_weight: float = 0.05
    travel_weight: float = 0.20
    clearance_weight: float = 0.10
    goal_weight: float = 1.00
    yaw_weight_m_per_rad: float = 2.00
    direction_deviation_weight_m_per_rad: float = 0.10
    hand_yaw_deviation_weight_m_per_rad: float = 0.03
    moment_arm_neutral_band_m: float = 0.002

    def __post_init__(self) -> None:
        if not 0.0 <= self.safe_threshold <= 1.0:
            raise ValueError("safe_threshold must be in [0, 1]")
        if not 0.0 <= self.protected_threshold <= 1.0:
            raise ValueError("protected_threshold must be in [0, 1]")
        positive = {
            "contact_distance_m": self.contact_distance_m,
            "forbidden_clearance_m": self.forbidden_clearance_m,
            "approach_clearance_m": self.approach_clearance_m,
            "precontact_standoff_m": self.precontact_standoff_m,
            "minimum_push_distance_m": self.minimum_push_distance_m,
            "maximum_push_distance_m": self.maximum_push_distance_m,
            "translation_efficiency": self.translation_efficiency,
            "rotation_efficiency": self.rotation_efficiency,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.contact_penetration_m < 0.0:
            raise ValueError("contact_penetration_m must be non-negative")
        if self.moment_arm_neutral_band_m < 0.0:
            raise ValueError("moment_arm_neutral_band_m must be non-negative")
        if self.support_clearance_m < 0.0:
            raise ValueError("support_clearance_m must be non-negative")
        if self.minimum_push_distance_m > self.maximum_push_distance_m:
            raise ValueError("minimum_push_distance_m exceeds maximum_push_distance_m")
        for name in (
            "safe_point_candidates",
            "hand_point_candidates",
            "push_direction_samples",
            "push_distance_samples",
            "hand_yaw_samples",
            "path_samples",
            "output_candidates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.path_samples < 2:
            raise ValueError("path_samples must be at least two")
        if self.push_direction_span_deg < 0.0 or self.push_direction_span_deg > 180.0:
            raise ValueError("push_direction_span_deg must be in [0, 180]")
        if self.hand_yaw_span_deg < 0.0 or self.hand_yaw_span_deg > 180.0:
            raise ValueError("hand_yaw_span_deg must be in [0, 180]")


@dataclass
class OraclePlanningScene:
    """Batched metric scene supplied by oracle simulation or RGB-D perception.

    Shapes are ``B x N x 3`` for target points, ``B x N`` for semantic
    scores, ``B x 3`` for poses, and either ``H x 3`` or ``B x H x 3`` for
    canonical hand samples expressed relative to the TCP.
    """

    target_points: torch.Tensor
    safe_scores: torch.Tensor
    protected_scores: torch.Tensor
    target_position: torch.Tensor
    goal_position: torch.Tensor
    tcp_position: torch.Tensor
    hand_points_local: torch.Tensor
    tcp_rotation: torch.Tensor | None = None
    yaw_error: torch.Tensor | None = None

    def validate(self) -> None:
        if self.target_points.ndim != 3 or self.target_points.shape[-1] != 3:
            raise ValueError("target_points must have shape [B, N, 3]")
        batch_size, point_count, _ = self.target_points.shape
        expected_scores = (batch_size, point_count)
        if self.safe_scores.shape != expected_scores:
            raise ValueError("safe_scores must have shape [B, N]")
        if self.protected_scores.shape != expected_scores:
            raise ValueError("protected_scores must have shape [B, N]")
        for name in ("target_position", "goal_position", "tcp_position"):
            value = getattr(self, name)
            if value.shape != (batch_size, 3):
                raise ValueError(f"{name} must have shape [B, 3]")
        if self.hand_points_local.ndim == 2:
            if self.hand_points_local.shape[-1] != 3:
                raise ValueError("hand_points_local must end in dimension 3")
        elif self.hand_points_local.ndim == 3:
            if self.hand_points_local.shape[0] != batch_size:
                raise ValueError("batched hand_points_local must match scene batch")
            if self.hand_points_local.shape[-1] != 3:
                raise ValueError("hand_points_local must end in dimension 3")
        else:
            raise ValueError("hand_points_local must have shape [H, 3] or [B, H, 3]")
        if self.yaw_error is not None and self.yaw_error.shape != (batch_size,):
            raise ValueError("yaw_error must have shape [B]")
        if self.tcp_rotation is not None and self.tcp_rotation.shape != (
            batch_size,
            3,
            3,
        ):
            raise ValueError("tcp_rotation must have shape [B, 3, 3]")
        tensors = (
            self.target_points,
            self.safe_scores,
            self.protected_scores,
            self.target_position,
            self.goal_position,
            self.tcp_position,
            self.hand_points_local,
        )
        if self.yaw_error is not None:
            tensors = (*tensors, self.yaw_error)
        if self.tcp_rotation is not None:
            tensors = (*tensors, self.tcp_rotation)
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("planning scene contains non-finite values")
        if torch.any(self.safe_scores < 0.0) or torch.any(self.safe_scores > 1.0):
            raise ValueError("safe_scores must be probabilities in [0, 1]")
        if torch.any(self.protected_scores < 0.0) or torch.any(
            self.protected_scores > 1.0
        ):
            raise ValueError("protected_scores must be probabilities in [0, 1]")


@dataclass
class OracleContactCandidateBatch:
    """Ranked candidates, padded to ``K=output_candidates`` per environment."""

    valid: torch.Tensor
    contact_tcp: torch.Tensor
    precontact_tcp: torch.Tensor
    push_tcp: torch.Tensor
    hand_rotation: torch.Tensor
    push_direction: torch.Tensor
    approach_direction: torch.Tensor
    hand_yaw_offset: torch.Tensor
    push_distance: torch.Tensor
    score: torch.Tensor
    target_point_index: torch.Tensor
    contact_point: torch.Tensor
    hand_point_index: torch.Tensor
    safe_distance: torch.Tensor
    forbidden_clearance: torch.Tensor
    support_clearance: torch.Tensor
    approach_clearance: torch.Tensor
    predicted_planar_error: torch.Tensor
    contact_moment_arm: torch.Tensor
    predicted_yaw_error: torch.Tensor

    @property
    def any_valid(self) -> torch.Tensor:
        return self.valid.any(dim=1)


def horizontal_push_frame(push_direction: torch.Tensor) -> torch.Tensor:
    """Construct a hand rotation whose local +Z axis is the planar push axis.

    The accepted Franka reset uses a horizontal hand: local -X points upward,
    local +Z points forward, and local +Y completes the right-handed frame.
    Keeping this convention makes the planned pose compatible with the existing
    DOMINO C1 hand point cache and avoids arbitrary roll/pitch changes.
    """

    if push_direction.ndim != 2 or push_direction.shape[-1] != 3:
        raise ValueError("push_direction must have shape [B, 3]")
    planar = push_direction.clone()
    planar[:, 2] = 0.0
    norm = torch.linalg.vector_norm(planar, dim=1, keepdim=True)
    if torch.any(norm <= 1.0e-8):
        raise ValueError("push direction must have non-zero planar magnitude")
    local_z = planar / norm
    local_x = torch.zeros_like(local_z)
    local_x[:, 2] = -1.0
    local_y = torch.linalg.cross(local_z, local_x, dim=1)
    return torch.stack((local_x, local_y, local_z), dim=2)


class OracleSafeContactPlanner:
    """Rank C1-legal short planar pushes from oracle semantic point clouds."""

    def __init__(self, cfg: OracleContactPlannerConfig | None = None) -> None:
        self.cfg = cfg or OracleContactPlannerConfig()

    def plan(self, scene: OraclePlanningScene) -> OracleContactCandidateBatch:
        scene.validate()
        cfg = self.cfg
        batch_size = scene.target_points.shape[0]
        device = scene.target_points.device
        dtype = scene.target_points.dtype
        if scene.hand_points_local.ndim == 2:
            hand_local = scene.hand_points_local.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        else:
            hand_local = scene.hand_points_local

        goal_delta = scene.goal_position - scene.target_position
        goal_delta[:, 2] = 0.0
        goal_distance = torch.linalg.vector_norm(goal_delta, dim=1)
        safe_goal_distance = torch.clamp(goal_distance, min=1.0e-8)
        base_push_direction = goal_delta / safe_goal_distance.unsqueeze(1)
        base_push_direction[goal_distance <= 1.0e-8] = torch.tensor(
            (1.0, 0.0, 0.0), device=device, dtype=dtype
        )
        push_distance = torch.clamp(
            goal_distance / cfg.translation_efficiency + cfg.push_overshoot_m,
            min=cfg.minimum_push_distance_m,
            max=cfg.maximum_push_distance_m,
        )
        yaw_error = (
            torch.zeros(batch_size, device=device, dtype=dtype)
            if scene.yaw_error is None
            else scene.yaw_error
        )
        completed = (goal_distance <= 1.0e-8) & (torch.abs(yaw_error) <= 1.0e-3)

        output = self._empty_output(
            batch_size, cfg.output_candidates, device=device, dtype=dtype
        )

        for env_id in range(batch_size):
            if completed[env_id]:
                continue
            points = scene.target_points[env_id]
            safe_mask = (
                scene.safe_scores[env_id] >= cfg.safe_threshold
            ) & (scene.protected_scores[env_id] < cfg.protected_threshold)
            if not bool(safe_mask.any()):
                continue
            safe_indices = torch.nonzero(safe_mask, as_tuple=False).flatten()
            safe_points = points[safe_indices]
            forbidden_points = points[~safe_mask]
            safe_center = safe_points.mean(dim=0)
            support_height = points[:, 2].min()
            centered_radius = points[:, :2] - scene.target_position[env_id, :2]
            planar_gyration_sq = torch.clamp(
                torch.mean(torch.sum(centered_radius.square(), dim=1)), min=1.0e-4
            )
            base_angle = torch.atan2(
                base_push_direction[env_id, 1], base_push_direction[env_id, 0]
            )
            direction_offsets = torch.linspace(
                -math.radians(cfg.push_direction_span_deg),
                math.radians(cfg.push_direction_span_deg),
                cfg.push_direction_samples,
                device=device,
                dtype=dtype,
            )
            direction_angles = base_angle + direction_offsets
            direction_variants = torch.stack(
                (
                    torch.cos(direction_angles),
                    torch.sin(direction_angles),
                    torch.zeros_like(direction_angles),
                ),
                dim=1,
            )
            if scene.tcp_rotation is None:
                rotations = horizontal_push_frame(direction_variants)
                direction_ids = torch.arange(
                    cfg.push_direction_samples, device=device
                )
                hand_yaw_offsets = torch.zeros_like(direction_offsets)
            else:
                # A planar push does not require the hand's local +Z axis to
                # equal the force direction.  Keeping the live horizontal
                # wrist pose avoids needless yaw flips; the leading surface
                # is selected geometrically for each push direction below.
                yaw_offsets = torch.linspace(
                    -math.radians(cfg.hand_yaw_span_deg),
                    math.radians(cfg.hand_yaw_span_deg),
                    cfg.hand_yaw_samples,
                    device=device,
                    dtype=dtype,
                )
                cosine, sine = torch.cos(yaw_offsets), torch.sin(yaw_offsets)
                zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
                yaw_rotations = torch.stack(
                    (
                        cosine,
                        -sine,
                        zero,
                        sine,
                        cosine,
                        zero,
                        zero,
                        zero,
                        one,
                    ),
                    dim=1,
                ).reshape(-1, 3, 3)
                wrist_rotations = (
                    yaw_rotations @ scene.tcp_rotation[env_id].unsqueeze(0)
                )
                direction_ids = torch.arange(
                    cfg.push_direction_samples, device=device
                ).repeat_interleave(cfg.hand_yaw_samples)
                direction_variants = direction_variants.repeat_interleave(
                    cfg.hand_yaw_samples, dim=0
                )
                direction_offsets = direction_offsets.repeat_interleave(
                    cfg.hand_yaw_samples
                )
                rotations = wrist_rotations.repeat(
                    cfg.push_direction_samples, 1, 1
                )
                hand_yaw_offsets = yaw_offsets.repeat(cfg.push_direction_samples)
            kept: list[dict[str, object]] = []

            for variant_index in range(direction_variants.shape[0]):
                direction_index = int(direction_ids[variant_index].item())
                direction = direction_variants[variant_index]
                rotation = rotations[variant_index]
                rotated_hand = hand_local[env_id] @ rotation.T
                hand_count = min(cfg.hand_point_candidates, rotated_hand.shape[0])

                trailing_projection = (safe_points - safe_center) @ direction
                safe_count = min(cfg.safe_point_candidates, safe_points.shape[0])
                trailing_order = torch.topk(
                    -trailing_projection, k=safe_count, largest=True
                ).indices
                chosen_safe_indices = safe_indices[trailing_order]
                chosen_safe_points = points[chosen_safe_indices]
                # Contact access and object motion are distinct decisions.
                # Approach the chosen surface point from outside the object,
                # then execute the independently sampled planar push.  Tying
                # both directions together makes a safe handle unreachable
                # whenever the functional head lies behind the desired force.
                outward = chosen_safe_points - scene.target_position[env_id]
                outward[:, 2] = 0.0
                outward_norm = torch.linalg.vector_norm(
                    outward[:, :2], dim=1, keepdim=True
                )
                fallback_outward = -direction.unsqueeze(0).expand_as(outward)
                outward = torch.where(
                    (outward_norm > 1.0e-6).expand_as(outward),
                    outward / torch.clamp(outward_norm, min=1.0e-6),
                    fallback_outward,
                )
                approach_direction = -outward
                hand_projection = approach_direction @ rotated_hand.T
                leading_hand_indices = torch.topk(
                    hand_projection, k=hand_count, dim=1, largest=True
                ).indices
                leading_hand = rotated_hand[leading_hand_indices]
                contact_tcp = (
                    chosen_safe_points[:, None, :] - leading_hand
                    + cfg.contact_penetration_m * approach_direction[:, None, :]
                ).reshape(-1, 3)
                hand_index = leading_hand_indices.reshape(-1)
                candidate_approach_direction = (
                    approach_direction[:, None, :]
                    .expand(-1, hand_count, -1)
                    .reshape(-1, 3)
                )
                posed_hand = rotated_hand.unsqueeze(0) + contact_tcp.unsqueeze(1)
                safe_distance, nearest_safe_local_index = (
                    self._minimum_distance_with_destination_index(
                        posed_hand, safe_points
                    )
                )
                contact_patch_center = self._contact_patch_center(
                    posed_hand,
                    safe_points,
                    influence_radius=2.0 * cfg.contact_distance_m,
                )
                forbidden_clearance = self._minimum_distance(
                    posed_hand, forbidden_points, empty_value=torch.inf
                )
                support_clearance = posed_hand[..., 2].amin(dim=1) - support_height
                legal = (safe_distance <= cfg.contact_distance_m) & (
                    forbidden_clearance > cfg.forbidden_clearance_m
                ) & (support_clearance > cfg.support_clearance_m)
                legal_indices = torch.nonzero(legal, as_tuple=False).flatten()
                if legal_indices.numel() == 0:
                    continue

                candidate_tcp = contact_tcp[legal_indices]
                candidate_target_index = safe_indices[
                    nearest_safe_local_index[legal_indices]
                ]
                candidate_hand_index = hand_index[legal_indices]
                candidate_safe_distance = safe_distance[legal_indices]
                candidate_forbidden_clearance = forbidden_clearance[legal_indices]
                candidate_support_clearance = support_clearance[legal_indices]
                candidate_contact_point = contact_patch_center[legal_indices]
                candidate_approach_direction = candidate_approach_direction[
                    legal_indices
                ]
                candidate_precontact = (
                    candidate_tcp
                    - cfg.precontact_standoff_m * candidate_approach_direction
                )
                candidate_trailing = (
                    points[candidate_target_index] - safe_center
                ) @ direction
                target_extent = torch.clamp(
                    trailing_projection.max() - trailing_projection.min(), min=1.0e-4
                )
                normalized_trailing = (
                    candidate_trailing - trailing_projection.min()
                ) / target_extent
                travel = torch.linalg.vector_norm(
                    candidate_precontact - scene.tcp_position[env_id], dim=1
                )
                predicted_translation_delta = (
                    cfg.translation_efficiency
                    * push_distance[env_id]
                    * direction[:2]
                )
                predicted_planar_error = torch.linalg.vector_norm(
                    goal_delta[env_id, :2] - predicted_translation_delta
                ).expand_as(travel)
                contact_radius = (
                    candidate_contact_point[:, :2]
                    - scene.target_position[env_id, :2]
                )
                candidate_moment_arm = (
                    contact_radius[:, 0] * direction[1]
                    - contact_radius[:, 1] * direction[0]
                )
                predicted_yaw_delta = (
                    cfg.rotation_efficiency
                    * push_distance[env_id]
                    * candidate_moment_arm
                    / planar_gyration_sq
                )
                predicted_yaw_error = torch.abs(
                    yaw_error[env_id] - predicted_yaw_delta
                )
                capped_clearance = torch.clamp(
                    candidate_forbidden_clearance,
                    max=4.0 * cfg.forbidden_clearance_m,
                )
                preliminary_score = (
                    cfg.goal_weight * predicted_planar_error
                    + cfg.yaw_weight_m_per_rad * predicted_yaw_error
                    + cfg.direction_deviation_weight_m_per_rad
                    * torch.abs(direction_offsets[variant_index])
                    + cfg.hand_yaw_deviation_weight_m_per_rad
                    * torch.abs(hand_yaw_offsets[variant_index])
                    + cfg.trailing_weight * normalized_trailing
                    + cfg.travel_weight * travel
                    - cfg.clearance_weight * capped_clearance
                )
                path_check_count = min(
                    max(cfg.output_candidates * 2, cfg.output_candidates),
                    preliminary_score.shape[0],
                )
                check_indices = torch.topk(
                    -preliminary_score, k=path_check_count, largest=True
                ).indices
                for candidate_index in check_indices.tolist():
                    approach_clearance = self._approach_clearance(
                        current_tcp=scene.tcp_position[env_id],
                        precontact_tcp=candidate_precontact[candidate_index],
                        rotated_hand=rotated_hand,
                        target_points=forbidden_points,
                    )
                    if approach_clearance <= cfg.approach_clearance_m:
                        continue
                    kept.append(
                        {
                            "direction_index": direction_index,
                            "score": float(preliminary_score[candidate_index].item()),
                            "contact_tcp": candidate_tcp[candidate_index],
                            "precontact_tcp": candidate_precontact[candidate_index],
                            "rotation": rotation,
                            "direction": direction,
                            "approach_direction": candidate_approach_direction[
                                candidate_index
                            ],
                            "hand_yaw_offset": hand_yaw_offsets[variant_index],
                            "target_index": candidate_target_index[candidate_index],
                            "contact_point": candidate_contact_point[candidate_index],
                            "hand_index": candidate_hand_index[candidate_index],
                            "safe_distance": candidate_safe_distance[candidate_index],
                            "forbidden_clearance": candidate_forbidden_clearance[
                                candidate_index
                            ],
                            "support_clearance": candidate_support_clearance[
                                candidate_index
                            ],
                            "approach_clearance": approach_clearance,
                            "push_distance": push_distance[env_id],
                            "distance_index": 0,
                            "predicted_planar_error": predicted_planar_error[
                                candidate_index
                            ],
                            "moment_arm": candidate_moment_arm[candidate_index],
                            "predicted_yaw_error": predicted_yaw_error[candidate_index],
                        }
                    )

            if cfg.push_distance_samples > 1 and kept:
                maximum_distance = float(push_distance[env_id].item())
                if maximum_distance > cfg.minimum_push_distance_m + 1.0e-8:
                    distance_variants = torch.linspace(
                        cfg.minimum_push_distance_m,
                        maximum_distance,
                        cfg.push_distance_samples,
                        device=device,
                        dtype=dtype,
                    )
                    expanded: list[dict[str, object]] = []
                    for candidate in kept:
                        geometry_score = (
                            float(candidate["score"])
                            - cfg.goal_weight
                            * float(candidate["predicted_planar_error"])
                            - cfg.yaw_weight_m_per_rad
                            * float(candidate["predicted_yaw_error"])
                        )
                        direction = candidate["direction"]
                        moment_arm = candidate["moment_arm"]
                        for distance_index, distance in enumerate(
                            distance_variants
                        ):
                            predicted_translation_delta = (
                                cfg.translation_efficiency
                                * distance
                                * direction[:2]
                            )
                            predicted_planar_error = torch.linalg.vector_norm(
                                goal_delta[env_id, :2]
                                - predicted_translation_delta
                            )
                            predicted_yaw_delta = (
                                cfg.rotation_efficiency
                                * distance
                                * moment_arm
                                / planar_gyration_sq
                            )
                            predicted_yaw_error = torch.abs(
                                yaw_error[env_id] - predicted_yaw_delta
                            )
                            variant = dict(candidate)
                            variant["distance_index"] = distance_index
                            variant["push_distance"] = distance
                            variant["predicted_planar_error"] = (
                                predicted_planar_error
                            )
                            variant["predicted_yaw_error"] = predicted_yaw_error
                            variant["score"] = float(
                                geometry_score
                                + cfg.goal_weight * predicted_planar_error
                                + cfg.yaw_weight_m_per_rad
                                * predicted_yaw_error
                            )
                            expanded.append(variant)
                    kept = expanded

            kept.sort(key=lambda item: float(item["score"]))
            # Preserve both force-direction and torque-mode diversity.
            # Physics can disagree with the analytic moment-arm sign, so
            # downstream rollout ranking must receive positive, near-zero,
            # and negative torque hypotheses instead of a top-K containing
            # only the analytically preferred sign.
            def moment_bin(candidate: dict[str, object]) -> int:
                moment = float(candidate["moment_arm"])
                if moment < -cfg.moment_arm_neutral_band_m:
                    return -1
                if moment > cfg.moment_arm_neutral_band_m:
                    return 1
                return 0

            best_per_moment: list[dict[str, object]] = []
            seen_moments: set[int] = set()
            for candidate in kept:
                bin_id = moment_bin(candidate)
                if bin_id in seen_moments:
                    continue
                seen_moments.add(bin_id)
                best_per_moment.append(candidate)

            best_per_distance: list[dict[str, object]] = []
            seen_distances: set[int] = set()
            for candidate in kept:
                distance_index = int(candidate["distance_index"])
                if distance_index in seen_distances:
                    continue
                seen_distances.add(distance_index)
                best_per_distance.append(candidate)

            # Without direction diversity, the top-K can still be many hand
            # samples for one push axis, leaving IK no useful fallback.
            best_per_direction: list[dict[str, object]] = []
            seen_directions: set[int] = set()
            for candidate in kept:
                direction_index = int(candidate["direction_index"])
                if direction_index in seen_directions:
                    continue
                seen_directions.add(direction_index)
                best_per_direction.append(candidate)
            selected: list[dict[str, object]] = []
            seeded_ids: set[int] = set()
            # Torque and push magnitude are the primary physics hypotheses;
            # reserve their representatives before spending the remaining
            # budget on force-direction diversity.
            for group in (best_per_moment, best_per_distance, best_per_direction):
                group.sort(key=lambda item: float(item["score"]))
                for candidate in group:
                    if id(candidate) in seeded_ids:
                        continue
                    selected.append(candidate)
                    seeded_ids.add(id(candidate))
                    if len(selected) >= cfg.output_candidates:
                        break
                if len(selected) >= cfg.output_candidates:
                    break
            selected_ids = {id(candidate) for candidate in selected}
            for candidate in kept:
                if len(selected) >= cfg.output_candidates:
                    break
                if id(candidate) in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(id(candidate))
            selected.sort(key=lambda item: float(item["score"]))
            for rank, candidate in enumerate(selected):
                output.valid[env_id, rank] = True
                output.contact_tcp[env_id, rank] = candidate["contact_tcp"]
                output.precontact_tcp[env_id, rank] = candidate["precontact_tcp"]
                candidate_push_distance = candidate["push_distance"]
                output.push_tcp[env_id, rank] = (
                    candidate["contact_tcp"]
                    + candidate_push_distance * candidate["direction"]
                )
                output.push_direction[env_id, rank] = candidate["direction"]
                output.approach_direction[env_id, rank] = candidate[
                    "approach_direction"
                ]
                output.hand_yaw_offset[env_id, rank] = candidate[
                    "hand_yaw_offset"
                ]
                output.hand_rotation[env_id, rank] = candidate["rotation"]
                output.push_distance[env_id, rank] = candidate_push_distance
                output.score[env_id, rank] = candidate["score"]
                output.target_point_index[env_id, rank] = candidate["target_index"]
                output.contact_point[env_id, rank] = candidate["contact_point"]
                output.hand_point_index[env_id, rank] = candidate["hand_index"]
                output.safe_distance[env_id, rank] = candidate["safe_distance"]
                output.forbidden_clearance[env_id, rank] = candidate[
                    "forbidden_clearance"
                ]
                output.support_clearance[env_id, rank] = candidate[
                    "support_clearance"
                ]
                output.approach_clearance[env_id, rank] = candidate[
                    "approach_clearance"
                ]
                output.predicted_planar_error[env_id, rank] = candidate[
                    "predicted_planar_error"
                ]
                output.contact_moment_arm[env_id, rank] = candidate["moment_arm"]
                output.predicted_yaw_error[env_id, rank] = candidate[
                    "predicted_yaw_error"
                ]

        return output

    def _approach_clearance(
        self,
        *,
        current_tcp: torch.Tensor,
        precontact_tcp: torch.Tensor,
        rotated_hand: torch.Tensor,
        target_points: torch.Tensor,
    ) -> float:
        if target_points.numel() == 0:
            return math.inf
        alpha = torch.linspace(
            0.0,
            1.0,
            self.cfg.path_samples,
            device=current_tcp.device,
            dtype=current_tcp.dtype,
        )
        tcp_path = current_tcp + alpha[:, None] * (precontact_tcp - current_tcp)
        swept_hand = rotated_hand.unsqueeze(0) + tcp_path.unsqueeze(1)
        clearance = self._minimum_distance(swept_hand, target_points)
        return float(clearance.min().item())

    @staticmethod
    def _minimum_distance(
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        empty_value: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Minimum point-set distance for ``source`` batches without huge broadcasts."""

        if destination.numel() == 0:
            if empty_value is None:
                raise ValueError("destination is empty")
            return torch.full(
                source.shape[:-2],
                float(empty_value),
                device=source.device,
                dtype=source.dtype,
            )
        flat_source = source.reshape(-1, source.shape[-2], 3)
        expanded_destination = destination.unsqueeze(0).expand(
            flat_source.shape[0], -1, -1
        )
        distance = torch.cdist(flat_source, expanded_destination).amin(dim=(1, 2))
        return distance.reshape(source.shape[:-2])

    @staticmethod
    def _minimum_distance_with_destination_index(
        source: torch.Tensor, destination: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return set distance and the nearest destination point index."""

        if destination.numel() == 0:
            raise ValueError("destination is empty")
        flat_source = source.reshape(-1, source.shape[-2], 3)
        expanded_destination = destination.unsqueeze(0).expand(
            flat_source.shape[0], -1, -1
        )
        pairwise = torch.cdist(flat_source, expanded_destination)
        flat_index = pairwise.reshape(pairwise.shape[0], -1).argmin(dim=1)
        distance = pairwise.reshape(pairwise.shape[0], -1).gather(
            1, flat_index[:, None]
        )[:, 0]
        destination_index = flat_index.remainder(destination.shape[0])
        output_shape = source.shape[:-2]
        return distance.reshape(output_shape), destination_index.reshape(output_shape)

    @staticmethod
    def _contact_patch_center(
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        influence_radius: float,
    ) -> torch.Tensor:
        """Estimate a pressure center from all nearby semantic surface points."""

        if influence_radius <= 0.0 or destination.numel() == 0:
            raise ValueError("contact patch requires a positive radius and points")
        flat_source = source.reshape(-1, source.shape[-2], 3)
        expanded_destination = destination.unsqueeze(0).expand(
            flat_source.shape[0], -1, -1
        )
        target_distance = torch.cdist(
            flat_source, expanded_destination
        ).amin(dim=1)
        weights = torch.clamp(
            1.0 - target_distance / influence_radius, min=0.0
        ).square()
        nearest = target_distance.argmin(dim=1)
        empty = weights.sum(dim=1) <= 1.0e-8
        if bool(empty.any()):
            weights[empty] = 0.0
            weights[empty, nearest[empty]] = 1.0
        center = torch.sum(
            weights.unsqueeze(-1) * expanded_destination, dim=1
        ) / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0e-8)
        return center.reshape(*source.shape[:-2], 3)

    @staticmethod
    def _empty_output(
        batch_size: int,
        candidate_count: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> OracleContactCandidateBatch:
        vector_shape = (batch_size, candidate_count, 3)
        scalar_shape = (batch_size, candidate_count)
        return OracleContactCandidateBatch(
            valid=torch.zeros(scalar_shape, device=device, dtype=torch.bool),
            contact_tcp=torch.full(vector_shape, torch.nan, device=device, dtype=dtype),
            precontact_tcp=torch.full(vector_shape, torch.nan, device=device, dtype=dtype),
            push_tcp=torch.full(vector_shape, torch.nan, device=device, dtype=dtype),
            hand_rotation=torch.full(
                (batch_size, candidate_count, 3, 3),
                torch.nan,
                device=device,
                dtype=dtype,
            ),
            push_direction=torch.full(vector_shape, torch.nan, device=device, dtype=dtype),
            approach_direction=torch.full(
                vector_shape, torch.nan, device=device, dtype=dtype
            ),
            hand_yaw_offset=torch.full(
                scalar_shape, torch.nan, device=device, dtype=dtype
            ),
            push_distance=torch.full(scalar_shape, torch.nan, device=device, dtype=dtype),
            score=torch.full(scalar_shape, torch.inf, device=device, dtype=dtype),
            target_point_index=torch.full(
                scalar_shape, -1, device=device, dtype=torch.long
            ),
            contact_point=torch.full(
                vector_shape, torch.nan, device=device, dtype=dtype
            ),
            hand_point_index=torch.full(
                scalar_shape, -1, device=device, dtype=torch.long
            ),
            safe_distance=torch.full(scalar_shape, torch.inf, device=device, dtype=dtype),
            forbidden_clearance=torch.full(
                scalar_shape, -torch.inf, device=device, dtype=dtype
            ),
            support_clearance=torch.full(
                scalar_shape, -torch.inf, device=device, dtype=dtype
            ),
            approach_clearance=torch.full(
                scalar_shape, -torch.inf, device=device, dtype=dtype
            ),
            predicted_planar_error=torch.full(
                scalar_shape, torch.inf, device=device, dtype=dtype
            ),
            contact_moment_arm=torch.full(
                scalar_shape, torch.nan, device=device, dtype=dtype
            ),
            predicted_yaw_error=torch.full(
                scalar_shape, torch.inf, device=device, dtype=dtype
            ),
        )
