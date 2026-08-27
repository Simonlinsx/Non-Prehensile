#!/usr/bin/env python3
"""Audit goal-wrench safe-contact selection on the actual DOMINO hammer.

This is simulator-independent evidence for the contact-manifold reward.  It
uses the real collision mesh, canonical safe/protected labels, and every goal
pose in a teacher manifest.  The report compares the legacy translation-only
trailing subset against the wrench-aware subset without exposing either subset
to the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from dapl.domino import (
    DominoDataPaths,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
)
from dapl.metrics import (
    signed_yaw_contact_moment_score,
    wrench_aware_contact_support_score,
    yaw_compatible_safe_point_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--domino-root", type=Path, required=True)
    parser.add_argument("--asset-id", default="020_hammer:0")
    parser.add_argument("--sample-count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--minimum-safe-score", type=float, default=0.25)
    parser.add_argument("--side-band-m", type=float, default=0.015)
    parser.add_argument("--yaw-moment-weight", type=float, default=1.0)
    parser.add_argument("--yaw-activation-rad", type=float, default=0.10)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "combined",
            "hierarchical",
            "yaw_first",
            "yaw_positive",
            "maximin",
        ),
        default="combined",
    )
    parser.add_argument("--yaw-side-band-m", type=float, default=0.005)
    parser.add_argument("--yaw-compatibility-floor-m", type=float, default=0.002)
    parser.add_argument(
        "--normalized-score-band",
        type=float,
        default=0.10,
        help=(
            "Dimensionless band below the best normalized maximin score. "
            "Only used by selection-mode=maximin."
        ),
    )
    parser.add_argument("--minimum-yaw-error-rad", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] *= -1.0
    return result


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_apply(quaternion: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    vector_quaternion = torch.cat(
        (torch.zeros_like(vectors[..., :1]), vectors), dim=-1
    )
    expanded = quaternion.expand(vector_quaternion.shape[0], -1)
    return _quat_mul(
        _quat_mul(expanded, vector_quaternion), _quat_conjugate(expanded)
    )[..., 1:]


def _relative_yaw(current: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    relative = _quat_mul(goal, _quat_conjugate(current))
    w, x, y, z = relative.unbind(-1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y.square() + z.square()),
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    if args.side_band_m < 0.0:
        raise ValueError("side-band-m must be non-negative")
    if (
        args.yaw_side_band_m < 0.0
        or args.yaw_compatibility_floor_m < 0.0
        or args.minimum_yaw_error_rad < 0.0
    ):
        raise ValueError("yaw side band and minimum yaw error must be non-negative")
    if not 0.0 <= args.normalized_score_band <= 1.0:
        raise ValueError("normalized-score-band must be in [0, 1]")

    paths = DominoDataPaths.resolve(args.domino_root)
    asset = paths.require_source_asset(args.asset_id)
    annotation = load_domino_affordance_annotation(asset)
    mesh_or_scene = trimesh.load(asset.collision_mesh, force="scene")
    mesh = (
        mesh_or_scene.to_geometry()
        if isinstance(mesh_or_scene, trimesh.Scene)
        else mesh_or_scene
    )
    np.random.seed(args.seed)
    sampled_raw, _ = trimesh.sample.sample_surface(mesh, args.sample_count)
    sampled_raw = torch.from_numpy(sampled_raw).to(torch.float32)
    semantics = domino_point_affordance_features(sampled_raw, annotation)
    safe_mask = semantics[:, 0] >= float(args.minimum_safe_score)
    protected_mask = semantics[:, 1] >= float(args.minimum_safe_score)
    if not bool(safe_mask.any()):
        raise RuntimeError("sampled hammer has no safe points")
    sampled_m = sampled_raw * torch.tensor(annotation.scale, dtype=torch.float32)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    old_compatibility: list[float] = []
    new_compatibility: list[float] = []
    old_trailing: list[float] = []
    new_trailing: list[float] = []
    selected_overlap: list[float] = []
    old_selected_fraction: list[float] = []
    new_selected_fraction: list[float] = []
    yaw_errors: list[float] = []
    by_yaw_sign: dict[str, dict[str, list[float] | int]] = {
        sign: {
            "yaw_errors": [],
            "old_compatibility": [],
            "new_compatibility": [],
            "old_trailing": [],
            "new_trailing": [],
            "selected_overlap": [],
            "old_selected_fraction": [],
            "new_selected_fraction": [],
            "protected_selected_old": 0,
            "protected_selected_new": 0,
        }
        for sign in ("negative", "positive")
    }
    protected_selected_old = 0
    protected_selected_new = 0
    yaw_positive_fallback_scenes = 0

    for row in rows:
        target = next(
            item for item in row["objects"] if item["instance_id"] == "target"
        )
        task = row["tasks"][0]
        current_pose = torch.tensor(target["pose"], dtype=torch.float32)
        goal_pose = torch.tensor(task["goal_pose"], dtype=torch.float32)
        current_quaternion = current_pose[3:7].unsqueeze(0)
        goal_quaternion = goal_pose[3:7].unsqueeze(0)
        point_offset = _quat_apply(current_quaternion, sampled_m)
        desired_translation = (goal_pose[:2] - current_pose[:2]).unsqueeze(0)
        yaw_error = _relative_yaw(current_quaternion, goal_quaternion)

        old_score = wrench_aware_contact_support_score(
            point_offset[None, :, :2],
            desired_translation,
            yaw_error,
            yaw_moment_weight=0.0,
            yaw_activation_rad=args.yaw_activation_rad,
        )[0]
        new_score = wrench_aware_contact_support_score(
            point_offset[None, :, :2],
            desired_translation,
            yaw_error,
            yaw_moment_weight=args.yaw_moment_weight,
            yaw_activation_rad=args.yaw_activation_rad,
        )[0]
        old_best = old_score.masked_fill(~safe_mask, -torch.inf).max()
        old_selected = safe_mask & (old_score >= old_best - args.side_band_m)

        direction = desired_translation[0] / torch.clamp(
            torch.linalg.vector_norm(desired_translation[0]), min=1.0e-6
        )
        trailing = torch.sum(point_offset[:, :2] * -direction, dim=-1)
        compatibility = signed_yaw_contact_moment_score(
            point_offset[None, :, :2],
            desired_translation,
            yaw_error,
            yaw_activation_rad=args.yaw_activation_rad,
        )[0]
        if args.selection_mode == "hierarchical":
            best_compatibility = compatibility.masked_fill(
                ~old_selected, -torch.inf
            ).max()
            yaw_selected = old_selected & (
                compatibility >= best_compatibility - args.yaw_side_band_m
            )
            new_selected = (
                yaw_selected
                if abs(float(yaw_error[0])) >= args.minimum_yaw_error_rad
                else old_selected
            )
        elif args.selection_mode == "yaw_first":
            # While signed yaw remains materially wrong, contact selection is
            # lexicographic in the physically required order: first retain
            # safe points with the best compatible moment arm.  Once yaw is
            # within tolerance, the observable current error switches the set
            # back to translation-only trailing support.  This is neither a
            # hidden phase nor a world-frame waypoint.
            yaw_selected, _ = yaw_compatible_safe_point_mask(
                compatibility[None],
                safe_mask[None],
                selection_mode="near_best",
                near_best_band_m=args.yaw_side_band_m,
                minimum_compatibility_m=args.yaw_compatibility_floor_m,
            )
            yaw_selected = yaw_selected[0]
            new_selected = (
                yaw_selected
                if abs(float(yaw_error[0])) >= args.minimum_yaw_error_rad
                else old_selected
            )
        elif args.selection_mode == "yaw_positive":
            # Keep the broad half of the safe handle whose signed moment has
            # the requested yaw sign.  A small positive floor excludes the
            # zero-moment centerline without collapsing the set to near-best
            # points.  The near-best set is only a finite fallback for an
            # unaudited asset with no point above the requested margin.
            yaw_selected, fallback = yaw_compatible_safe_point_mask(
                compatibility[None],
                safe_mask[None],
                selection_mode="positive_halfspace",
                near_best_band_m=args.yaw_side_band_m,
                minimum_compatibility_m=args.yaw_compatibility_floor_m,
            )
            yaw_selected = yaw_selected[0]
            yaw_positive_fallback_scenes += int(fallback[0])
            new_selected = (
                yaw_selected
                if abs(float(yaw_error[0])) >= args.minimum_yaw_error_rad
                else old_selected
            )
        elif args.selection_mode == "maximin":
            # Scale-free Pareto compromise: a candidate is only as good as
            # its weaker objective.  Unlike adding two metric-valued scores,
            # this cannot make translation and yaw gradients cancel merely
            # because their numeric scales differ.  The band retains a set
            # of near-optimal contacts instead of prescribing one waypoint.
            safe_trailing = trailing.masked_fill(~safe_mask, torch.inf)
            trailing_min = safe_trailing.min()
            trailing_max = trailing.masked_fill(~safe_mask, -torch.inf).max()
            safe_compatibility = compatibility.masked_fill(~safe_mask, torch.inf)
            compatibility_min = safe_compatibility.min()
            compatibility_max = compatibility.masked_fill(
                ~safe_mask, -torch.inf
            ).max()
            trailing_normalized = (trailing - trailing_min) / torch.clamp(
                trailing_max - trailing_min, min=1.0e-6
            )
            compatibility_normalized = (
                compatibility - compatibility_min
            ) / torch.clamp(
                compatibility_max - compatibility_min, min=1.0e-6
            )
            compromise = torch.minimum(
                trailing_normalized, compatibility_normalized
            )
            best_compromise = compromise.masked_fill(
                ~safe_mask, -torch.inf
            ).max()
            compromise_selected = safe_mask & (
                compromise
                >= best_compromise - float(args.normalized_score_band)
            )
            new_selected = (
                compromise_selected
                if abs(float(yaw_error[0])) >= args.minimum_yaw_error_rad
                else old_selected
            )
        else:
            new_best = new_score.masked_fill(~safe_mask, -torch.inf).max()
            new_selected = safe_mask & (new_score >= new_best - args.side_band_m)

        old_compatibility.append(float(compatibility[old_selected].mean()))
        new_compatibility.append(float(compatibility[new_selected].mean()))
        old_trailing.append(float(trailing[old_selected].mean()))
        new_trailing.append(float(trailing[new_selected].mean()))
        union = old_selected | new_selected
        selected_overlap.append(
            float((old_selected & new_selected).sum() / torch.clamp(union.sum(), min=1))
        )
        safe_count = torch.clamp(safe_mask.sum(), min=1)
        old_selected_fraction.append(float(old_selected.sum() / safe_count))
        new_selected_fraction.append(float(new_selected.sum() / safe_count))
        yaw_errors.append(float(yaw_error[0]))
        protected_old_count = int((protected_mask & old_selected).sum())
        protected_new_count = int((protected_mask & new_selected).sum())
        protected_selected_old += protected_old_count
        protected_selected_new += protected_new_count
        sign = "positive" if float(yaw_error[0]) >= 0.0 else "negative"
        sign_values = by_yaw_sign[sign]
        sign_values["yaw_errors"].append(float(yaw_error[0]))
        sign_values["old_compatibility"].append(old_compatibility[-1])
        sign_values["new_compatibility"].append(new_compatibility[-1])
        sign_values["old_trailing"].append(old_trailing[-1])
        sign_values["new_trailing"].append(new_trailing[-1])
        sign_values["selected_overlap"].append(selected_overlap[-1])
        sign_values["old_selected_fraction"].append(old_selected_fraction[-1])
        sign_values["new_selected_fraction"].append(new_selected_fraction[-1])
        sign_values["protected_selected_old"] += protected_old_count
        sign_values["protected_selected_new"] += protected_new_count

    old_compatibility_array = np.asarray(old_compatibility)
    new_compatibility_array = np.asarray(new_compatibility)
    report = {
        "asset_id": args.asset_id,
        "manifest": str(args.manifest.resolve()),
        "scene_count": len(rows),
        "sample_count": args.sample_count,
        "safe_sample_count": int(safe_mask.sum()),
        "protected_sample_count": int(protected_mask.sum()),
        "minimum_safe_score": args.minimum_safe_score,
        "side_band_m": args.side_band_m,
        "yaw_moment_weight": args.yaw_moment_weight,
        "yaw_activation_rad": args.yaw_activation_rad,
        "selection_mode": args.selection_mode,
        "yaw_side_band_m": args.yaw_side_band_m,
        "yaw_compatibility_floor_m": args.yaw_compatibility_floor_m,
        "normalized_score_band": args.normalized_score_band,
        "minimum_yaw_error_rad": args.minimum_yaw_error_rad,
        "yaw_error_rad": _summary(yaw_errors),
        "legacy_selected_signed_moment_m": _summary(old_compatibility),
        "wrench_selected_signed_moment_m": _summary(new_compatibility),
        "signed_moment_improvement_m": _summary(
            (new_compatibility_array - old_compatibility_array).tolist()
        ),
        "strictly_improved_scene_fraction": float(
            np.mean(new_compatibility_array > old_compatibility_array + 1.0e-6)
        ),
        "legacy_selected_trailing_support_m": _summary(old_trailing),
        "wrench_selected_trailing_support_m": _summary(new_trailing),
        "selected_subset_jaccard": _summary(selected_overlap),
        "legacy_selected_fraction_of_safe": _summary(old_selected_fraction),
        "wrench_selected_fraction_of_safe": _summary(new_selected_fraction),
        "yaw_positive_fallback_scenes": yaw_positive_fallback_scenes,
        "protected_points_selected": {
            "legacy": protected_selected_old,
            "wrench": protected_selected_new,
        },
        "by_yaw_sign": {
            sign: {
                "scene_count": len(values["yaw_errors"]),
                "yaw_error_rad": _summary(values["yaw_errors"]),
                "legacy_selected_signed_moment_m": _summary(
                    values["old_compatibility"]
                ),
                "wrench_selected_signed_moment_m": _summary(
                    values["new_compatibility"]
                ),
                "signed_moment_improvement_m": _summary(
                    (
                        np.asarray(values["new_compatibility"])
                        - np.asarray(values["old_compatibility"])
                    ).tolist()
                ),
                "legacy_selected_trailing_support_m": _summary(
                    values["old_trailing"]
                ),
                "wrench_selected_trailing_support_m": _summary(
                    values["new_trailing"]
                ),
                "selected_subset_jaccard": _summary(
                    values["selected_overlap"]
                ),
                "legacy_selected_fraction_of_safe": _summary(
                    values["old_selected_fraction"]
                ),
                "wrench_selected_fraction_of_safe": _summary(
                    values["new_selected_fraction"]
                ),
                "protected_points_selected": {
                    "legacy": values["protected_selected_old"],
                    "wrench": values["protected_selected_new"],
                },
            }
            for sign, values in by_yaw_sign.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
