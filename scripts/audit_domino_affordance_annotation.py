#!/usr/bin/env python3
"""Visualize sparse DOMINO anchors and the runtime point-wise semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from dapl.domino import (
    DominoDataPaths,
    default_affordance_radius,
    domino_point_affordance_features,
    load_domino_affordance_annotation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domino-root", type=Path, required=True)
    parser.add_argument("--asset-id", default="020_hammer:0")
    parser.add_argument("--sample-count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/affordance_annotation_audit/hammer_020_part_mask_v2"),
    )
    return parser.parse_args()


def _gaussian_anchor_features(
    points: torch.Tensor,
    annotation,
    radius_m: float,
) -> torch.Tensor:
    scale = torch.tensor(annotation.scale, dtype=points.dtype)
    scaled_points = points * scale

    def score(kind: str) -> torch.Tensor:
        anchors = annotation.anchor_positions(kind, dtype=points.dtype)
        minimum = torch.cdist(scaled_points.unsqueeze(0), anchors.unsqueeze(0)).amin(-1)[0]
        return torch.exp(-0.5 * (minimum / radius_m).square())

    safe = score("contact")
    protected = score("functional")
    return torch.stack((safe * (1.0 - protected), protected), dim=-1)


def _counts(features: torch.Tensor, threshold: float = 0.25) -> dict[str, int | float]:
    safe = features[:, 0] >= threshold
    protected = features[:, 1] >= threshold
    total = features.shape[0]
    return {
        "safe_count": int(safe.sum()),
        "protected_count": int(protected.sum()),
        "overlap_count": int((safe & protected).sum()),
        "neutral_count": int((~safe & ~protected).sum()),
        "safe_fraction": float(safe.float().mean()),
        "protected_fraction": float(protected.float().mean()),
        "neutral_fraction": float((~safe & ~protected).float().mean()),
        "total": total,
    }


def _scatter_semantics(
    axis,
    points_m: np.ndarray,
    features: torch.Tensor,
    title: str,
    threshold: float = 0.25,
) -> None:
    safe = features[:, 0].numpy() >= threshold
    protected = features[:, 1].numpy() >= threshold
    neutral = ~safe & ~protected
    for mask, color, label, size, alpha in (
        (neutral, "#9ca3af", "neutral", 1.0, 0.25),
        (safe, "#16a34a", "safe handle", 2.0, 0.75),
        (protected, "#dc2626", "protected tool end", 2.0, 0.75),
    ):
        axis.scatter(
            points_m[mask, 1],
            points_m[mask, 2],
            c=color,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("mesh Y (m)")
    axis.set_ylabel("mesh Z (m)")
    axis.set_aspect("equal")
    axis.grid(alpha=0.15)


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")

    paths = DominoDataPaths.resolve(args.domino_root)
    asset = paths.require_source_asset(args.asset_id)
    annotation = load_domino_affordance_annotation(asset)
    scene = trimesh.load(asset.collision_mesh, force="scene")
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    np.random.seed(args.seed)
    sampled, _ = trimesh.sample.sample_surface(mesh, args.sample_count)
    points = torch.from_numpy(sampled).to(torch.float32)

    radius_m = default_affordance_radius(annotation)
    old_features = _gaussian_anchor_features(points, annotation, radius_m)
    current_features = domino_point_affordance_features(points, annotation)
    scale = np.asarray(annotation.scale, dtype=np.float32)
    points_m = sampled * scale

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    _scatter_semantics(axes[0], points_m, old_features, "Previous sparse-anchor Gaussian proxy")
    _scatter_semantics(axes[1], points_m, current_features, "Current canonical part mask")
    contact = annotation.anchor_positions("contact").numpy()
    functional = annotation.anchor_positions("functional").numpy()
    for axis in axes:
        if len(contact):
            axis.scatter(
                contact[:, 1], contact[:, 2], marker="*", s=150, c="#14532d", label="contact anchor"
            )
        if len(functional):
            axis.scatter(
                functional[:, 1], functional[:, 2], marker="X", s=95, c="#7f1d1d", label="functional anchor"
            )
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(f"DOMINO {args.asset_id}: surface semantics audit", fontsize=15)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    image_path = args.output_prefix.with_suffix(".png")
    json_path = args.output_prefix.with_suffix(".json")
    figure.savefig(image_path, dpi=220)
    plt.close(figure)
    report = {
        "asset_id": args.asset_id,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "scale": annotation.scale,
        "threshold": 0.25,
        "default_gaussian_radius_m": radius_m,
        "previous_anchor_proxy": _counts(old_features),
        "current_part_mask": _counts(current_features),
        "part_rule_raw_mesh_frame": {
            "safe": "y <= 0.35",
            "protected": "y >= 0.62 or (y >= 0.45 and z <= -0.20)",
            "otherwise": "neutral",
        },
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(image_path)
    print(json_path)


if __name__ == "__main__":
    main()
