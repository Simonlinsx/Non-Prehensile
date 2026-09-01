#!/usr/bin/env python3
"""Render an audited Push Anything C1 trajectory as a semantic top-down MP4."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import trimesh


COLORS = {
    "safe": "#20b26b",
    "protected": "#e34850",
    "neutral": "#6f7682",
    "goal": "#20c7df",
    "ee": "#276ef1",
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=repo_root / "data/push_anything_semantics/020_hammer_0",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--speedup", type=float, default=4.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--ee-radius-m", type=float, default=0.0195)
    parser.add_argument("--freeze-final-s", type=float, default=1.0)
    return parser.parse_args()


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def transform_points(
    points: np.ndarray, quaternion: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    rotation = quaternion_wxyz_to_matrix(quaternion)
    return points @ rotation.T + translation[None, :]


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "c1_semantic_audit.csv"
    if not path.is_file():
        raise FileNotFoundError(f"audited trajectory is missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty audited trajectory: {path}")
    return rows


def select_frame_indices(
    wall_times: np.ndarray, fps: float, speedup: float
) -> np.ndarray:
    if fps <= 0.0 or speedup <= 0.0:
        raise ValueError("fps and speedup must be positive")
    query = np.arange(wall_times[0], wall_times[-1] + 1e-9, speedup / fps)
    indices = np.searchsorted(wall_times, query, side="left")
    return np.unique(np.clip(indices, 0, len(wall_times) - 1))


def mesh_centroids(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.size == 0:
        raise ValueError(f"invalid semantic mesh: {path}")
    return np.asarray(mesh.triangles_center, dtype=np.float64)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.ee_radius_m <= 0.0:
        raise ValueError("image dimensions and EE radius must be positive")
    run_dir = args.run_dir.resolve()
    semantic_dir = args.semantic_dir.resolve()
    rows = load_rows(run_dir)
    stage = json.loads((run_dir / "stage_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "scene_result.json").read_text(encoding="utf-8"))
    semantic_manifest = json.loads(
        (semantic_dir / "semantic_mesh_manifest.json").read_text(encoding="utf-8")
    )

    wall_times = np.asarray([float(row["wall_time_s"]) for row in rows])
    indices = select_frame_indices(wall_times, args.fps, args.speedup)
    object_positions = np.asarray(
        [
            [float(row["object_x_m"]), float(row["object_y_m"]), float(row["object_z_m"])]
            for row in rows
        ]
    )
    object_quaternions = np.asarray(
        [
            [
                float(row["object_qw"]),
                float(row["object_qx"]),
                float(row["object_qy"]),
                float(row["object_qz"]),
            ]
            for row in rows
        ]
    )
    ee_positions = np.asarray(
        [[float(row["ee_x_m"]), float(row["ee_y_m"]), float(row["ee_z_m"])] for row in rows]
    )
    position_errors = np.asarray([float(row["position_error_m"]) for row in rows])
    rotation_errors = np.asarray([float(row["rotation_error_rad"]) for row in rows])
    legal_contacts = np.asarray([int(row["legal_safe_contact"]) != 0 for row in rows])
    violations = np.asarray([int(row["c1_violation"]) != 0 for row in rows])

    centroids = {
        name: mesh_centroids(semantic_dir / semantic_manifest["meshes"][name])
        for name in ("safe", "protected", "neutral")
    }
    full_centroids = mesh_centroids(semantic_dir / semantic_manifest["physical_mesh"])
    goal_yaw = math.radians(float(stage["goal_yaw_deg"]))
    goal_quaternion = np.asarray(
        [math.cos(goal_yaw / 2.0), 0.0, 0.0, math.sin(goal_yaw / 2.0)]
    )
    goal_position = np.asarray(
        [stage["goal_xy_m"][0], stage["goal_xy_m"][1], stage["root_height_m"]]
    )
    goal_points = transform_points(full_centroids, goal_quaternion, goal_position)

    all_xy = np.vstack(
        [
            object_positions[:, :2],
            ee_positions[:, :2],
            goal_points[:, :2],
            np.asarray(stage["initial_xy_m"], dtype=np.float64)[None, :],
        ]
    )
    lower = all_xy.min(axis=0) - 0.055
    upper = all_xy.max(axis=0) + 0.055
    center = 0.5 * (lower + upper)
    span = max(upper[0] - lower[0], upper[1] - lower[1], 0.24)
    x_limits = (center[0] - span / 2.0, center[0] + span / 2.0)
    y_limits = (center[1] - span / 2.0, center[1] + span / 2.0)

    dpi = 100
    figure, axis = plt.subplots(figsize=(args.width / dpi, args.height / dpi), dpi=dpi)
    figure.patch.set_facecolor("#d9ecf6")
    axis.set_facecolor("#d9ecf6")
    axis.set_aspect("equal")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.grid(color="white", linewidth=0.8, alpha=0.8)
    axis.set_xlabel("world X (m)")
    axis.set_ylabel("world Y (m)")

    goal_artist = axis.scatter(
        goal_points[:, 0], goal_points[:, 1], s=4, c=COLORS["goal"], alpha=0.22,
        linewidths=0, label="transparent goal pose", zorder=1,
    )
    semantic_artists = {
        name: axis.scatter([], [], s=7 if name != "neutral" else 5,
                           c=COLORS[name], alpha=0.86, linewidths=0, zorder=3)
        for name in ("protected", "neutral", "safe")
    }
    trail_artist, = axis.plot([], [], color=COLORS["ee"], alpha=0.35, linewidth=1.5, zorder=2)
    ee_circle = Circle((0.0, 0.0), args.ee_radius_m, facecolor="none",
                       edgecolor=COLORS["ee"], linewidth=2.5, zorder=5)
    axis.add_patch(ee_circle)
    initial_xy = np.asarray(stage["initial_xy_m"], dtype=np.float64)
    goal_arrow = FancyArrowPatch(
        initial_xy, np.asarray(stage["goal_xy_m"], dtype=np.float64),
        arrowstyle="-|>", mutation_scale=14, color=COLORS["goal"],
        linewidth=2.0, alpha=0.8, zorder=2,
    )
    axis.add_patch(goal_arrow)
    info_artist = axis.text(
        0.015, 0.985, "", transform=axis.transAxes, ha="left", va="top",
        family="monospace", fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.88,
              "edgecolor": "#9ab3c4"}, zorder=10,
    )
    outcome = "SUCCESS" if result["accepted"] else "FAILURE"
    axis.set_title(
        f"{result['scene_id']}  {outcome}  |  "
        f"direction {result['goal_direction_deg']:+.1f} deg  |  "
        f"distance {100 * result['goal_distance_m']:.0f} cm  |  "
        f"yaw {result['goal_yaw_deg']:+.0f} deg"
    )
    legend_handles = [
        Line2D([], [], marker="o", linestyle="", color=COLORS["safe"], label="safe region"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["protected"], label="protected region"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["neutral"], label="neutral boundary"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
               color=COLORS["ee"], label="modeled EE sphere"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["goal"], alpha=0.4,
               label="goal ghost"),
    ]
    axis.legend(handles=legend_handles, loc="lower left", fontsize=8, framealpha=0.9)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer: {args.output}")

    def render_frame(index: int) -> None:
        for name, points in centroids.items():
            world = transform_points(
                points, object_quaternions[index], object_positions[index]
            )
            semantic_artists[name].set_offsets(world[:, :2])
        ee_circle.center = tuple(ee_positions[index, :2])
        if violations[index]:
            ee_circle.set_edgecolor("#b00020")
            contact_label = "VIOLATION"
        elif legal_contacts[index]:
            ee_circle.set_edgecolor("#08a045")
            contact_label = "LEGAL SAFE CONTACT"
        else:
            ee_circle.set_edgecolor(COLORS["ee"])
            contact_label = "NO CONTACT"
        trail_artist.set_data(ee_positions[: index + 1, 0], ee_positions[: index + 1, 1])
        info_artist.set_text(
            f"time       {wall_times[index]:6.1f} s\n"
            f"pos error  {1000 * position_errors[index]:6.1f} mm\n"
            f"SO(3) err  {math.degrees(rotation_errors[index]):6.1f} deg\n"
            f"contact    {contact_label}"
        )
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        writer.write(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))

    try:
        for index in indices:
            render_frame(int(index))
        for _ in range(max(0, int(round(args.freeze_final_s * args.fps)))):
            render_frame(int(indices[-1]))
    finally:
        writer.release()
        plt.close(figure)

    metadata = {
        "schema": "nonprehensile.push_anything_c1_video.v1",
        "source_run": str(run_dir),
        "output": str(args.output.resolve()),
        "fps": args.fps,
        "speedup": args.speedup,
        "frames": int(len(indices) + round(args.freeze_final_s * args.fps)),
        "representation": "audited semantic top-down planner replay",
        "contains_full_robot": False,
        "accepted": result["accepted"],
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
