"""Isaac-side equivalent of the native Franka OSC semantic trajectory guard.

Uses staged unsafe meshes and measured object poses in the C3 frame. This
shield holds/retreats and removes wrench/velocity commands; it never creates
task-directed pushing actions. It does not replace the independent C1 audit.
"""

from pathlib import Path

import numpy as np
import trimesh
import yaml


def rotation_wxyz(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid semantic-guard object quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


class SemanticTrajectoryGuard:
    def __init__(self, meshes, stop_distance_m):
        if not meshes or not np.isfinite(stop_distance_m) or stop_distance_m <= 0:
            raise ValueError("Missing unsafe geometry or invalid semantic stop distance")
        self.meshes = meshes
        self.stop_distance_m = float(stop_distance_m)
        for mesh in meshes:
            if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
                raise ValueError("Invalid semantic unsafe mesh")

    @classmethod
    def from_runtime(cls, root, demo_name="anything"):
        root = Path(root)
        path = root / "examples/sampling_c3" / demo_name / "parameters/sampling_c3_controller_params.yaml"
        params = yaml.safe_load(path.read_text())
        meshes = [trimesh.load(root / relative, force="mesh", process=False)
                  for relative in params["unsafe_meshes"]]
        return cls(meshes, params.get("semantic_guard_stop_distance",
                   params.get("semantic_guard_clearance", .025) + .010))

    def apply(self, measured_position, commanded_position, object_states):
        """Return (position, held, minimum distance), matching native ShouldHold.

        The caller zeros feedforward wrench and desired velocity while held.
        Invalid geometry/state raises before any subsequent physics step.
        """
        points = np.asarray([measured_position, commanded_position], dtype=float)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("Invalid semantic-guard task position")
        if len(object_states) != len(self.meshes):
            raise ValueError("Semantic mesh/object-state count mismatch")
        distances = np.full(2, np.inf)
        closest_measured = None
        for mesh, state in zip(self.meshes, object_states):
            position = np.asarray(state.position_m, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("Invalid semantic-guard object position")
            rotation = rotation_wxyz(state.quaternion_wxyz)
            local = (points - position) @ rotation
            closest, distance, _ = trimesh.proximity.closest_point_naive(mesh, local)
            if not np.isfinite(distance).all():
                raise ValueError("Nonfinite semantic surface distance")
            if distance[0] < distances[0]:
                closest_measured = closest[0] @ rotation.T + position
            distances = np.minimum(distances, distance)
        held = bool(distances.min() <= self.stop_distance_m)
        target = points[1].copy()
        if held:
            target = points[0].copy()
            if distances[0] <= self.stop_distance_m:
                escape = points[0] - closest_measured
                norm = np.linalg.norm(escape)
                if norm > 1e-9:
                    target += (self.stop_distance_m + .003 - distances[0]) * escape / norm
        return target, held, float(distances.min())
