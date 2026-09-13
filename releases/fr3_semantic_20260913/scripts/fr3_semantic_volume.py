"""Explicit 3-D point-contact cost for the procedural convex-component assets.

This is a visualization/query layer, not the running planar placement planner.
Distance is to the union of closed collision volumes. Never assign a finite
penalty to a forbidden point. Component interiors use convex halfspaces;
outside distances use exact point/triangle closest points (trimesh).
"""
import numpy as np
from scipy.spatial import ConvexHull
import trimesh


def distance_to_volume(points, mesh):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite Nx3 points')
    inside = np.zeros(len(points), dtype=bool)
    components = mesh.split(only_watertight=False)
    if not components:
        raise ValueError('Empty geometry')
    for part in components:
        if not part.is_watertight or not part.is_convex:
            raise ValueError('This asset query requires closed convex components')
        equations = ConvexHull(part.vertices).equations
        for start in range(0, len(points), 2048):
            p = points[start:start+2048]
            inside[start:start+len(p)] |= np.all(p @ equations[:,:3].T + equations[:,3] <= 1e-10, axis=1)
    distances = np.zeros(len(points))
    outside = np.flatnonzero(~inside)
    for start in range(0, len(outside), 512):
        indices = outside[start:start+512]
        _, distances[indices], _ = trimesh.proximity.closest_point(mesh, points[indices])
    return distances


def point_contact_cost(points, objects, margin=.012):
    if margin <= 0:
        raise ValueError('Positive margin required')
    soft = np.zeros(len(points))
    forbidden = np.zeros(len(points), dtype=bool)
    distances = {}
    for rule, mesh in objects:
        # Whole-object exclusion fills protected cavities conservatively:
        # a liquid cup's empty collision-mesh interior must not appear safe.
        query_mesh = mesh.convex_hull if rule['mode'] == 'forbidden' else mesh
        d = distance_to_volume(points, query_mesh)
        distances[rule['id']] = d
        if rule['mode'] == 'forbidden':
            forbidden |= d <= margin
        elif rule['mode'] == 'acceptable':
            soft += rule['weight'] * np.clip(1-d/margin, 0, 1)
        else:
            raise ValueError(rule['mode'])
    total = soft.copy()
    total[forbidden] = np.inf
    return dict(soft_cost=soft, forbidden=forbidden, total_cost=total, distance_m=distances)
