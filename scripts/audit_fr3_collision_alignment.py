"""Verify each FR3 finger component and safety samples against live USD geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from analyze_closed_gripper_geometry import mesh_from_record, rotation
from dapl.contact_planner.fr3_collision_geometry import collision_candidates


def audit(export_path, runtime, output):
    live = json.loads(export_path.read_text())
    contract_path = runtime / 'shared_contact_model.json'
    contract = json.loads(contract_path.read_text())
    assert contract['source_export_sha256'] == hashlib.sha256(export_path.read_bytes()).hexdigest()
    assert contract['finger_convex_count'] == 8
    assert abs(contract['reference_offset_m'] - .1034) < 1e-12
    hand = live['body_poses_w'][live['body_names'].index('panda_hand')]
    samples = []
    point_checks = []
    for name in ('panda_hand', 'panda_leftfinger', 'panda_rightfinger'):
        colliders = [c for c in live['colliders'] if c['body_path'].endswith('/'+name)]
        points = collision_candidates(live['robot_asset_path'], name)
        distance = np.full(len(points), np.inf)
        for c in colliders:
            if 'vertices_body_m' not in c:
                raise ValueError('Missing live collision vertices')
            mesh = mesh_from_record(c)
            distance = np.minimum(distance, trimesh.proximity.closest_point_naive(mesh, points)[1])
        point_checks.append(dict(body=name, sample_count=len(points),
                                 maximum_safety_sample_surface_distance_m=float(distance.max())))
        if name == 'panda_hand':
            continue
        assert len(colliders) == 4
        body = live['body_poses_w'][live['body_names'].index(name)]
        for i, c in enumerate(colliders):
            path = runtime/'examples/sampling_c3/urdf/shared_physx_contact'/f'{name}_{i}.obj'
            expected = np.array(c['vertices_body_m'])
            actual = np.asarray(trimesh.load(path, force='mesh', process=False).vertices)
            world = rotation(hand).apply(actual + [0, 0, .1034]) + hand[:3]
            actual_body = rotation(body).inv().apply(world - body[:3])
            error = max(cKDTree(expected).query(actual_body)[0].max(), cKDTree(actual_body).query(expected)[0].max())
            samples.append(dict(body=name, component=i, live_primitive=c['type'],
                symmetric_vertex_distance_m=float(error), staged_mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    passed = all(p['symmetric_vertex_distance_m'] < 2e-7 for p in samples) and all(p['maximum_safety_sample_surface_distance_m'] < 2e-7 for p in point_checks)
    result = dict(schema='nonprehensile.fr3_collision_alignment.v1', geometry_alignment_pass=passed,
        export_sha256=hashlib.sha256(export_path.read_bytes()).hexdigest(),
        shared_contact_model_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        components=samples, safety_cloud=point_checks,
        scope='Exact component geometry and safety sample frames. Contact-force/dynamics accuracy and task success are separate checks.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(geometry_alignment_pass=passed, component_count=len(samples),
        maximum_component_error_m=max(p['symmetric_vertex_distance_m'] for p in samples),
        safety_cloud=point_checks),indent=2))
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    raise SystemExit(0 if audit(args.export,args.runtime,args.output) else 2)
