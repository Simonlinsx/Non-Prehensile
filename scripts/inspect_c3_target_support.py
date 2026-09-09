#!/usr/bin/env python3
"""Export resting PhysX target-ground contact points and material parameters."""
import json
import os
import traceback
import itertools

import run_c3_online_isaaclab_task as runner
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics


def main():
    args = runner.args_cli
    os.environ["DAPL_CLUTTER_MANIFEST"] = str(args.manifest.resolve())
    os.environ["DAPL_CLUTTER_ASSET_SOURCE"] = "domino"
    os.environ["DOMINO_ROOT"] = str(args.domino_root.resolve())
    os.environ["DOMINO_USD_ROOT"] = str(args.domino_usd_root.resolve())
    env = runner.gym.make(args.task, cfg=runner._configure_env())
    try:
        base = env.unwrapped
        env.reset()
        target, robot = base.scene['target'], base.scene['robot']
        contacts = []
        from isaacsim.core.simulation_manager import SimulationManager
        view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
            '/World/envs/env_0/Target',
            filter_patterns=['/World/ground/geometry/mesh'], max_contact_data_count=8192)
        # Hold the remote arm kinematically; settle only the target under its
        # existing gravity/material settings. No pushing controller is used.
        q = robot.data.joint_pos.clone()
        for _ in range(240):
            robot.write_joint_state_to_sim(q, runner.torch.zeros_like(q))
            base.scene.write_data_to_sim()
            base.sim.step(render=False)
            base.scene.update(base.sim.get_physics_dt())
            forces, points, normals, distances, counts, starts = view.get_contact_data(dt=base.sim.get_physics_dt())
            batch = []
            first, count = int(starts.reshape(-1)[0]), int(counts.reshape(-1)[0])
            for i in range(first, first + count):
                batch.append({'point_w_m': points[i].tolist(), 'normal_w': normals[i].tolist(),
                              'normal_force_n': float(forces[i].reshape(-1)[0]),
                              'separation_m': float(distances[i].reshape(-1)[0])})
            if batch:
                contacts[:] = batch
        if len(contacts) < 3:
            raise ValueError(f'Insufficient resting contacts: {len(contacts)}')
        positions = np.asarray([c['point_w_m'] for c in contacts])
        com = target.data.root_com_pos_w[0].cpu().numpy()
        active = [i for i, c in enumerate(contacts) if c["normal_force_n"] > 1e-5]
        hull = ConvexHull(positions[active, :2])
        best = None
        for ids in itertools.combinations(sorted(active[i] for i in hull.vertices), 3):
            xy = positions[list(ids), :2]
            matrix = np.vstack([xy.T, np.ones(3)])
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            barycentric = np.linalg.solve(matrix, np.r_[com[:2], 1])
            area = abs(np.linalg.det(matrix)) / 2
            if min(barycentric) >= -1e-6 and (best is None or area > best[0]):
                best = (area, ids, barycentric)
        if best is None:
            raise ValueError('No support triangle contains the resting COM')
        pose = np.r_[target.data.root_pos_w[0].cpu().numpy(), target.data.root_quat_w[0].cpu().numpy()]
        R = Rotation.from_quat(np.roll(pose[3:], -1))
        radius = .001
        centers_body = R.inv().apply(positions[list(best[1])] + [0, 0, radius] - pose[:3])
        materials = []
        for prim in Usd.PrimRange(base.sim.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                materials.append({'path': str(prim.GetPath()), 'attributes': {
                    attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                    if any(k in attr.GetName().lower() for k in ('friction', 'restitution'))}})
        report = {
            'schema': 'nonprehensile.target_support_export.v1', 'target_pose_w': pose.tolist(),
            'target_com_w_m': com.tolist(), 'contacts': contacts,
            'active_force_threshold_n': 1e-5, 'selected_contact_indices': list(best[1]), 'selected_triangle_area_m2': best[0],
            'com_barycentric': best[2].tolist(), 'sphere_radius_m': radius,
            'support_sphere_centers_body_m': centers_body.tolist(),
            'target_material_properties': target.root_physx_view.get_material_properties().tolist(),
            'robot_material_properties': robot.root_physx_view.get_material_properties().tolist(),
            'usd_materials': materials,
            'scope': 'Three-point approximation of measured resting support; not goal-dependent contact selection.',
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('TARGET_SUPPORT_EXPORT_OK', len(contacts), 'contacts', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        runner.args_cli.output.write_text(json.dumps({'error': str(exc)}) + '\n')
        traceback.print_exc()
    finally:
        runner.simulation_app.close()
