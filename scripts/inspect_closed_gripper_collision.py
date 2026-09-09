#!/usr/bin/env python3
"""Export the live C3 executor's composed collision assets without a planner.

Accepts run_c3_online_isaaclab_task.py arguments; --output is the audit JSON.
Mesh vertices are expressed in rigid-body frames using USD relative transforms,
never USD world poses (which can be stale when Fabric is enabled).
"""
from pathlib import Path
import json
import os
import traceback
import argparse
import sys
import hashlib

audit_parser = argparse.ArgumentParser(add_help=False)
audit_parser.add_argument("--executor-mode", choices=("task", "effort"), default="task")
audit_parser.add_argument("--reference-results", type=Path, nargs="+", default=[])
audit_parser.add_argument("--probe-contact-normals", action="store_true", help="advance one physics step at each reconstructed pose and export contact manifolds; diagnostic only")
audit_args, runner_args = audit_parser.parse_known_args()
sys.argv = [sys.argv[0], *runner_args]

if audit_args.executor_mode == "effort":
    import run_c3_online_isaaclab as runner
else:
    import run_c3_online_isaaclab_task as runner
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
from omni.physx import get_physx_cooking_interface
import numpy as np


def main():
    args = runner.args_cli
    if getattr(args, "use_push_anything_end_effector", False):
        raise ValueError("This audit requires the stock closed gripper")
    if audit_args.executor_mode == "effort" and not args.stock_closed_gripper:
        raise ValueError("Effort export requires the stock closed gripper")
    os.environ["DAPL_CLUTTER_MANIFEST"] = str(args.manifest.resolve())
    os.environ["DAPL_CLUTTER_ASSET_SOURCE"] = "domino"
    os.environ["DOMINO_ROOT"] = str(args.domino_root.resolve())
    os.environ["DOMINO_USD_ROOT"] = str(args.domino_usd_root.resolve())
    cfg = runner._configure_env()
    env = runner.gym.make(args.task, cfg=cfg)
    try:
        env.reset()
        base = env.unwrapped
        robot, target = base.scene["robot"], base.scene["target"]
        base.sim.forward()
        base.scene.update(0.0)
        stage = base.sim.stage
        cache = UsdGeom.XformCache()
        records = []
        for root in ("/World/envs/env_0/Robot", "/World/envs/env_0/Target"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                body = prim
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if not body:
                    raise ValueError(f"No rigid body for {prim.GetPath()}")
                # Remove only the body's rigid pose, retaining its asset scale.
                body_transform = Gf.Transform(cache.GetLocalToWorldTransform(body))
                body_rigid = Gf.Matrix4d(body_transform.GetRotation(), body_transform.GetTranslation())
                relative = cache.GetLocalToWorldTransform(prim) * body_rigid.GetInverse()
                record = {
                    "path": str(prim.GetPath()), "body_path": str(body.GetPath()),
                    "type": prim.GetTypeName(), "apis": list(prim.GetAppliedSchemas()),
                    "collision_enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                    "attributes": {},
                }
                for attr in prim.GetAttributes():
                    if attr.GetName().startswith(("physics:", "physx")):
                        value = attr.Get()
                        record["attributes"][attr.GetName()] = {
                            "value": (str(value) if isinstance(value, float) and not np.isfinite(value)
                                      else value if isinstance(value, (str, bool, int, float, type(None))) else str(value)),
                            "authored": attr.HasAuthoredValueOpinion(),
                        }
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    points = np.array([relative.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
                    record.update(vertices_body_m=points.tolist(),
                                  face_vertex_counts=list(mesh.GetFaceVertexCountsAttr().Get()),
                                  face_vertex_indices=list(mesh.GetFaceVertexIndicesAttr().Get()),
                                  bounds_body_m=[points.min(axis=0).tolist(), points.max(axis=0).tolist()])
                    if body.GetName() in ("panda_hand", "panda_leftfinger", "panda_rightfinger", "Target"):
                        def cooked_callback(status, convexes):
                            record["cooking_status"] = str(status)
                            record["cooked_convexes"] = [{
                                "vertices_body_m": [list(relative.Transform(Gf.Vec3d(v.x, v.y, v.z))) for v in convex.vertices],
                                "face_vertex_counts": [p.num_vertices for p in convex.polygons],
                                "face_vertex_indices": [int(i) for i in convex.indices],
                            } for convex in convexes]
                        get_physx_cooking_interface().request_convex_collision_representation(
                            stage_id=UsdUtils.StageCache.Get().GetId(stage).ToLongInt(),
                            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                            run_asynchronously=False, on_result=cooked_callback)
                elif prim.IsA(UsdGeom.Cube):
                    # Boxes are already exact convexes. Preserve each box;
                    # combining all finger boxes into one hull fills gaps.
                    import trimesh
                    cube = trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())] * 3)
                    points = np.array([relative.Transform(Gf.Vec3d(*p)) for p in cube.vertices])
                    convex = dict(vertices_body_m=points.tolist(),
                                  face_vertex_counts=[3] * len(cube.faces),
                                  face_vertex_indices=cube.faces.reshape(-1).tolist())
                    record.update(**convex, cooked_convexes=[convex],
                                  convex_source="exact USD Cube primitive; no mesh cooking needed")
                records.append(record)
        data = {
            "schema": "nonprehensile.live_collision_export.v1",
            "robot_model_contract": getattr(runner, "robot_model_contract", None),
            "robot_asset_path": cfg.scene.robot.spawn.asset_path,
            "robot_usd_path": str(Path(cfg.scene.robot.spawn.usd_dir) / cfg.scene.robot.spawn.usd_file_name),
            "physics_dt_s": cfg.sim.dt,
            "control_decimation": cfg.decimation,
            "env_origin_w": base.scene.env_origins[0].tolist(),
            "robot_root_pose_w": robot.data.root_state_w[0, :7].tolist(),
            "joint_names": robot.joint_names,
            "joint_position_rad_or_m": robot.data.joint_pos[0].tolist(),
            "joint_target_rad_or_m": robot.data.joint_pos_target[0].tolist(),
            "body_names": robot.body_names,
            "body_poses_w": robot.data.body_state_w[0, :, :7].tolist(),
            "target_pose_w": target.data.root_state_w[0, :7].tolist(),
            "robot_runtime_contact_offsets_m": robot.root_physx_view.get_contact_offsets().tolist(),
            "robot_runtime_rest_offsets_m": robot.root_physx_view.get_rest_offsets().tolist(),
            "target_runtime_contact_offsets_m": target.root_physx_view.get_contact_offsets().tolist(),
            "target_runtime_rest_offsets_m": target.root_physx_view.get_rest_offsets().tolist(),
            "sensors": {name: {"history_length": sensor.cfg.history_length,
                               "update_period_s": sensor.cfg.update_period,
                               "filters": sensor.cfg.filter_prim_paths_expr}
                        for name, sensor in base.scene.sensors.items()
                        if name.startswith("target_")},
            "colliders": records,
            "target_dynamics": {
                "inertia_frame": "body_axes_about_com",
                "mass_kg": float(target.root_physx_view.get_masses()[0].reshape(-1)[0]),
                "com_position_body_m": target.root_physx_view.get_coms()[0, :3].tolist(),
                "inertia_body_about_com_kg_m2": target.root_physx_view.get_inertias()[0].reshape(3, 3).T.tolist(),
            },
            "asset_contract": {
                "robot_urdf_sha256": hashlib.sha256(Path(cfg.scene.robot.spawn.asset_path).read_bytes()).hexdigest(),
            },
        }
        data["reconstructed_contact_poses"] = []
        detail_view = None
        if audit_args.probe_contact_normals:
            from isaacsim.core.simulation_manager import SimulationManager
            detail_view = SimulationManager.get_physics_sim_view().create_rigid_contact_view(
                "/World/envs/env_0/Target",
                filter_patterns=["/World/envs/env_0/Robot/panda_leftfinger", "/World/envs/env_0/Robot/panda_rightfinger"],
                max_contact_data_count=8192,
            )
        # Kinematic reconstruction only: historical finger displacement was not
        # logged, so these explicitly assume zero finger displacement.
        for result_path in audit_args.reference_results:
            reference = json.loads(result_path.read_text())
            for row in reference["trace"]:
                if not row.get("legal_safe_robot_contact"):
                    continue
                q = robot.data.default_joint_pos.clone()
                q[0, :7] = runner.torch.tensor(row["measured_joint_position_rad"], device=base.device)
                robot.write_joint_state_to_sim(q, runner.torch.zeros_like(q))
                base.sim.forward()
                base.scene.update(0.0)
                data["reconstructed_contact_poses"].append({
                    "result": str(result_path), "step": row["step"],
                    "assumed_finger_positions_m": [0.0, 0.0],
                    "body_poses_w": robot.data.body_state_w[0, :, :7].tolist(),
                })
                if detail_view is not None:
                    pose = runner.torch.tensor([row["target_position_m"] + row["target_quaternion_wxyz"]], device=base.device)
                    pose[:, :3] += base.scene.env_origins
                    target.write_root_pose_to_sim(pose)
                    target.write_root_velocity_to_sim(runner.torch.zeros((1, 6), device=base.device))
                    base.scene.write_data_to_sim()
                    base.sim.step(render=False)
                    base.scene.update(cfg.sim.dt)
                    forces, points, normals, distances, counts, starts = detail_view.get_contact_data(dt=cfg.sim.dt)
                    contacts = []
                    for filter_index, name in enumerate(("panda_leftfinger", "panda_rightfinger")):
                        start = int(starts.reshape(-1)[filter_index])
                        count = int(counts.reshape(-1)[filter_index])
                        for index in range(start, start + count):
                            contacts.append({"finger": name, "normal_force_n": float(forces[index].reshape(-1)[0]),
                                             "point_w_m": points[index].tolist(), "normal_w": normals[index].tolist(),
                                             "separation_m": float(distances[index].reshape(-1)[0])})
                    data["reconstructed_contact_poses"][-1]["one_step_contact_probe"] = {
                        "physics_dt_s": cfg.sim.dt, "contacts": contacts,
                        "limitation": "Zero initial velocity; this is not a reproduction of the historical force or trajectory",
                    }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        print(f"COLLISION_EXPORT_OK {args.output} colliders={len(records)}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()
        runner.simulation_app.close()


if __name__ == "__main__":
    main()
