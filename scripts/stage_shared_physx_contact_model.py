#!/usr/bin/env python3
"""Stage identical PhysX target/finger convexes into an isolated C3 runtime.

The source export must come from inspect_closed_gripper_collision.py. Isaac
keeps its existing physical assets. C3 consumes their actual cooked shapes,
with the target support rotation baked in and fingers in the hand orientation
frame about the existing task reference point. Semantic sampling stays intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

if __package__:
    from .analyze_closed_gripper_geometry import mesh_from_record, rotation
else:
    from analyze_closed_gripper_geometry import mesh_from_record, rotation


def materialize_directory(path):
    """Copy a runtime directory on write, leaving unrelated assets symlinked."""
    if path.is_symlink():
        source = path.resolve()
        path.unlink()
        path.mkdir()
        for child in source.iterdir():
            (path / child.name).symlink_to(child, target_is_directory=child.is_dir())
    elif not path.exists():
        path.mkdir(parents=True)


def target_inertial_in_support_frame(dynamics, support):
    """Rotate COM and central inertia from PhysX body axes into C3 axes.

    get_inertias() already expresses central inertia in rigid-body-prim axes;
    get_coms()' principal-axis quaternion must NOT be applied again. SDF's
    inertial pose carries the COM translation, so no parallel-axis term is
    added to the central inertia stored in the SDF.
    """
    if dynamics["inertia_frame"] != "body_axes_about_com":
        raise ValueError("Unknown target inertia frame")
    mass = float(dynamics["mass_kg"])
    com = np.asarray(dynamics["com_position_body_m"], dtype=float)
    inertia = np.asarray(dynamics["inertia_body_about_com_kg_m2"], dtype=float)
    if (not np.isfinite(mass) or mass <= 0 or com.shape != (3,)
            or inertia.shape != (3, 3) or not np.all(np.isfinite(com))
            or not np.all(np.isfinite(inertia))):
        raise ValueError("Invalid target inertial parameters")
    if not np.allclose(inertia, inertia.T, atol=1e-10, rtol=1e-6):
        raise ValueError("Target inertia must be symmetric")
    inertia = (inertia + inertia.T) / 2
    moments = np.linalg.eigvalsh(inertia)
    if moments[0] <= 0 or moments[2] > moments[0] + moments[1] + 1e-10:
        raise ValueError("Target principal moments are not physically valid")
    R = support.as_matrix()
    return {"mass_kg": mass, "com_position_body_m": (R @ com).tolist(),
            "inertia_body_about_com_kg_m2": (R @ inertia @ R.T).tolist(),
            "inertia_frame": "body_axes_about_com"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--stage-manifest", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--planner-clock-mode", choices=("wall", "simulation"), default="wall")
    p.add_argument("--spatial-safe-sampling", action="store_true")
    p.add_argument("--c3-admm-iterations", type=int, default=3)
    p.add_argument("--c3-qp-max-iterations", type=int, default=200)
    p.add_argument("--c3-nonnegative-contact-forces", action="store_true")
    p.add_argument("--c3-quaternion-cost-weight", type=float, default=1000.)
    p.add_argument("--c3-pd-rollout-interpolation", choices=("zoh", "foh"), default="zoh")
    p.add_argument("--c3-relinearized-pd-cost", action="store_true")
    p.add_argument("--c3-osc-matched-coarse-model", action="store_true")
    p.add_argument("--planner-finger-table-clearance", type=float)
    p.add_argument("--c3-pd-rollout-kp", type=float, nargs=3)
    p.add_argument("--c3-pd-rollout-kd", type=float, nargs=3)
    p.add_argument("--c3-progress-window-loops", type=int)
    p.add_argument("--c3-progress-cost-drop", type=float)
    p.add_argument("--c3-final-contact-scaling-mode", choices=("ee", "all"), default="ee")
    p.add_argument("--c3-end-on-qp-step", action="store_true")
    p.add_argument("--enforce-actor-workspace", action="store_true")
    p.add_argument("--execution-height-mode", choices=("planar", "optimized"), default="planar")
    args = p.parse_args()
    if args.runtime.exists():
        raise ValueError("Use a new runtime directory to preserve prior evidence")
    live = json.loads(args.export.read_text())
    stage = json.loads(args.stage_manifest.read_text())
    if stage["end_effector"]["mode"] != "closed-gripper-proxy":
        raise ValueError("Expected a closed-gripper stage contract")
    if np.max(np.abs(live["joint_position_rad_or_m"][-2:])) > 1e-6:
        raise ValueError("Export must have fully closed finger joints")
    source = args.upstream_root.resolve()
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True)
    for child in source.iterdir():
        (runtime / child.name).symlink_to(child, target_is_directory=child.is_dir())
    if (runtime / "solvers/osqp_options_default.yaml").is_file():
        materialize_directory(runtime / "solvers")
        solver_path = runtime / "solvers/osqp_options_default.yaml"
        solver_options = yaml.safe_load(solver_path.read_text())
        if args.c3_qp_max_iterations < 200:
            raise ValueError("QP diagnostic budget must not undercut the original 200 iterations")
        solver_options["int_options"]["max_iter"] = args.c3_qp_max_iterations
        solver_path.unlink()
        solver_path.write_text(yaml.safe_dump(solver_options, sort_keys=False))
    elif args.c3_qp_max_iterations != 200:
        raise ValueError("Missing native QP solver options")
    for relative in ("examples", "examples/sampling_c3", "examples/sampling_c3/urdf"):
        materialize_directory(runtime / relative)
    anything = runtime / "examples/sampling_c3/anything"
    anything_source = anything.resolve()
    anything.unlink()
    shutil.copytree(anything_source, anything)
    asset_name = "DOMINO_020_hammer_safe"
    asset_dir = runtime / "examples/sampling_c3/urdf" / asset_name
    asset_source = asset_dir.resolve()
    asset_dir.unlink()
    shutil.copytree(asset_source, asset_dir)
    contract_dir = runtime / "examples/sampling_c3/urdf/shared_physx_contact"
    if contract_dir.is_symlink():
        contract_dir.unlink()
    contract_dir.mkdir()
    support = Rotation.from_quat(np.roll(stage["support_quaternion_wxyz"], -1))
    source_dynamics = live.get("target_dynamics")
    target_inertial = (target_inertial_in_support_frame(source_dynamics, support)
                       if source_dynamics else None)
    source_support = live.get("target_support")
    support_centers = None
    if source_support:
        centers = np.asarray(source_support["support_sphere_centers_body_m"], dtype=float)
        support_radius = float(source_support["sphere_radius_m"])
        if centers.shape != (3, 3) or not np.all(np.isfinite(centers)) or not (0 < support_radius < .01):
            raise ValueError("Invalid measured support geometry")
        support_centers = support.apply(centers)
    target = next(c for c in live["colliders"] if c["body_path"].endswith("/Target"))
    if not target.get("cooked_convexes"):
        raise ValueError("Missing actual cooked target convexes")
    target_files = []
    for i, record in enumerate(target["cooked_convexes"]):
        mesh = mesh_from_record(record)
        mesh.vertices = support.apply(mesh.vertices)
        path = contract_dir / f"target_{i:02d}.obj"
        mesh.export(path, digits=12)
        target_files.append(path)
    hand_pose = live["body_poses_w"][live["body_names"].index("panda_hand")]
    tip_offset = stage["end_effector"]["reference_offset_m"]
    finger_files = []
    for name in ("panda_leftfinger", "panda_rightfinger"):
        colliders = [c for c in live["colliders"] if c["body_path"].endswith("/" + name)]
        parts = [part for c in colliders for part in c.get("cooked_convexes", [])]
        expected_parts = 4 if (live.get("robot_model_contract") or {}).get("robot") == "FR3" else 1
        if len(parts) != expected_parts:
            raise ValueError(f"Expected {expected_parts} convex components per stock finger")
        body_pose = live["body_poses_w"][live["body_names"].index(name)]
        for index, part in enumerate(parts):
            mesh = mesh_from_record(part)
            vertices_w = rotation(body_pose).apply(mesh.vertices) + body_pose[:3]
            mesh.vertices = rotation(hand_pose).inv().apply(vertices_w - hand_pose[:3]) - [0, 0, tip_offset]
            suffix = "" if expected_parts == 1 else f"_{index}"
            path = contract_dir / f"{name}{suffix}.obj"
            mesh.export(path, digits=12)
            finger_files.append(path)
    # Preserve upstream's three-sphere support approximation and contact count.
    # A measured support export replaces their bounding-box corner positions.
    ET.register_namespace("drake", "uri:drake")
    for suffix in ("_controller.sdf", ".sdf"):
        path = asset_dir / f"{asset_name}{suffix}"
        sdf_text = path.read_text()
        if 'xmlns:drake=' not in sdf_text:
            sdf_text = sdf_text.replace('<sdf ', '<sdf xmlns:drake="uri:drake" ', 1)
        tree = ET.ElementTree(ET.fromstring(sdf_text))
        link = tree.getroot().find("model/link")
        if target_inertial:
            inertial = link.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(link, "inertial")
            else:
                inertial.clear()
            ET.SubElement(inertial, "pose").text = " ".join(
                format(v, ".17g") for v in target_inertial["com_position_body_m"] + [0, 0, 0])
            ET.SubElement(inertial, "mass").text = format(target_inertial["mass_kg"], ".17g")
            matrix = target_inertial["inertia_body_about_com_kg_m2"]
            tensor = ET.SubElement(inertial, "inertia")
            for key, i, j in [("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2),
                              ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2)]:
                ET.SubElement(tensor, key).text = format(matrix[i][j], ".17g")
        old_contacts = list(link.findall("collision"))
        retained = old_contacts[-3:] if suffix == "_controller.sdf" else []
        if suffix == "_controller.sdf" and (len(old_contacts) < 4 or not all(c.find("geometry/sphere") is not None for c in retained)):
            raise ValueError("Unexpected support collision ordering")
        for contact in old_contacts:
            link.remove(contact)
        for i, mesh_path in enumerate(target_files):
            collision = ET.SubElement(link, "collision", name=f"shared_physx_{i:02d}")
            mesh = ET.SubElement(ET.SubElement(collision, "geometry"), "mesh")
            ET.SubElement(mesh, "uri").text = "../shared_physx_contact/" + mesh_path.name
            ET.SubElement(mesh, "{uri:drake}declare_convex")
            properties = ET.SubElement(collision, "{uri:drake}proximity_properties")
            ET.SubElement(properties, "{uri:drake}mu_static").text = "0.3"
            ET.SubElement(properties, "{uri:drake}mu_dynamic").text = "0.3"
        for i, contact in enumerate(retained):
            if support_centers is not None:
                pose = contact.find("pose")
                if pose is None:
                    pose = ET.SubElement(contact, "pose")
                pose.text = " ".join(format(v, ".17g") for v in [*support_centers[i], 0, 0, 0])
                contact.find("geometry/sphere/radius").text = format(support_radius, ".17g")
            link.append(contact)
        ET.indent(tree)
        tree.write(path, encoding="unicode", xml_declaration=True)
    config = anything / "parameters/sampling_c3_controller_params.yaml"
    params = yaml.safe_load(config.read_text())
    params["closed_gripper_collision_meshes"] = [str(f.relative_to(runtime)) for f in finger_files]
    params["use_simulation_time_for_plans"] = args.planner_clock_mode == "simulation"
    params["enforce_actor_workspace_bounds"] = args.enforce_actor_workspace
    params["enforce_nonnegative_contact_forces"] = args.c3_nonnegative_contact_forces
    params["use_foh_pd_rollout"] = args.c3_pd_rollout_interpolation == "foh"
    params["use_relinearized_pd_cost"] = getattr(args, "c3_relinearized_pd_cost", False)
    params["use_osc_matched_coarse_model"] = getattr(args, "c3_osc_matched_coarse_model", False)
    params["planner_finger_table_clearance"] = getattr(args, "planner_finger_table_clearance", None)
    if params["planner_finger_table_clearance"] is not None and not (0 <= params["planner_finger_table_clearance"] < 1):
        raise ValueError("Planner finger clearance must be finite and in [0,1) metres")
    if params["use_osc_matched_coarse_model"] and not params["use_relinearized_pd_cost"]:
        raise ValueError("OSC matched coarse model requires relinearized PD cost")
    if params["use_relinearized_pd_cost"] and not params["use_foh_pd_rollout"]:
        raise ValueError("Relinearized PD cost requires FOH references")
    # Drake rejects unvisited YAML keys even when their value is false/null.
    # Omit disabled optional additions so archived controller binaries can
    # consume their original configuration schema. Pop inherited stale keys
    # as well; merely skipping assignment would retain a previous trial's flag.
    for key in ("use_relinearized_pd_cost", "use_osc_matched_coarse_model"):
        if not params[key]:
            params.pop(key)
    if params["planner_finger_table_clearance"] is None:
        params.pop("planner_finger_table_clearance")
    config.write_text(yaml.safe_dump(params, sort_keys=False))
    if args.c3_progress_window_loops is not None or args.c3_progress_cost_drop is not None:
        progress_path = runtime / params["progress_params_file"]
        if not progress_path.resolve().is_relative_to(anything):
            raise ValueError("Progress override must stay inside the isolated demo runtime")
        progress = yaml.safe_load(progress_path.read_text())
        if progress["track_c3_progress_via"] != 3:
            raise ValueError("Progress overrides require native kConfigCostDrop mode")
        if args.c3_progress_window_loops is not None:
            if not 2 <= args.c3_progress_window_loops <= 1000:
                raise ValueError("Progress window must contain 2--1000 planner updates")
            progress["progress_enforced_over_n_loops"] = args.c3_progress_window_loops
        if args.c3_progress_cost_drop is not None:
            if not np.isfinite(args.c3_progress_cost_drop) or not 0 <= args.c3_progress_cost_drop < 1:
                raise ValueError("Progress cost drop must be finite in [0, 1)")
            progress["progress_enforced_cost_drop"] = args.c3_progress_cost_drop
        progress_path.write_text(yaml.safe_dump(progress, sort_keys=False))
    for filename in ("sampling_c3_options.yaml", "sampling_c3plus_options.yaml"):
        path = anything / "parameters" / filename
        options = yaml.safe_load(path.read_text())
        options["planar_demo"] = args.execution_height_mode == "planar"
        if not 1 <= args.c3_admm_iterations <= 30:
            raise ValueError("C3 ADMM iteration count must be between 1 and 30")
        options["admm_iter"] = args.c3_admm_iterations
        if not np.isfinite(args.c3_quaternion_cost_weight) or args.c3_quaternion_cost_weight <= 0:
            raise ValueError("Quaternion cost weight must be finite and positive")
        options["q_quaternion_dependent_weight"] = args.c3_quaternion_cost_weight
        for key, gains in (("Kp_for_ee_pd_rollout", args.c3_pd_rollout_kp),
                           ("Kd_for_ee_pd_rollout", args.c3_pd_rollout_kd)):
            if gains is not None:
                if not np.isfinite(gains).all() or min(gains) <= 0:
                    raise ValueError("PD rollout gains must be finite and positive")
                options[key] = gains
        if args.c3_final_contact_scaling_mode == "all":
            counts = options["resolve_contacts_to_lists"][options["num_contacts_index"]]
            planar = options["resolve_as_planar_contacts_list"]
            directions = options["num_friction_directions"]
            n_lambda = sum(n * (2 if flat else 2 * directions) for n, flat in zip(counts, planar))
            if options["contact_model"] != "anitescu":
                raise ValueError("All-contact final scaling currently requires Anitescu contact coordinates")
            options["final_augmented_cost_contact_indices"] = list(range(n_lambda))
        options["end_on_qp_step"] = args.c3_end_on_qp_step
        path.write_text(yaml.safe_dump(options, sort_keys=False))
    if args.spatial_safe_sampling:
        sampling_path = anything / "parameters/sampling_params.yaml"
        sampling_options = yaml.safe_load(sampling_path.read_text())
        sampling_options["gen_planar_samples"] = False
        sampling_path.write_text(yaml.safe_dump(sampling_options, sort_keys=False))
    friction = live.get("contact_friction")
    if friction:
        for key in ("ee_object", "object_ground"):
            if not np.isfinite(friction[key]) or friction[key] < 0:
                raise ValueError("Invalid measured pair friction")
        for filename in ("sampling_c3_options.yaml", "sampling_c3plus_options.yaml"):
            path = anything / "parameters" / filename
            options = yaml.safe_load(path.read_text())
            coefficients = options["mu_per_pair_type"]
            coefficients[1] = friction["ee_object"]
            coefficients[2] = friction["object_ground"]
            path.write_text(yaml.safe_dump(options, sort_keys=False))
    contract = {
        "schema": "nonprehensile.shared_physx_contact_model.v1",
        "source_export": str(args.export.resolve()),
        "source_export_sha256": hashlib.sha256(args.export.read_bytes()).hexdigest(),
        "robot_model_contract": live.get("robot_model_contract"),
        "asset_contract": live.get("asset_contract"),
        "source_target_dynamics": source_dynamics,
        "target_inertial_c3": target_inertial,
        "source_target_support": source_support,
        "contact_friction": friction,
        "support_sphere_centers_c3_m": support_centers.tolist() if support_centers is not None else None,
        "planner_clock_mode": args.planner_clock_mode,
        "enforce_actor_workspace_bounds": args.enforce_actor_workspace,
        "c3_end_on_qp_step": args.c3_end_on_qp_step,
        "c3_admm_iterations": args.c3_admm_iterations,
        "c3_qp_max_iterations": args.c3_qp_max_iterations,
        "c3_nonnegative_contact_forces": args.c3_nonnegative_contact_forces,
        "c3_quaternion_cost_weight": args.c3_quaternion_cost_weight,
        "c3_pd_rollout_interpolation": args.c3_pd_rollout_interpolation,
        "c3_pd_rollout_kp": args.c3_pd_rollout_kp,
        "c3_pd_rollout_kd": args.c3_pd_rollout_kd,
        "c3_progress_window_loops": args.c3_progress_window_loops,
        "c3_progress_cost_drop": args.c3_progress_cost_drop,
        "c3_final_contact_scaling_mode": args.c3_final_contact_scaling_mode,
        "spatial_safe_sampling": args.spatial_safe_sampling,
        "execution_height_mode": args.execution_height_mode,
        "physical_target_source": "existing Isaac visual-mesh PhysX cooked convexes",
        "target_convex_count": len(target_files), "finger_convex_count": len(finger_files),
        "orientation": "measured panda_hand FK, refreshed each replan and held over the prediction horizon",
        "reference_offset_m": tip_offset,
        "support_quaternion_wxyz": stage["support_quaternion_wxyz"],
        "files": {str(f.relative_to(runtime)): hashlib.sha256(f.read_bytes()).hexdigest() for f in target_files + finger_files},
        "unchanged": ["Isaac physical assets", "C3+ algorithm", "safe sampling mesh", "acceptance thresholds"],
    }
    (runtime / "shared_contact_model.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
