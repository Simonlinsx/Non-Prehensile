"""Bounded FR3 execution adapter for the frozen M1 comparison only.

Uses an explicit 100/50 ms joint-position command, preserving phase durations.
Physics and independent acceptance run at 1 ms. No C3 solver is used here.
"""
import json
import math
import os
from pathlib import Path
import time

from fr3_execution_runtime.typed_constraints import read_typed_state, should_stop
from fr3_execution_runtime.audit_packet import read_packet
from fr3_execution_runtime.acceptance import acceptance_errors
import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat
from fr3_robot_model_contract import load_contract
from dapl.contact_planner.fr3_model_runtime import (
    spawn_fr3_with_rigid_finger_coupling, synchronize_arm_friction,
    synchronize_finger_limits,
)


from m1_control_timing import decimation, install_action_clock


def configure(cfg, control_rate_hz=10):
    model = load_contract(os.environ["PUSH_ANYTHING_ROBOT_MODEL_MANIFEST"])
    cfg.use_torch_compile = False
    cfg.enable_training_metrics = False
    cfg.enforce_joint_limits = False
    assert control_rate_hz == 20
    cfg.sim.dt = 1 / 240
    cfg.decimation = 12
    cfg.sim.render_interval = 24
    cfg.scene.robot.spawn.func = spawn_fr3_with_rigid_finger_coupling
    cfg.scene.robot.spawn.rigid_props.disable_gravity = False
    pos = cfg.scene.robot.init_state.pos
    cfg.scene.robot.init_state.pos = (pos[0], pos[1], pos[2] + .029)
    for i, q in enumerate((2.191, 1.1, -1.33, -2.22, 1.30, 2.02, .08), 1):
        cfg.scene.robot.init_state.joint_pos[f"panda_joint{i}"] = q
    cfg.scene.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.
    for group, indices in (("panda_shoulder", range(1, 5)), ("panda_forearm", range(5, 8))):
        act = cfg.scene.robot.actuators[group]
        # Preserve M1's position-drive gains; these are explicitly a controller
        # difference from C3's zero-stiffness torque interface.
        act.friction = 0.
        act.armature = {f"panda_joint{i}": model["armature_kg_m2"][i-1] for i in indices}
        act.effort_limit_sim = {f"panda_joint{i}": model["joint_limits"][f"panda_joint{i}"]["effort"] for i in indices}
        act.velocity_limit_sim = {f"panda_joint{i}": model["joint_limits"][f"panda_joint{i}"]["velocity"] for i in indices}
    hand = cfg.scene.robot.actuators["panda_hand"]
    for attr, field in (("effort_limit_sim", "effort"), ("velocity_limit_sim", "velocity")):
        setattr(hand, attr, {f"panda_finger_joint{i}": model["joint_limits"][f"panda_finger_joint{i}"][field] for i in (1, 2)})
    for name in ("affordance_scene", "hand_state", "previous_action", "rel_goal", "target_twist"):
        if hasattr(cfg.observations.policy, name):
            setattr(cfg.observations.policy, name, None)
    cfg.observations.critic = None
    for name in ("task_success", "safe_region_distance", "safe_region_progress", "first_safe_region_contact", "safe_contact_planar_progress", "safe_contact_height_progress", "safe_contact_rotation_progress", "near_goal_target_motion", "action_magnitude", "action_rate", "forbidden_region_contact", "forbidden_region_clearance"):
        if hasattr(cfg.rewards, name):
            setattr(cfg.rewards, name, None)


def install(base, ik):
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile import mdp
    model = load_contract(os.environ["PUSH_ANYTHING_ROBOT_MODEL_MANIFEST"])
    out = Path(os.environ["M1_COMPARISON_OUTPUT"])
    robot = base.scene["robot"]
    arm = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    fingers = [robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
    hand = robot.body_names.index("panda_hand")
    friction = synchronize_arm_friction(robot.root_physx_view, arm)
    limits = synchronize_finger_limits(robot, model)
    initial_q = robot.data.joint_pos.new_tensor([[2.191, 1.1, -1.33, -2.22, 1.30, 2.02, .08]])
    robot.write_joint_state_to_sim(initial_q, torch.zeros_like(initial_q), joint_ids=arm)
    base.sim.forward()
    base.scene.update(0.)
    fk = ik.forward(robot.data.joint_pos[0, arm].cpu().numpy())
    root_rotation = matrix_from_quat(robot.data.root_quat_w[0]).cpu().numpy()
    expected_pos = root_rotation @ fk.translation + robot.data.root_pos_w[0].cpu().numpy()
    hand_rotation = matrix_from_quat(robot.data.body_quat_w[0, hand]).cpu().numpy()
    actual_pos = robot.data.body_pos_w[0, hand].cpu().numpy() + hand_rotation @ np.array([0., 0., .1034])
    fk_error = float(np.linalg.norm(expected_pos - actual_pos))
    rotation_error = float(np.linalg.norm(root_rotation @ fk.rotation - hand_rotation))
    audit = {"fk_position_error_m": fk_error, "fk_rotation_matrix_error": rotation_error,
             "model_sha256": model["manifest_sha256"], "friction": friction, "finger_limits": limits,
             "physics_dt_s": base.physics_dt, "control_dt_s": base.step_dt,
             "gravity_enabled": True, "controller": "M1 latched joint position, original PD gains",
             "stiffness": robot.root_physx_view.get_dof_stiffnesses().cpu().tolist(),
             "damping": robot.root_physx_view.get_dof_dampings().cpu().tolist(),
             "initial_q": robot.data.joint_pos[0].cpu().tolist(),
             "initial_base_pos": robot.data.root_pos_w[0].cpu().tolist(),
             "body_names": list(robot.body_names),
             "body_masses_kg": robot.root_physx_view.get_masses()[0].cpu().tolist(),
             "body_com_pose_xyzw": robot.root_physx_view.get_coms()[0].cpu().tolist(),
             "body_inertias_kg_m2": robot.root_physx_view.get_inertias()[0].cpu().tolist(),
             "target_mass_kg": base.scene["target"].root_physx_view.get_masses()[0].cpu().tolist(),
             "target_material": base.scene["target"].root_physx_view.get_material_properties()[0].cpu().tolist()}
    (out / "model_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if fk_error > 1e-5 or rotation_error > 1e-5:
        raise RuntimeError("FR3 IK does not match the live robot FK")
    goal = base.command_manager.get_command("target_object_pose")[0].detach().clone()
    goal_cpu = goal.cpu().tolist()
    target = base.scene["target"]
    initial = target.data.root_pos_w[0].detach().clone()
    initial_pose = torch.cat((initial - base.scene.env_origins[0], target.data.root_quat_w[0])).cpu().tolist()
    rows = (out / "physics.jsonl").open("x")
    typed_scope = os.environ.get("M1_TYPED_STOP_SCOPE", "observe")
    should_stop(False, False, typed_scope)
    state = {"c2": False, "c3": False, "steps": 0, "dwell": 0, "max_dwell": 0, "contact_steps_002": 0,
             "contact_steps_05": 0, "safe_05": False, "safe_002": False,
             "c1": False, "max_finger_m": 0., "max_displacement_m": 0., "first_contact_s": None}
    action_rows = install_action_clock(base, state, out)
    base._c3_semantic_distance_cuda_graph_enabled = True
    start = time.monotonic()
    original_update = base.scene.update
    original_write = base.scene.write_data_to_sim
    def write_data(*args, **kwargs):
        # The original M1 task disabled arm gravity. Under the matched FR3
        # plant, supply the same nominal gravity compensation available to
        # native OSC, without changing the M1 position-reference controller.
        gravity = robot.root_physx_view.get_gravity_compensation_forces()[:, arm].to(base.device)
        robot.set_joint_effort_target(gravity, joint_ids=arm)
        robot.set_joint_position_target(torch.zeros((1, 2), device=base.device), joint_ids=fingers)
        return original_write(*args, **kwargs)
    base.scene.write_data_to_sim = write_data
    max_steps = round(float(os.environ.get("M1_COMPARISON_SIM_S", "30")) / base.physics_dt)

    def finish(reason):
        rows.flush()
        rows.close()
        action_rows.close()
        result = {**state, "stop_reason": reason, "sim_s": state["steps"] * base.physics_dt,
                  "main_wall_s": time.monotonic() - start, "initial_pose_wxyz": initial_pose,
                  "goal_pose_wxyz": goal.cpu().tolist(), "last": state.get("last"),
                  "constrained_success": reason == "strict_pose_dwell_success" and not state["c1"] and not state["c2"] and not state["c3"] and state["max_finger_m"] < .001,
                  "typed_stop_scope": typed_scope, "clutter_force_threshold_n": .02,
                  "scope_note": "All constraints observed; stop scope does not change the C1 persistent planner. Stop on collision is not anticipatory avoidance.",
                  "thresholds": {"xy_m": .02, "height_m": .01, "so3_rad": .105, "dwell_s": .5,
                                 "contact_force_n": .5, "semantic_distance_m": .01},
                  "audit_cadence": "every physics step, before the next substep",
                  "controller_contact_gate_n": .02}
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("M1_FR3_COMPARISON", json.dumps(result), flush=True)
        raise SystemExit(0)

    def update(dt, *args, **kwargs):
        original_update(dt, *args, **kwargs)
        if dt <= 0:
            return
        state["steps"] += 1
        # M1 has multiple physics substeps per command. Invalidate control-step caches so
        # all semantic checks consume the just-updated physical state.
        for name in ("_domino_affordance_geometry_cache", "_domino_robot_target_geometry_cache"):
            if hasattr(base, name):
                delattr(base, name)
        base._domino_affordance_state_cache = {}
        contact = mdp.domino_affordance_contact_state(base, contact_distance_m=.01,
            evaluate_protected=False, evaluate_robot_obstacle=False,
            physical_contact_force_threshold_n=.5,
            robot_target_sensor_name="target_robot_contacts", hand_target_sensor_name="target_hand_contacts")
        packet = read_packet(contact,
            base.scene.sensors["target_hand_contacts"].data.force_matrix_w[0],
            target.data.root_pos_w[0], target.data.root_quat_w[0],
            base.scene.env_origins[0], goal, initial, robot.data.joint_pos[0, fingers])
        force, safe, c1 = packet["force"], packet["safe"], packet["c1"]
        safe_low = force > .02 and packet["legal_low"]
        state["safe_05"] |= safe
        state["safe_002"] |= safe_low
        state["c1"] |= c1
        state["contact_steps_002"] += int(force > .02)
        state["contact_steps_05"] += int(force > .5)
        if force > .02 and state["first_contact_s"] is None:
            state["first_contact_s"] = state["steps"] * base.physics_dt
        xy, height, rotation = acceptance_errors(packet, goal_cpu)
        strict = xy < .02 and height < .01 and rotation < .105
        state["dwell"] = state["dwell"] + 1 if strict else 0
        state["max_dwell"] = max(state["max_dwell"], state["dwell"])
        state["max_finger_m"] = max(state["max_finger_m"], packet["max_finger"])
        state["max_displacement_m"] = max(state["max_displacement_m"], packet["displacement"])
        typed = read_typed_state(base)
        typed_row = {k: (bool(v[0].item()) if k in ("c2", "c3") else float(v[0].item())) for k, v in typed.items()}
        if not math.isfinite(typed_row["protected_clearance_m"]):
            typed_row["protected_clearance_m"] = None
        state["c2"] |= typed_row["c2"]
        state["c3"] |= typed_row["c3"]
        row = {**typed_row, "step": state["steps"], "xy_m": xy, "height_m": height, "rotation_rad": rotation,
               "pose_wxyz": packet["pose"], "force_n": force, "safe": safe,
               "c1": c1, "finger_m": packet["fingers"],
               "legacy_pose_errors": [packet["xy"], packet["height"], packet["rotation"]]}
        state["last"] = row
        rows.write(json.dumps(row) + "\n")
        if state["steps"] % 240 == 0:
            rows.flush()
            print("M1_FR3_PROGRESS", state["steps"], xy, rotation, state["contact_steps_002"], flush=True)
        if state["c1"]:
            finish("c1_violation")
        if state["max_finger_m"] >= .001:
            finish("gripper_not_closed")
        if should_stop(state["c2"], state["c3"], typed_scope):
            finish("typed_constraint_violation")
        if state["dwell"] >= 120 and state["safe_05"]:
            finish("strict_pose_dwell_success")
        if state["steps"] >= max_steps:
            finish("simulation_budget")

    base.scene.update = update
    return finish
