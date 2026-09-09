"""Preserve already measured CPU evidence when the online executor fails.

This module never queries a possibly broken simulator or reconstructs missing
measurements. Partial results remain ineligible for acceptance.
"""


def execution_failure_evidence(state, configuration):
    evidence = {
        "stopped_reason": "executor_exception",
        "execution_evidence_complete": False,
        "online_closed_loop_success": False,
        "s2_single_scene_pass": False,
        "acceptance_eligible": False,
        "physics_step_in_progress_at_error": state.get("physics_step_in_progress", False),
        "executed_steps_definition": "Number of env.step calls that returned successfully",
    }
    fields = {
        "executed_steps": "executed_steps",
        "control_period_s": "control_period_s",
        "trace": "trace",
        "command_counts": "counters",
        "contact_audited_steps": "contact_audited_steps",
        "safe_robot_contact_ever": "safe_contact_ever",
        "forbidden_robot_contact_ever": "forbidden_contact_ever",
        "first_safe_contact_step": "first_safe_contact_step",
        "first_forbidden_contact_step": "first_forbidden_contact_step",
        "peak_robot_target_contact_force_n_by_sensor": "peak_robot_target_contact_forces",
        "initial_target_pose_wxyz": "initial_target_pose",
        "goal_pose_wxyz": "goal_pose",
        "initial_robot_joint_position_rad": "initial_robot_joint_position",
        "initial_robot_dynamics": "initial_robot_dynamics",
        "initial_tcp_position_m": "initial_tcp_position",
        "controller_parameters": "controller_metadata",
        "friction_backend_application": "friction_application",
        "finger_limits_backend_application": "finger_limits_application",
        "maximum_finger_opening_m": "maximum_finger_opening_m",
        "maximum_finger_mismatch_m": "maximum_finger_mismatch_m",
        "maximum_strict_dwell_steps": "maximum_strict_dwell",
        "strict_dwell_required_steps": "dwell_steps",
        "maximum_command_age_s": "maximum_command_age_s",
        "phase_wall_time_s": "phase_wall_time_s",
    }
    for output_key, local_key in fields.items():
        if local_key in state:
            evidence[output_key] = state[local_key]
    for key in ("physics_dt_s", "control_decimation", "audit_stride", "trace_stride"):
        if key in configuration:
            evidence[key] = configuration[key]
    if "executed_steps" in evidence and "control_period_s" in evidence:
        evidence["executed_sim_time_s"] = evidence["executed_steps"] * evidence["control_period_s"]
    if "finger_state_recorded_steps" in state:
        evidence["finger_state_artifact"] = {
            "path": str(state["finger_state_path"]),
            "recorded_steps": state["finger_state_recorded_steps"],
        }
    if evidence.get("trace"):
        evidence["last_recorded_trace_step"] = evidence["trace"][-1]["step"]
    return evidence
