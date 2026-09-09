#!/usr/bin/env python3
"""Numerically validate the live Isaac OSC task-point Jacobian and twist.

Accepts the online task runner's arguments, with --output naming this audit.
No C3 process or contact-driving policy is launched.
"""
import json
import os
import traceback

import run_c3_online_isaaclab_task as runner
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    args = runner.args_cli
    if args.osc_control_point != "tip":
        raise ValueError("This check requires --osc-control-point tip")
    os.environ["DAPL_CLUTTER_MANIFEST"] = str(args.manifest.resolve())
    os.environ["DAPL_CLUTTER_ASSET_SOURCE"] = "domino"
    os.environ["DOMINO_ROOT"] = str(args.domino_root.resolve())
    os.environ["DOMINO_USD_ROOT"] = str(args.domino_usd_root.resolve())
    env = runner.gym.make(args.task, cfg=runner._configure_env())
    try:
        env.reset()
        base = env.unwrapped
        robot = base.scene["robot"]
        action = base.action_manager.get_term("arm_action")
        initial = robot.data.joint_pos.clone()
        qdot = runner.torch.zeros_like(initial)
        qdot[0, :7] = runner.torch.tensor([.1, -.07, .04, -.05, .03, .09, -.08], device=base.device)

        def set_pose(q):
            robot.write_joint_state_to_sim(q, qdot)
            base.sim.forward()
            base.scene.update(.001)
            action._compute_ee_pose()
            return action._ee_pose_b[0].detach().cpu().numpy().copy()

        checks = []
        epsilon = 1e-3
        for displacement in [0.0, .15, -.15]:
            q = initial.clone()
            q[0, 0] += displacement
            q[0, 4] += displacement
            set_pose(q)
            action._compute_ee_jacobian()
            J = action._jacobian_b[0].detach().cpu().numpy().copy()
            action._compute_ee_velocity()
            twist = action._ee_vel_b[0].detach().cpu().numpy().copy()
            columns = []
            for joint in range(7):
                poses = []
                for sign in [1, -1]:
                    dq = q.clone()
                    dq[0, joint] += sign * epsilon
                    poses.append(set_pose(dq))
                angular = (Rotation.from_quat(np.roll(poses[0][3:7], -1)) *
                           Rotation.from_quat(np.roll(poses[1][3:7], -1)).inv()).as_rotvec() / (2 * epsilon)
                columns.append(np.r_[(poses[0][:3] - poses[1][:3]) / (2 * epsilon), angular])
            numerical = np.stack(columns, axis=1)
            reference_twist = numerical @ qdot[0, :7].cpu().numpy()
            # Virtual work: a force at the tip equals a force plus r x F at
            # the hand. This checks that a distal force carries wrist moment.
            set_pose(q)
            raw = action.jacobian_w[0].detach().cpu().numpy().copy()
            hand_id = robot.body_names.index("panda_hand")
            hand_q = robot.data.body_quat_w[0, hand_id].cpu().numpy()
            r = Rotation.from_quat(np.roll(hand_q, -1)).apply([0, 0, args.drake_tip_from_hand_m])
            force = np.array([.3, -.2, .4])
            at_hand = raw[:3].T @ force + raw[3:].T @ np.cross(r, force)
            checks.append({
                "joint_displacement_rad": displacement,
                "maximum_linear_jacobian_error_m_per_rad": float(np.max(np.abs(J[:3] - numerical[:3]))),
                "maximum_angular_jacobian_error": float(np.max(np.abs(J[3:] - numerical[3:]))),
                "maximum_twist_error": float(np.max(np.abs(twist - reference_twist))),
                "maximum_wrench_virtual_work_error_nm": float(np.max(np.abs(J[:3].T @ force - at_hand))),
            })
        report = {"schema": "nonprehensile.task_point_kinematics_validation.v1", "checks": checks,
                  "passed": all(c["maximum_linear_jacobian_error_m_per_rad"] < 1e-3
                                and c["maximum_angular_jacobian_error"] < 1e-3
                                and c["maximum_twist_error"] < 5e-4
                                and c["maximum_wrench_virtual_work_error_nm"] < 1e-6 for c in checks)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("TASK_POINT_KINEMATICS_AUDIT", json.dumps(report), flush=True)
        return 0 if report["passed"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    status = 1
    try:
        status = main()
    except Exception as exc:
        runner.args_cli.output.write_text(json.dumps({"passed": False, "error": str(exc)}) + "\n")
        traceback.print_exc()
    finally:
        runner.simulation_app.close()
    raise SystemExit(status)
