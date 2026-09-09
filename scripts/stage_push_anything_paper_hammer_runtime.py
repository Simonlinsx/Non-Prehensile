#!/usr/bin/env python3
"""Create an isolated paper-controller runtime for one C1 hammer scene.

By default only the initial pose, goal pose, and sampling seed are changed.
The default keeps Push Anything's released sim-to-control separation: Isaac
uses the physical 50 g hammer while C3 plans with the stable 1 kg surrogate
controller body.  ``--planner-model physical`` remains available as an
explicit dynamics ablation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil


PLANNER_ROOT_HEIGHT_M = -0.01606310028273897


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--template-runtime", type=Path, required=True)
    parser.add_argument("--output-runtime", type=Path, required=True)
    parser.add_argument(
        "--planner-model",
        choices=("surrogate", "physical"),
        default="surrogate",
        help=(
            "C3 object model: Push Anything-style stable surrogate (default) "
            "or the physical 50 g hammer ablation"
        ),
    )
    parser.add_argument(
        "--pose-cost-switch-distance-m",
        type=float,
        help=(
            "optional native Push Anything XY-to-full-pose objective switch; "
            "omitting it preserves the template value"
        ),
    )
    parser.add_argument(
        "--num-reposition-samples",
        type=int,
        help="optional number of additional safe-handle reposition candidates",
    )
    parser.add_argument(
        "--num-c3-samples",
        type=int,
        help="optional number of additional safe-handle C3 candidates",
    )
    parser.add_argument(
        "--object-com-offset-xyz-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "optional target COM in the supported C3 link frame; used to "
            "align the planner dynamics with simulator/CAD mass properties"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-max-moment-arm-m",
        type=float,
        help=(
            "optional maximum moment arm for safe contacts while yaw is "
            "already acceptable"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-terminal-moment-residual-m",
        type=float,
        help=(
            "optional near-goal floor for the signed-yaw moment residual; "
            "the active bound grows continuously with the requested yaw "
            "correction up to the configured maximum"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-activation-threshold-rad",
        type=float,
        help="yaw-error envelope in which the low-moment contact filter is active",
    )
    parser.add_argument(
        "--neutral-yaw-contact-corrective-sign-deadband-rad",
        type=float,
        help=(
            "optional signed-yaw error deadband; outside it, safe contact "
            "candidates must produce a moment with the corrective sign"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-prediction-regression-tolerance-rad",
        type=float,
        help=(
            "optional tolerance for rejecting a candidate whose own C3 "
            "rollout predicts a larger terminal absolute yaw error"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-prediction-regression-activation-rad",
        type=float,
        help=(
            "optional absolute yaw error above which the monotone C3 "
            "prediction check is active"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-prediction-max-abs-error-rad",
        type=float,
        help=(
            "optional absolute yaw envelope for each candidate's terminal "
            "C3 rollout"
        ),
    )
    parser.add_argument(
        "--neutral-yaw-contact-com-offset-xy-m",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="target COM in the supported C3 link frame for moment-arm scoring",
    )
    parser.add_argument(
        "--neutral-yaw-contact-yaw-moment-gain-m-per-rad",
        type=float,
        help=(
            "optional signed yaw correction gain; zero recovers the original "
            "low-moment contact filter"
        ),
    )
    return parser.parse_args()


def replace_yaml_line(path: Path, key: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(?!\s*#)\s*{re.escape(key)}\s*:.*$", re.MULTILINE)
    if not pattern.search(text):
        raise KeyError(f"{path} does not define required key {key!r}")
    path.write_text(
        pattern.sub(f"{key}: {value}", text, count=1), encoding="utf-8"
    )


def load_scene(path: Path, index: int) -> dict[str, object]:
    scenes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 0 <= index < len(scenes):
        raise IndexError("scene index is outside the manifest")
    scene = scenes[index]
    if scene.get("schema") not in (
        "nonprehensile.push_anything_c1_scene.v1",
        "nonprehensile.push_anything_c1_scene.v2",
    ):
        raise ValueError("paper hammer runtime requires a C1 scene manifest")
    if scene.get("asset_id") != "020_hammer:0" or scene.get("clutter_count") != 0:
        raise ValueError("paper hammer runtime supports one uncluttered hammer")
    return scene


def replace_inertial_pose(path: Path, xyz: list[float]) -> None:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(<inertial>\s*<pose>)[^<]*(</pose>)",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise KeyError(f"{path} does not define an inertial pose")
    pose = " ".join(f"{value:.12g}" for value in (*xyz, 0.0, 0.0, 0.0))
    path.write_text(
        pattern.sub(rf"\g<1>{pose}\g<2>", text, count=1),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.scene_index < 0:
        raise ValueError("scene index must be non-negative")
    if (
        args.pose_cost_switch_distance_m is not None
        and args.pose_cost_switch_distance_m <= 0.0
    ):
        raise ValueError("pose-cost switch distance must be positive")
    if args.num_reposition_samples is not None and args.num_reposition_samples <= 0:
        raise ValueError("reposition sample count must be positive")
    if args.num_c3_samples is not None and args.num_c3_samples <= 0:
        raise ValueError("C3 sample count must be positive")
    if args.object_com_offset_xyz_m is not None and not all(
        math.isfinite(value) for value in args.object_com_offset_xyz_m
    ):
        raise ValueError("object COM offset must contain finite values")
    neutral_yaw_values = (
        args.neutral_yaw_contact_max_moment_arm_m,
        args.neutral_yaw_contact_activation_threshold_rad,
        args.neutral_yaw_contact_com_offset_xy_m,
    )
    if any(value is not None for value in neutral_yaw_values) and not all(
        value is not None for value in neutral_yaw_values
    ):
        raise ValueError("all neutral-yaw contact filter parameters are required")
    if args.neutral_yaw_contact_max_moment_arm_m is not None:
        if args.neutral_yaw_contact_max_moment_arm_m < 0.0:
            raise ValueError("neutral-yaw maximum moment arm must be non-negative")
        if args.neutral_yaw_contact_activation_threshold_rad < 0.0:
            raise ValueError("neutral-yaw activation threshold must be non-negative")
        if not all(
            math.isfinite(value)
            for value in args.neutral_yaw_contact_com_offset_xy_m
        ):
            raise ValueError("neutral-yaw COM offset must contain finite values")
    if args.neutral_yaw_contact_terminal_moment_residual_m is not None:
        if args.neutral_yaw_contact_max_moment_arm_m is None:
            raise ValueError(
                "neutral-yaw terminal moment residual requires the base "
                "contact filter parameters"
            )
        if not 0.0 <= args.neutral_yaw_contact_terminal_moment_residual_m <= (
            args.neutral_yaw_contact_max_moment_arm_m
        ):
            raise ValueError(
                "neutral-yaw terminal moment residual must lie between zero "
                "and the maximum moment residual"
            )
    if args.neutral_yaw_contact_yaw_moment_gain_m_per_rad is not None:
        if args.neutral_yaw_contact_max_moment_arm_m is None:
            raise ValueError(
                "signed yaw moment gain requires the base contact filter parameters"
            )
        if (
            not math.isfinite(
                args.neutral_yaw_contact_yaw_moment_gain_m_per_rad
            )
            or args.neutral_yaw_contact_yaw_moment_gain_m_per_rad < 0.0
        ):
            raise ValueError("signed yaw moment gain must be finite and non-negative")
    if args.neutral_yaw_contact_corrective_sign_deadband_rad is not None:
        if args.neutral_yaw_contact_max_moment_arm_m is None:
            raise ValueError(
                "corrective-sign deadband requires the base contact filter parameters"
            )
        if not 0.0 <= args.neutral_yaw_contact_corrective_sign_deadband_rad <= (
            args.neutral_yaw_contact_activation_threshold_rad
        ):
            raise ValueError(
                "corrective-sign deadband must lie between zero and the "
                "contact-filter activation threshold"
            )
    if args.neutral_yaw_contact_prediction_regression_tolerance_rad is not None:
        if args.neutral_yaw_contact_max_moment_arm_m is None:
            raise ValueError(
                "prediction regression tolerance requires the base contact "
                "filter parameters"
            )
        if (
            not math.isfinite(
                args.neutral_yaw_contact_prediction_regression_tolerance_rad
            )
            or args.neutral_yaw_contact_prediction_regression_tolerance_rad < 0.0
        ):
            raise ValueError(
                "prediction regression tolerance must be finite and non-negative"
            )
    if args.neutral_yaw_contact_prediction_regression_activation_rad is not None:
        if args.neutral_yaw_contact_prediction_regression_tolerance_rad is None:
            raise ValueError(
                "prediction regression activation requires its tolerance"
            )
        if (
            not math.isfinite(
                args.neutral_yaw_contact_prediction_regression_activation_rad
            )
            or args.neutral_yaw_contact_prediction_regression_activation_rad < 0.0
        ):
            raise ValueError(
                "prediction regression activation must be finite and non-negative"
            )
    if args.neutral_yaw_contact_prediction_max_abs_error_rad is not None:
        if args.neutral_yaw_contact_max_moment_arm_m is None:
            raise ValueError(
                "prediction yaw envelope requires the base contact filter parameters"
            )
        if (
            not math.isfinite(
                args.neutral_yaw_contact_prediction_max_abs_error_rad
            )
            or args.neutral_yaw_contact_prediction_max_abs_error_rad <= 0.0
        ):
            raise ValueError("prediction yaw envelope must be finite and positive")
    source_manifest = args.scene_manifest.expanduser().resolve()
    template = args.template_runtime.expanduser().resolve()
    output = args.output_runtime.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite isolated runtime: {output}")
    demo_name = (
        "hammer_safe"
        if args.planner_model == "surrogate"
        else "hammer_safe_physical"
    )
    required = (
        template / "examples/sampling_c3",
        template / "common/parameters",
        template / "solvers/osqp_options_default.yaml",
        template
        / f"examples/sampling_c3/{demo_name}/parameters"
        / "sampling_c3_controller_params.yaml",
    )
    if not source_manifest.is_file() or not all(path.exists() for path in required):
        raise FileNotFoundError("scene manifest or paper runtime template is missing")

    scene = load_scene(source_manifest, args.scene_index)
    initial_xy = [float(value) for value in scene["initial_xy_m"]]
    goal_xy = [float(value) for value in scene["goal_xy_m"]]
    goal_yaw_deg = float(scene["goal_yaw_deg"])
    sampling_seed = int(scene["sampling_seed"])
    if not all(math.isfinite(value) for value in (*initial_xy, *goal_xy, goal_yaw_deg)):
        raise ValueError("scene pose contains non-finite values")

    (output / "examples").mkdir(parents=True)
    (output / "common").mkdir()
    shutil.copytree(
        template / "examples/sampling_c3",
        output / "examples/sampling_c3",
        symlinks=True,
    )
    shutil.copytree(
        template / "common/parameters",
        output / "common/parameters",
        symlinks=True,
    )
    shutil.copytree(template / "solvers", output / "solvers", symlinks=True)

    params = output / "examples/sampling_c3"
    sim = params / "hammer_full/parameters/sim_params.yaml"
    goal = params / "hammer_full/parameters/goal_params.yaml"
    sampling = (
        params / "hammer_full/parameters/sampling_params.yaml"
        if args.planner_model == "surrogate"
        else params / "hammer_safe_physical/parameters/sampling_params.yaml"
    )
    controller = (
        params
        / f"{demo_name}/parameters/sampling_c3_controller_params.yaml"
    )
    progress = params / "anything/parameters/progress_params_c3plus.yaml"
    planner_model = (
        params
        / "urdf/DOMINO_020_hammer_safe"
        / (
            "DOMINO_020_hammer_safe_controller.sdf"
            if args.planner_model == "surrogate"
            else "DOMINO_020_hammer_safe.sdf"
        )
    )
    half_yaw = 0.5 * math.radians(goal_yaw_deg)
    goal_quaternion = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    replace_yaml_line(
        sim,
        "q_init_object",
        json.dumps([1.0, 0.0, 0.0, 0.0, *initial_xy, PLANNER_ROOT_HEIGHT_M]),
    )
    replace_yaml_line(
        goal,
        "fixed_target_position",
        json.dumps([*goal_xy, PLANNER_ROOT_HEIGHT_M]),
    )
    replace_yaml_line(
        goal, "fixed_target_orientation", json.dumps(goal_quaternion)
    )
    replace_yaml_line(sampling, "random_seed", str(sampling_seed))
    if args.num_reposition_samples is not None:
        replace_yaml_line(
            sampling,
            "num_additional_samples_repos",
            str(args.num_reposition_samples),
        )
    if args.num_c3_samples is not None:
        replace_yaml_line(
            sampling,
            "num_additional_samples_c3",
            str(args.num_c3_samples),
        )
    if args.object_com_offset_xyz_m is not None:
        replace_inertial_pose(
            planner_model,
            [float(value) for value in args.object_com_offset_xyz_m],
        )
    if args.neutral_yaw_contact_max_moment_arm_m is not None:
        filter_values = {
            "neutral_yaw_contact_max_moment_arm": (
                f"{args.neutral_yaw_contact_max_moment_arm_m:.12g}"
            ),
            "neutral_yaw_contact_activation_threshold": (
                f"{args.neutral_yaw_contact_activation_threshold_rad:.12g}"
            ),
            "neutral_yaw_contact_com_offset_xy": json.dumps(
                [
                    float(value)
                    for value in args.neutral_yaw_contact_com_offset_xy_m
                ]
            ),
        }
        if args.neutral_yaw_contact_yaw_moment_gain_m_per_rad is not None:
            filter_values["neutral_yaw_contact_yaw_moment_gain"] = (
                f"{args.neutral_yaw_contact_yaw_moment_gain_m_per_rad:.12g}"
            )
        if args.neutral_yaw_contact_terminal_moment_residual_m is not None:
            filter_values["neutral_yaw_contact_terminal_moment_residual"] = (
                f"{args.neutral_yaw_contact_terminal_moment_residual_m:.12g}"
            )
        if args.neutral_yaw_contact_corrective_sign_deadband_rad is not None:
            filter_values[
                "neutral_yaw_contact_corrective_sign_deadband"
            ] = f"{args.neutral_yaw_contact_corrective_sign_deadband_rad:.12g}"
        if (
            args.neutral_yaw_contact_prediction_regression_tolerance_rad
            is not None
        ):
            filter_values[
                "neutral_yaw_contact_prediction_regression_tolerance"
            ] = (
                f"{args.neutral_yaw_contact_prediction_regression_tolerance_rad:.12g}"
            )
        if (
            args.neutral_yaw_contact_prediction_regression_activation_rad
            is not None
        ):
            filter_values[
                "neutral_yaw_contact_prediction_regression_activation_threshold"
            ] = (
                f"{args.neutral_yaw_contact_prediction_regression_activation_rad:.12g}"
            )
        if args.neutral_yaw_contact_prediction_max_abs_error_rad is not None:
            filter_values[
                "neutral_yaw_contact_prediction_max_abs_error"
            ] = f"{args.neutral_yaw_contact_prediction_max_abs_error_rad:.12g}"
        controller_text = controller.read_text(encoding="utf-8")
        for key, value in filter_values.items():
            if re.search(
                rf"^(?!\s*#)\s*{re.escape(key)}\s*:",
                controller_text,
                re.MULTILINE,
            ):
                replace_yaml_line(controller, key, value)
            else:
                controller_text = controller.read_text(encoding="utf-8")
                controller.write_text(
                    controller_text.rstrip() + f"\n{key}: {value}\n",
                    encoding="utf-8",
                )
    if args.pose_cost_switch_distance_m is not None:
        replace_yaml_line(
            progress,
            "cost_switching_threshold_distance",
            f"{args.pose_cost_switch_distance_m:.12g}",
        )

    record = {
        "schema": "nonprehensile.push_anything_paper_runtime.v1",
        "source_manifest": str(source_manifest),
        "scene_index": args.scene_index,
        "scene_id": scene["scene_id"],
        "template_runtime": str(template),
        "runtime": str(output),
        "demo_name": demo_name,
        "planner_model": args.planner_model,
        "planner_object_model": str(planner_model),
        "initial_xy_m": initial_xy,
        "goal_xy_m": goal_xy,
        "goal_yaw_deg": goal_yaw_deg,
        "sampling_seed": sampling_seed,
        "pose_cost_switch_distance_m": args.pose_cost_switch_distance_m,
        "num_reposition_samples": args.num_reposition_samples,
        "num_c3_samples": args.num_c3_samples,
        "object_com_offset_xyz_m": args.object_com_offset_xyz_m,
        "neutral_yaw_contact_max_moment_arm_m": (
            args.neutral_yaw_contact_max_moment_arm_m
        ),
        "neutral_yaw_contact_activation_threshold_rad": (
            args.neutral_yaw_contact_activation_threshold_rad
        ),
        "neutral_yaw_contact_com_offset_xy_m": (
            args.neutral_yaw_contact_com_offset_xy_m
        ),
        "neutral_yaw_contact_yaw_moment_gain_m_per_rad": (
            args.neutral_yaw_contact_yaw_moment_gain_m_per_rad
        ),
        "neutral_yaw_contact_terminal_moment_residual_m": (
            args.neutral_yaw_contact_terminal_moment_residual_m
        ),
        "neutral_yaw_contact_corrective_sign_deadband_rad": (
            args.neutral_yaw_contact_corrective_sign_deadband_rad
        ),
        "neutral_yaw_contact_prediction_regression_tolerance_rad": (
            args.neutral_yaw_contact_prediction_regression_tolerance_rad
        ),
        "neutral_yaw_contact_prediction_regression_activation_rad": (
            args.neutral_yaw_contact_prediction_regression_activation_rad
        ),
        "neutral_yaw_contact_prediction_max_abs_error_rad": (
            args.neutral_yaw_contact_prediction_max_abs_error_rad
        ),
        "changed_parameters": [
            "q_init_object",
            "fixed_target_position",
            "fixed_target_orientation",
            "random_seed",
            *(
                ["num_additional_samples_repos"]
                if args.num_reposition_samples is not None else []
            ),
            *(
                ["num_additional_samples_c3"]
                if args.num_c3_samples is not None else []
            ),
            *(
                ["physical_model_inertial_pose"]
                if args.object_com_offset_xyz_m is not None else []
            ),
            *(
                ["neutral_yaw_contact_filter"]
                if args.neutral_yaw_contact_max_moment_arm_m is not None else []
            ),
            *(
                ["cost_switching_threshold_distance"]
                if args.pose_cost_switch_distance_m is not None else []
            ),
        ],
        "controller_parameters_changed": (
            args.pose_cost_switch_distance_m is not None
            or args.num_reposition_samples is not None
            or args.num_c3_samples is not None
            or args.object_com_offset_xyz_m is not None
            or args.neutral_yaw_contact_max_moment_arm_m is not None
            or args.neutral_yaw_contact_terminal_moment_residual_m is not None
            or args.neutral_yaw_contact_corrective_sign_deadband_rad is not None
            or args.neutral_yaw_contact_prediction_regression_tolerance_rad
            is not None
            or args.neutral_yaw_contact_prediction_regression_activation_rad
            is not None
            or args.neutral_yaw_contact_prediction_max_abs_error_rad is not None
        ),
    }
    (output / "runtime_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
