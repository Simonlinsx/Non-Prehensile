#!/usr/bin/env bash
set -euo pipefail

# Reproducible quantitative evaluation and qualitative video entry point for
# the oracle-affordance teacher. PROFILE selects the typed constraint task and
# its disjoint diagnostic manifest.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
PROFILE="${PROFILE:-c1}"
GPU_ID="${GPU_ID:-6}"
SEED="${SEED:-7001}"
NUM_ENVS="${NUM_ENVS:-64}"
NUM_EPISODES="${NUM_EPISODES:-1024}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
VIDEO="${VIDEO:-0}"
ZERO_ACTIONS="${ZERO_ACTIONS:-0}"
# The base environment viewer is intentionally wide (eye=(8, 0, 5)).  That is
# useful for debugging tiled environments but makes a single-environment proof
# video unreadable.  Keep an explicit, overridable close-up camera contract for
# all teacher demos.
CAMERA_EYE="${CAMERA_EYE:-1.15 -1.05 0.85}"
CAMERA_LOOKAT="${CAMERA_LOOKAT:-0.48 0.00 0.10}"
SCENE_INDEX="${SCENE_INDEX:-0}"
RUN_LABEL="${RUN_LABEL:-$(basename -- "${CHECKPOINT:-checkpoint}" .pt)}"
DOMINO_DATA_ROOT="${DOMINO_ROOT:-/data1/linsixu/DOMINO}"
DOMINO_CONVERTED_ROOT="${DOMINO_USD_ROOT:-$REPO_ROOT/data/domino_usd}"
DAPL_LOCAL_FRANKA_USD_DIR="${DAPL_LOCAL_FRANKA_USD_DIR:-/data1/linsixu/tmp/isaaclab_nonprehensile_teacher/franka_usd}"

case "${OMNI_KIT_ACCEPT_EULA:-}" in
  y|Y|yes|YES|1) ;;
  *)
    echo "Isaac Sim requires OMNI_KIT_ACCEPT_EULA=YES." >&2
    exit 2
    ;;
esac

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "Set CHECKPOINT to an existing teacher model_*.pt." >&2
  exit 2
fi

case "$PROFILE" in
  t0_frozenv7_goalwrench_forward)
    TASK="Isaac-AffordanceTeacher-T0-FrozenV7-GoalWrench-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_forward_v9/hammer_teacher_forward_heldout128_seed7829.jsonl"
    ;;
  t0_frozenv7_goalwrench_dir45)
    TASK="Isaac-AffordanceTeacher-T0-FrozenV7-GoalWrench-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl"
    ;;
  c1soft_goalwrench_dir45)
    TASK="Isaac-AffordanceTeacher-GoalWrench-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl"
    ;;
  c1_frozenv7_goalwrench_forward)
    TASK="Isaac-AffordanceTeacher-FrozenV7-GoalWrench-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_forward_v9/hammer_teacher_forward_heldout128_seed7829.jsonl"
    ;;
  c1_frozenv7_goalwrench_dir45)
    TASK="Isaac-AffordanceTeacher-FrozenV7-GoalWrench-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl"
    ;;
  c2_goalwrench_short)
    TASK="Isaac-AffordanceTeacher-GoalWrench-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_short_eraser_eval_v15/hammer_teacher_c2_short12cm_eraser_eval_stable_resetclear_seed20851.jsonl"
    ;;
  c2_goalwrench_short_forward)
    TASK="Isaac-AffordanceTeacher-GoalWrench-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_short_eraser_dir45_v31/hammer_teacher_c2_short12cm_eraser_dir45_eval_seed20851.jsonl"
    ;;
  c2_goalwrench_matched_phone)
    TASK="Isaac-AffordanceTeacher-GoalWrench-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_matched_v35/hammer_teacher_dir45_c2_phone_eval_final_stable_seed9833.jsonl"
    ;;
  c2_goalwrench_matched_box_clear10)
    TASK="Isaac-AffordanceTeacher-GoalWrench-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_matched_v40/hammer_teacher_dir45_c2_plasticbox_uniform006_kinematic_eval_clear10_seed9833.jsonl"
    ;;
  t0_frozenv7_goalwrench_armdiv_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-FrozenV7-GoalWrench-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dywa_armdiv_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_v7_reward_armdiv_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-FrozenV7-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dywa_armdiv_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_dywa_matched_potentials_cartesian_noc1_armdiv_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DyWAMatchedPotentialsCartesian-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dywa_armdiv_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_dywa_matched_potentials_multiquery16_action010_noc1_armdiv_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DyWAMatchedPotentialsMultiQuery16Action010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dywa_armdiv_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_dywa_matched_potentials_action010_noc1_armdiv_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DyWAMatchedPotentialsAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dywa_armdiv_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_dywa_matched_potentials_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DyWAMatchedPotentialsAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_dywa_bbox_fullscale_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DyWABBoxFullScaleAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_dywa_keypoint_potential_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DyWAKeypointPotentialAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_pareto_pose_improvement_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-ParetoPoseImprovementAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_positive_component_improvement_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-PositiveComponentImprovementAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_weighted_component_progress_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-WeightedComponentProgressAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_positive_initial_relative_joint_goal_action010_noc1_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-PositiveInitialRelativeJointGoalAction010-NoC1Diagnostic-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_leaky_signed_initial_relative_joint_goal_action010_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-LeakySignedInitialRelativeJointGoalAction010-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_signed_initial_relative_joint_goal_action010_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-SignedInitialRelativeJointGoalAction010-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_positive_initial_relative_joint_goal_action010_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-PositiveInitialRelativeJointGoalAction010-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_positive_initial_relative_joint_goal_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-PositiveInitialRelativeJointGoal-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_positive_initial_relative_dapl_goal_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-PositiveInitialRelativeDAPLGoal-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_initial_relative_dapl_goal_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-InitialRelativeDAPLGoal-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_distance_dapl_goal_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-DistanceDAPLGoal-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_unified_distance_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-UnifiedDistance-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_unified_planarpush)
    TASK="Isaac-AffordanceTeacher-T0-UnifiedProgress-C1-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl"
    ;;
  t0_forward)
    TASK="Isaac-AffordanceTeacher-T0-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_forward_v9/hammer_teacher_forward_heldout128_seed7829.jsonl"
    ;;
  t0_dir45)
    TASK="Isaac-AffordanceTeacher-T0-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl"
    ;;
  t0_dir90)
    TASK="Isaac-AffordanceTeacher-T0-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir90_eval128_seed11839.jsonl"
    ;;
  t0_goal_side_dir90)
    TASK="Isaac-AffordanceTeacher-T0-GoalSide-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir90_eval128_seed11839.jsonl"
    ;;
  t0_dir360_fixedyaw)
    TASK="Isaac-AffordanceTeacher-T0-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir360_fixedyaw_eval128_seed13843.jsonl"
    ;;
  c1_dir45)
    TASK="Isaac-AffordanceTeacher-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl"
    ;;
  c1_dir90)
    TASK="Isaac-AffordanceTeacher-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir90_eval128_seed11839.jsonl"
    ;;
  c1_relation_dir90)
    # Relation-aware checkpoints have additional point-relation encoder
    # parameters.  Evaluate them in the identical hard-C1 environment while
    # instantiating the matching policy architecture.
    TASK="Isaac-AffordanceTeacher-Relation-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir90_eval128_seed11839.jsonl"
    ;;
  c1_relation_wrench_dir90)
    TASK="Isaac-AffordanceTeacher-RelationWrench-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_biyaw_v18/hammer_teacher_dir90_biyaw_eval128_seed12839.jsonl"
    ;;
  c1_relation_wrench_separated_dir90)
    TASK="Isaac-AffordanceTeacher-RelationWrenchSeparated-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_biyaw_v18/hammer_teacher_dir90_biyaw_eval128_seed12839.jsonl"
    ;;
  c2_relation_wrench_separated_short)
    TASK="Isaac-AffordanceTeacher-RelationWrenchSeparated-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_short_eraser_eval_v15/hammer_teacher_c2_short12cm_eraser_eval_stable_resetclear_seed20851.jsonl"
    ;;
  c2_protected_obstacle_short)
    TASK="Isaac-AffordanceTeacher-ProtectedObstacle-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_short_eraser_eval_v15/hammer_teacher_c2_short12cm_eraser_eval_stable_resetclear_seed20851.jsonl"
    ;;
  c3_relation_wrench_separated_short)
    TASK="Isaac-AffordanceTeacher-RelationWrenchSeparated-C3-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c3_short_eval_v12/hammer_teacher_c3_short12cm_eval_stable_resetclear_seed15851.jsonl"
    ;;
  combined_relation_wrench_separated)
    TASK="Isaac-AffordanceTeacher-RelationWrenchSeparated-Combined-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_v7/hammer_teacher_combined_stable_seed5821.jsonl"
    ;;
  combined_relation_wrench_separated_demo)
    TASK="Isaac-AffordanceTeacher-RelationWrenchSeparated-Combined-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v4/hammer_teacher_combined_final_seed1817.jsonl"
    ;;
  c1_dir360_fixedyaw)
    TASK="Isaac-AffordanceTeacher-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir360_fixedyaw_eval128_seed13843.jsonl"
    ;;
  t0)
    TASK="Isaac-AffordanceTeacher-T0-Soft-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_pose_v8/hammer_teacher_pose_heldout256_stable_seed6827.jsonl"
    ;;
  c1_matched)
    TASK="Isaac-AffordanceTeacher-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_pose_v8/hammer_teacher_pose_heldout256_stable_seed6827.jsonl"
    ;;
  c1_forward)
    TASK="Isaac-AffordanceTeacher-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_forward_v9/hammer_teacher_forward_heldout128_seed7829.jsonl"
    ;;
  c1)
    TASK="Isaac-AffordanceTeacher-C1-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v4/hammer_teacher_t0_c1_32_seed1817.jsonl"
    ;;
  c2)
    TASK="Isaac-AffordanceTeacher-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v3/hammer_teacher_c2_final_seed1817.jsonl"
    ;;
  c2_short)
    TASK="Isaac-AffordanceTeacher-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_short_eraser_eval_v15/hammer_teacher_c2_short12cm_eraser_eval_stable_resetclear_seed20851.jsonl"
    ;;
  c2_relation)
    TASK="Isaac-AffordanceTeacher-Relation-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v3/hammer_teacher_c2_final_seed1817.jsonl"
    ;;
  c2_relation_short)
    TASK="Isaac-AffordanceTeacher-Relation-C2-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c2_short_eraser_eval_v15/hammer_teacher_c2_short12cm_eraser_eval_stable_resetclear_seed20851.jsonl"
    ;;
  c3)
    TASK="Isaac-AffordanceTeacher-C3-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v3/hammer_teacher_c3_final_seed1817.jsonl"
    ;;
  c3_short)
    TASK="Isaac-AffordanceTeacher-C3-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c3_short_eval_v12/hammer_teacher_c3_short12cm_eval_stable_resetclear_seed15851.jsonl"
    ;;
  c3_relation)
    TASK="Isaac-AffordanceTeacher-Relation-C3-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v3/hammer_teacher_c3_final_seed1817.jsonl"
    ;;
  c3_relation_short)
    TASK="Isaac-AffordanceTeacher-Relation-C3-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_c3_short_eval_v12/hammer_teacher_c3_short12cm_eval_stable_resetclear_seed15851.jsonl"
    ;;
  combined)
    TASK="Isaac-AffordanceTeacher-Combined-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_v7/hammer_teacher_combined_stable_seed5821.jsonl"
    ;;
  combined_demo)
    TASK="Isaac-AffordanceTeacher-Combined-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v4/hammer_teacher_combined_final_seed1817.jsonl"
    ;;
  combined_relation)
    TASK="Isaac-AffordanceTeacher-Relation-Combined-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_heldout_v7/hammer_teacher_combined_stable_seed5821.jsonl"
    ;;
  combined_relation_demo)
    TASK="Isaac-AffordanceTeacher-Relation-Combined-Franka-v0"
    DEFAULT_MANIFEST="$REPO_ROOT/data/manifests/teacher_diagnostics_v4/hammer_teacher_combined_final_seed1817.jsonl"
    ;;
  *)
    echo "Unknown PROFILE: $PROFILE" >&2
    exit 2
    ;;
esac

# Keep the evaluation entry point compatible with the training/audit contract:
# callers historically select a scene set through DAPL_CLUTTER_MANIFEST.  An
# explicit MANIFEST remains the highest-priority, script-local override.
MANIFEST="${MANIFEST:-${DAPL_CLUTTER_MANIFEST:-$DEFAULT_MANIFEST}}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing evaluation manifest: $MANIFEST" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/teacher_eval/${RUN_LABEL}_${PROFILE}_seed${SEED}}"
mkdir -p "$OUTPUT_DIR"

video_args=()
if [[ "$VIDEO" == "1" ]]; then
  NUM_ENVS=1
  NUM_EPISODES="${VIDEO_NUM_EPISODES:-1}"
  VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
  VIDEO_FOLDER="${VIDEO_FOLDER:-$REPO_ROOT/outputs/teacher_demos/${RUN_LABEL}/${PROFILE}}"
  mkdir -p "$VIDEO_FOLDER"
  video_args=(
    --video
    --video_length "$VIDEO_LENGTH"
    --video_folder "$VIDEO_FOLDER"
    --video_name_prefix "${VIDEO_NAME_PREFIX:-${RUN_LABEL}_${PROFILE}}"
    --visualize_goal
    --goal_ghost_opacity "${GOAL_GHOST_OPACITY:-0.55}"
    --visualize_affordance
    --trace_episode_ends
    --camera_eye $CAMERA_EYE
    --camera_lookat $CAMERA_LOOKAT
  )
fi

diagnostic_args=()
if [[ "$ZERO_ACTIONS" == "1" ]]; then
  diagnostic_args=(--zero_actions)
fi

cd "$REPO_ROOT"
echo "Evaluating teacher: profile=$PROFILE seed=$SEED envs=$NUM_ENVS episodes=$NUM_EPISODES manifest=$MANIFEST"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
DOMINO_ROOT="$DOMINO_DATA_ROOT" \
DOMINO_USD_ROOT="$DOMINO_CONVERTED_ROOT" \
DAPL_CLUTTER_ASSET_SOURCE=domino \
DAPL_CLUTTER_MANIFEST="$MANIFEST" \
DAPL_CLUTTER_SCENE_OFFSET="$SCENE_INDEX" \
DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0 \
DAPL_LOCAL_FRANKA_USD_DIR="$DAPL_LOCAL_FRANKA_USD_DIR" \
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/source/IsaacLab_nonPrehensile${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" scripts/eval.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --num_envs "$NUM_ENVS" \
    --num_episodes "$NUM_EPISODES" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --deterministic \
    --headless \
    "${diagnostic_args[@]}" \
    "${video_args[@]}"
