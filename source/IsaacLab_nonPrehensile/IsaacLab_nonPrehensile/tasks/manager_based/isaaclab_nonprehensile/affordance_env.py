"""DOMINO-backed curriculum for part-aware non-prehensile manipulation."""

from __future__ import annotations

import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.clutter_env import (
    Clutter6DEnv,
    Clutter6DEnvCfg,
    Clutter6DObservationsCfg,
    Clutter6DRewardsCfg,
    Clutter6DSceneCfg,
    Clutter6DTerminationsCfg,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env import (
    CartesianDeltaPoseActionsCfg,
)


HAMMER_PROOF_ARM_JOINT_POS = {
    "panda_joint1": 0.1958014377,
    "panda_joint2": 0.3190638271,
    "panda_joint3": -0.1205807571,
    "panda_joint4": -2.7115141420,
    "panda_joint5": 0.3226507934,
    "panda_joint6": 3.0223125043,
    "panda_joint7": 0.5461376014,
}

# Midpoint of the repository's conservative Franka joint box.  Unlike the
# proof pose above, this is an observation/approach pose rather than a
# pre-contact pose, so arbitrary supported hammer yaw cannot spawn through the
# fingers before the first policy action.
DIRECTIONAL_CLUTTER_ARM_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.0398,
    "panda_joint3": 0.0,
    "panda_joint4": -2.13345,
    "panda_joint5": 0.0,
    "panda_joint6": 2.05065,
    "panda_joint7": 0.0,
}

TEACHER_ROBOT_TARGET_SENSOR = "target_robot_contacts"
TEACHER_ROBOT_OBSTACLE_SENSORS = tuple(
    [f"robot_obstacle_link{index}_contacts" for index in range(8)]
    + [
        "robot_obstacle_hand_contacts",
        "robot_obstacle_leftfinger_contacts",
        "robot_obstacle_rightfinger_contacts",
    ]
)
TEACHER_TARGET_OBSTACLE_SENSOR = "target_obstacle_contacts"


def _semantic_params() -> dict:
    return {
        "safe_radius_m": None,
        "protected_radius_m": None,
        "target_cfg": SceneEntityCfg("target"),
    }


def _policy_scene_params() -> dict:
    return {
        "target_point_count": 512,
        "obstacle_point_count": 512,
        "safe_radius_m": None,
        "protected_radius_m": None,
        "target_cfg": SceneEntityCfg("target"),
        "obstacles_cfg": SceneEntityCfg("obstacles"),
    }


def _contact_params(*, evaluate_protected: bool = True) -> dict:
    return {
        # Surface clouds contain only 512 target and 256 hand points.  A
        # 10-mm proxy radius covers their sampling gap; the previous 8 mm
        # radius missed a verified PhysX handle contact by 0.5 mm.
        "contact_distance_m": 0.010,
        "minimum_safe_score": 0.25,
        "minimum_protected_score": 0.25,
        "protected_point_count": 64,
        "protected_clearance_m": 0.005,
        "evaluate_protected": evaluate_protected,
        "safe_radius_m": None,
        "protected_radius_m": None,
        "target_cfg": SceneEntityCfg("target"),
        "obstacles_cfg": SceneEntityCfg("obstacles"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    }


def _teacher_contact_params(*, evaluate_protected: bool = True) -> dict:
    """Exact teacher audit contract layered on the semantic point proxy."""

    params = _contact_params(evaluate_protected=evaluate_protected)
    params.update(
        {
            "physical_contact_force_threshold_n": 0.5,
            "require_physical_protected_contact": True,
            "robot_target_sensor_name": TEACHER_ROBOT_TARGET_SENSOR,
            "target_obstacle_sensor_name": TEACHER_TARGET_OBSTACLE_SENSOR,
        }
    )
    return params


def _teacher_c1_params(*, evaluate_protected: bool = True) -> dict:
    params = _contact_params(evaluate_protected=evaluate_protected)
    params.update(
        {
            "physical_contact_force_threshold_n": 0.5,
            "robot_target_sensor_name": TEACHER_ROBOT_TARGET_SENSOR,
        }
    )
    return params


def _teacher_c2_params(*, evaluate_protected: bool = True) -> dict:
    params = _contact_params(evaluate_protected=evaluate_protected)
    params.update(
        {
            "physical_contact_force_threshold_n": 0.5,
            "require_physical_protected_contact": True,
            "target_obstacle_sensor_name": TEACHER_TARGET_OBSTACLE_SENSOR,
        }
    )
    return params


def _teacher_robot_obstacle_params() -> dict:
    return {
        "robot_obstacle_clearance_m": 0.005,
        "physical_contact_force_threshold_n": 0.5,
        "robot_obstacle_sensor_name": TEACHER_ROBOT_OBSTACLE_SENSORS,
        "target_cfg": SceneEntityCfg("target"),
        "obstacles_cfg": SceneEntityCfg("obstacles"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    }


def _goal_params() -> dict:
    return {
        "command_name": "target_object_pose",
        "planar_position_threshold": 0.02,
        "height_threshold": 0.01,
        "rotation_threshold": 0.1,
        "consecutive_steps": 5,
        "target_cfg": SceneEntityCfg("target"),
    }


def _joint_pose_reward_params() -> dict:
    return {
        "command_name": "target_object_pose",
        "planar_position_threshold": 0.02,
        "height_threshold": 0.01,
        "rotation_threshold": 0.1,
        "smooth_max_temperature": 0.25,
        "target_cfg": SceneEntityCfg("target"),
    }


def _teacher_obstacle_filter_paths() -> list[str]:
    """Return one per-slot filter; each expression must match once per env."""

    return [f"{{ENV_REGEX_NS}}/Obstacle_{index:02d}" for index in range(3)]


@configclass
class AffordanceAwareObservationsCfg(Clutter6DObservationsCfg):
    """Point-aligned semantic target, obstacle geometry, and 50-D state."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ActorCriticAffordance parses this fixed 4,096-D prefix first.
        affordance_scene = ObsTerm(
            func=mdp.domino_affordance_policy_scene,
            params=_policy_scene_params(),
        )
        hand_state = ObsTerm(
            func=mdp.hand_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )
        robot_state = ObsTerm(func=mdp.robot_state)
        previous_action = ObsTerm(func=mdp.last_action)
        rel_goal = ObsTerm(
            func=mdp.rel_pose_goal,
            params={
                "command_name": "target_object_pose",
                "object_cfg": SceneEntityCfg("target"),
            },
        )
        target_twist = ObsTerm(
            func=mdp.object_twist_in_env_frame,
            params={"object_cfg": SceneEntityCfg("target")},
        )
        phys_params = ObsTerm(
            func=mdp.phys_params,
            params={"object_cfg": SceneEntityCfg("target")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class WorldModelCfg(Clutter6DObservationsCfg.WorldModelCfg):
        target_affordance = ObsTerm(
            func=mdp.domino_target_affordance,
            params=_semantic_params(),
        )

        def __post_init__(self) -> None:
            super().__post_init__()

    world_model: WorldModelCfg = WorldModelCfg()


@configclass
class AffordanceTeacherSceneCfg(Clutter6DSceneCfg):
    """Add independent PhysX reporters for the three teacher constraints."""

    # A filtered ContactSensor can own only one rigid-body reporter per
    # environment.  C1/C2 therefore invert the query and report on the single
    # target body; C3 uses one reporter per Franka body.
    target_robot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        update_period=0.0,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/panda_link{index}" for index in range(8)
        ],
    )
    target_obstacle_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )

    robot_obstacle_link0_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link1_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link1",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link2_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link2",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link3_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link3",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link4_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link4",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link5_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link5",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link6_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link6",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_link7_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link7",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_hand_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_leftfinger_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )
    robot_obstacle_rightfinger_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        filter_prim_paths_expr=_teacher_obstacle_filter_paths(),
    )


@configclass
class AffordanceTeacherObservationsCfg(AffordanceAwareObservationsCfg):
    """Recoverable actor inputs plus an independent privileged value baseline."""

    @configclass
    class PolicyCfg(AffordanceAwareObservationsCfg.PolicyCfg):
        # Object twist is recoverable from an RGB-D history, but never exact
        # on hardware.  Add conservative noise now so it cannot become a
        # brittle simulator shortcut.
        target_twist = ObsTerm(
            func=mdp.object_twist_in_env_frame,
            params={"object_cfg": SceneEntityCfg("target")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        )
        # Exact mass/friction/restitution remain available in the world-model
        # observation and simulator labels, not in the deployable actor.
        phys_params = None

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(AffordanceAwareObservationsCfg.PolicyCfg):
        """Training-only critic view with exact scalar dynamics labels.

        The actor never consumes this group.  ActorCriticAffordance also uses
        an independent critic encoder, so these five scalars cannot leak into
        the action pathway through shared PointNet or attention weights.
        """

        target_twist = ObsTerm(
            func=mdp.object_twist_in_env_frame,
            params={"object_cfg": SceneEntityCfg("target")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class AffordanceAwareRewardsCfg(Clutter6DRewardsCfg):
    """Dense progress plus progressively activated semantic constraints."""

    contact_reward = None
    object_goal_tracking = None
    object_goal_tracking_fine_grained = None
    task_success = RewTerm(
        func=mdp.affordance_task_success_reward,
        params={"termination_term_name": "reached", **_contact_params()},
        # RewardManager multiplies weights by dt=0.1.  This makes reaching the
        # terminal goal preferable to lingering nearby for dense rewards.
        weight=2000.0,
    )
    safe_region_distance = RewTerm(
        func=mdp.safe_region_distance_penalty,
        params={"normalization_distance_m": 0.10, **_contact_params()},
        weight=-2.0,
    )
    safe_region_progress = RewTerm(
        func=mdp.safe_region_distance_progress_reward,
        params={"normalization_distance_m": 0.02, **_contact_params()},
        weight=8.0,
    )
    # Optional direction-aware set potential.  It remains disabled in the
    # baseline profiles and is activated only by the explicit goal-side
    # ablation, preserving checkpoint/reward comparability.
    goal_conditioned_safe_region_distance = None
    goal_conditioned_safe_region_progress = None
    semantic_corridor_approach = None
    semantic_corridor_progress = None
    semantic_geodesic_approach = None
    semantic_geodesic_progress = None
    semantic_vector_field_progress = None
    first_safe_region_contact = RewTerm(
        func=mdp.first_safe_region_contact_reward,
        params=_contact_params(),
        # With dt=0.1 this produces a one-time reward of 5.0.
        weight=50.0,
    )
    # Activated only by the safe-start approach curriculum.  Keeping this
    # field in the shared config makes the reward-manager schema explicit.
    safe_contact_push_progress = None
    safe_contact_joint_pose_progress = None
    safe_contact_planar_progress = None
    safe_contact_height_progress = None
    safe_contact_rotation_progress = None
    near_goal_target_motion = None
    # Do not separately optimize XY and orientation: that lets the policy
    # trade one success condition against another.  One signed potential uses
    # the same XY/Z/SO(3) thresholds as strict success.
    joint_pose_progress = RewTerm(
        func=mdp.affordance_joint_pose_progress_reward,
        params=_joint_pose_reward_params(),
        weight=16.0,
    )
    joint_pose_tracking = RewTerm(
        func=mdp.affordance_joint_pose_reward,
        params=_joint_pose_reward_params(),
        weight=2.0,
    )
    initial_relative_dapl_pose_score = None
    weighted_component_pose_progress = None
    positive_reference_component_pose_improvement = None
    positive_reference_pareto_pose_improvement = None
    dywa_keypoint_pose_potential = None
    post_contact_joint_pose_tracking = None
    post_contact_joint_pose_improvement = None
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.10)
    forbidden_region_contact = RewTerm(
        func=mdp.forbidden_region_contact_penalty,
        params=_contact_params(),
        # Stages 0/1 keep this constraint soft, but make violating the mask
        # costly enough that it cannot be ignored while learning to push.
        weight=-5.0,
    )
    protected_region_clearance = RewTerm(
        func=mdp.protected_region_clearance_penalty,
        params={"activation_distance_m": 0.05, **_contact_params()},
        weight=-4.0,
    )
    protected_region_clearance_progress = None
    protected_region_geodesic_progress = None
    protected_region_lateral_escape_progress = None
    protected_region_collision = RewTerm(
        func=mdp.protected_region_collision_penalty,
        params=_contact_params(),
        weight=-10.0,
    )


@configclass
class AffordanceAwareTerminationsCfg(Clutter6DTerminationsCfg):
    """Final-stage hard constraints; earlier stages disable selected terms."""

    reached = DoneTerm(func=mdp.affordance_target_reached_goal, params=_goal_params())
    forbidden_region_contact = DoneTerm(
        func=mdp.forbidden_region_contact,
        params=_contact_params(),
    )
    protected_region_collision = DoneTerm(
        func=mdp.protected_region_collision,
        params=_contact_params(),
    )


@configclass
class AffordanceTeacherRewardsCfg(AffordanceAwareRewardsCfg):
    """All three typed costs, with physical contacts used for hard events."""

    task_success = RewTerm(
        func=mdp.affordance_task_success_reward,
        params={
            "termination_term_name": "reached",
            "evaluate_robot_obstacle": True,
            "robot_obstacle_clearance_m": 0.005,
            "robot_obstacle_sensor_name": TEACHER_ROBOT_OBSTACLE_SENSORS,
            **_teacher_contact_params(),
        },
        weight=2000.0,
    )
    forbidden_region_contact = RewTerm(
        func=mdp.forbidden_region_contact_penalty,
        params=_teacher_c1_params(),
        weight=-25.0,
    )
    forbidden_region_clearance = RewTerm(
        func=mdp.robot_forbidden_region_clearance_penalty,
        params={"activation_distance_m": 0.02, **_teacher_c1_params()},
        weight=-4.0,
    )
    protected_region_collision = RewTerm(
        func=mdp.protected_region_collision_penalty,
        params=_teacher_c2_params(),
        weight=-10.0,
    )
    robot_obstacle_clearance = RewTerm(
        func=mdp.robot_obstacle_clearance_penalty,
        params={"activation_distance_m": 0.05, **_teacher_robot_obstacle_params()},
        weight=-4.0,
    )
    robot_obstacle_collision = RewTerm(
        func=mdp.robot_obstacle_collision_penalty,
        params=_teacher_robot_obstacle_params(),
        weight=-10.0,
    )


@configclass
class AffordanceTeacherTerminationsCfg(AffordanceAwareTerminationsCfg):
    forbidden_region_contact = DoneTerm(
        func=mdp.forbidden_region_contact,
        params=_teacher_c1_params(),
    )
    protected_region_collision = DoneTerm(
        func=mdp.protected_region_collision,
        params=_teacher_c2_params(),
    )
    robot_obstacle_collision = DoneTerm(
        func=mdp.robot_obstacle_collision,
        params=_teacher_robot_obstacle_params(),
    )


@configclass
class AffordanceAwareClutterEnvCfg(Clutter6DEnvCfg):
    """Part-aware hammer task with one strict pose-success definition."""

    scene: Clutter6DSceneCfg = Clutter6DSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: AffordanceAwareObservationsCfg = AffordanceAwareObservationsCfg()
    rewards: AffordanceAwareRewardsCfg = AffordanceAwareRewardsCfg()
    terminations: AffordanceAwareTerminationsCfg = AffordanceAwareTerminationsCfg()
    clutter_asset_source: str = "domino"
    # This now controls only obstacle/contact difficulty.  Pose success is
    # fixed to XY + height + full SO(3) + five-step dwell at every level.
    curriculum_stage: int = 3
    # None activates every manifest obstacle.  Stages 0/1 override this to
    # zero while keeping the fixed 512x3 obstacle observation tensor.
    active_obstacle_count: int | None = None
    enable_world_model_observation: bool = False

    def _configure_curriculum_stage(self) -> None:
        if self.curriculum_stage not in (0, 1, 2, 3):
            raise ValueError("affordance curriculum_stage must be 0, 1, 2, or 3")

        evaluate_protected = self.curriculum_stage >= 2
        self.active_obstacle_count = 0 if self.curriculum_stage < 2 else None
        self.terminations.reached.params = _goal_params()

        for term_name in (
            "task_success",
            "safe_region_distance",
            "safe_region_progress",
            "first_safe_region_contact",
            "forbidden_region_contact",
        ):
            getattr(self.rewards, term_name).params["evaluate_protected"] = evaluate_protected

        if self.curriculum_stage < 2:
            self.rewards.protected_region_clearance = None
            self.rewards.protected_region_collision = None

        # First learn useful pushing with soft penalties. Stage 2 hardens robot
        # contact; stage 3 also hardens protected-part contact with clutter.
        if self.curriculum_stage < 2:
            self.terminations.forbidden_region_contact = None
        else:
            self.terminations.forbidden_region_contact.params[
                "evaluate_protected"
            ] = evaluate_protected
            self.rewards.forbidden_region_contact.weight = -5.0
        if self.curriculum_stage < 3:
            self.terminations.protected_region_collision = None

    def __post_init__(self) -> None:
        self._configure_curriculum_stage()
        super().__post_init__()
        if self.clutter_asset_source != "domino":
            raise ValueError(
                "AffordanceAwareClutterEnvCfg requires DAPL_CLUTTER_ASSET_SOURCE=domino"
            )


@configclass
class AffordanceHammerXYEnvCfg(AffordanceAwareClutterEnvCfg):
    """Legacy task id; success now uses the same strict full pose as all tasks."""

    curriculum_stage: int = 0


@configclass
class AffordanceHammerYawEnvCfg(AffordanceAwareClutterEnvCfg):
    """Legacy task id; success now uses the same strict full pose as all tasks."""

    curriculum_stage: int = 1


@configclass
class AffordanceHammerPoseEnvCfg(AffordanceAwareClutterEnvCfg):
    """Single-target, no-obstacle proof task with strict full-pose success."""

    curriculum_stage: int = 0

    def __post_init__(self) -> None:
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(HAMMER_PROOF_ARM_JOINT_POS)
        self.scene.robot.init_state.joint_pos = joint_pos
        super().__post_init__()
        # Proof-task exploration must remain a controlled push.  A 0.1-rad
        # relative step combined with unit actor noise flung the hammer before
        # the critic observed useful transitions.
        self.actions.arm_action.scale = 0.03
        # This remains a soft penalty (no stage-0 termination), but touching a
        # non-safe point must not be profitable relative to safe contact.
        self.rewards.forbidden_region_contact.weight = -15.0


@configclass
class AffordanceHammerPoseHardEnvCfg(AffordanceHammerPoseEnvCfg):
    """Learned-push phase: no clutter, with illegal robot contact terminated."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Stage 0 deliberately removes this term while the policy first learns
        # to push.  This follow-up phase keeps the identical single-hammer
        # scene and strict pose goal, but hardens contact outside the safe mask.
        self.terminations.forbidden_region_contact = DoneTerm(
            func=mdp.forbidden_region_contact,
            params=_contact_params(evaluate_protected=False),
        )


@configclass
class AffordanceHammerLowClutterEnvCfg(AffordanceHammerPoseEnvCfg):
    """One distant obstacle with soft semantic constraints for clutter transfer."""

    curriculum_stage: int = 2

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # Give the obstacle encoder a real, non-zero cloud before increasing
        # clutter density.  Keep both illegal robot contact and protected-part
        # clearance as soft costs during this transfer phase.
        self.active_obstacle_count = 1
        self.terminations.forbidden_region_contact = None


@configclass
class AffordanceHammerApproachEnvCfg(AffordanceHammerLowClutterEnvCfg):
    """Learn safe-region approach before widening hammer directions."""

    def __post_init__(self) -> None:
        super().__post_init__()
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(DIRECTIONAL_CLUTTER_ARM_JOINT_POS)
        self.scene.robot.init_state.joint_pos = joint_pos
        # The safe surface starts roughly 0.25--0.28 m from the fingers.  Keep
        # the distance cost linear over that entire interval instead of
        # saturating at the proof-task's 0.10 m pre-contact range.
        self.rewards.safe_region_distance.params["normalization_distance_m"] = 0.30
        # This stage must first discover and retain legal contact.  Keep the
        # semantic constraint soft, make a first legal touch worth +10 after
        # dt scaling, and reward only XY progress produced during that touch.
        self.rewards.forbidden_region_contact.weight = -3.0
        self.rewards.first_safe_region_contact.weight = 100.0
        self.rewards.safe_contact_push_progress = RewTerm(
            func=mdp.safe_contact_planar_goal_progress_reward,
            params={
                "normalization_distance_m": 0.01,
                "command_name": "target_object_pose",
                **_contact_params(),
            },
            weight=20.0,
        )


@configclass
class AffordanceHammerApproachRefineEnvCfg(AffordanceHammerApproachEnvCfg):
    """Low-noise precision and settling phase after safe approach is learned."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.first_safe_region_contact.weight = 50.0
        self.rewards.forbidden_region_contact.weight = -5.0
        self.rewards.safe_contact_push_progress = None
        self.rewards.safe_contact_joint_pose_progress = RewTerm(
            func=mdp.safe_contact_joint_pose_progress_reward,
            params={
                "normalization_pose_error": 1.0,
                **_joint_pose_reward_params(),
                **_contact_params(),
            },
            weight=24.0,
        )
        self.rewards.near_goal_target_motion = RewTerm(
            func=mdp.near_goal_target_motion_penalty,
            params={
                "activation_pose_error": 2.0,
                "linear_speed_scale": 0.03,
                "angular_speed_scale": 0.30,
                **_joint_pose_reward_params(),
            },
            weight=-4.0,
        )
        self.rewards.joint_pose_progress.weight = 24.0
        self.rewards.action_rate.weight = -0.20


@configclass
class AffordanceHammerApproachScratchEnvCfg(AffordanceHammerApproachEnvCfg):
    """From-scratch safe-region reaching and componentwise pushing task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Reaching remains object-centric: the closest DOMINO safe surface is
        # the target, without a hand-authored waypoint or fixed push direction.
        self.rewards.safe_region_distance.weight = -1.0
        self.rewards.safe_region_distance.params["normalization_distance_m"] = 0.30
        self.rewards.safe_region_progress.weight = 12.0
        self.rewards.safe_region_progress.params["normalization_distance_m"] = 0.01
        self.rewards.first_safe_region_contact.weight = 50.0

        # Replace the positive-only XY term and max-combined potential.  Each
        # strict success component now supplies signed progress independently,
        # but only while the robot is legally touching the safe region.
        self.rewards.safe_contact_push_progress = None
        self.rewards.safe_contact_joint_pose_progress = None
        self.rewards.joint_pose_progress = None
        self.rewards.joint_pose_tracking = None
        component_params = {
            "normalization_planar_distance_m": 0.01,
            "normalization_height_m": 0.005,
            "normalization_rotation_rad": 0.05,
            "command_name": "target_object_pose",
            **_contact_params(),
        }
        self.rewards.safe_contact_planar_progress = RewTerm(
            func=mdp.safe_contact_pose_component_progress_reward,
            params={"component": "planar", **component_params},
            weight=20.0,
        )
        self.rewards.safe_contact_height_progress = RewTerm(
            func=mdp.safe_contact_pose_component_progress_reward,
            params={"component": "height", **component_params},
            weight=4.0,
        )
        self.rewards.safe_contact_rotation_progress = RewTerm(
            func=mdp.safe_contact_pose_component_progress_reward,
            params={"component": "rotation", **component_params},
            weight=8.0,
        )
        self.rewards.near_goal_target_motion = RewTerm(
            func=mdp.near_goal_target_motion_penalty,
            params={
                "activation_pose_error": 1.5,
                "linear_speed_scale": 0.03,
                "angular_speed_scale": 0.30,
                **_joint_pose_reward_params(),
            },
            weight=-2.0,
        )
        self.rewards.forbidden_region_contact.weight = -5.0
        self.rewards.action_magnitude.weight = -0.02
        self.rewards.action_rate.weight = -0.05


@configclass
class AffordanceHammerTeacherEnvCfg(AffordanceAwareClutterEnvCfg):
    """Deployment-aligned oracle teacher with auditable C1/C2/C3 events."""

    scene: AffordanceTeacherSceneCfg = AffordanceTeacherSceneCfg(
        num_envs=1024, env_spacing=4.0
    )
    observations: AffordanceTeacherObservationsCfg = (
        AffordanceTeacherObservationsCfg()
    )
    rewards: AffordanceTeacherRewardsCfg = AffordanceTeacherRewardsCfg()
    terminations: AffordanceTeacherTerminationsCfg = (
        AffordanceTeacherTerminationsCfg()
    )
    curriculum_stage: int = 3
    active_obstacle_count: int | None = 2
    activate_clutter_contact_sensors: bool = True
    evaluate_robot_obstacle: bool = True
    evaluate_protected_metric: bool = True
    require_physical_protected_contact: bool = True
    robot_target_sensor_name: str | None = TEACHER_ROBOT_TARGET_SENSOR
    robot_obstacle_sensor_name: tuple[str, ...] | None = (
        TEACHER_ROBOT_OBSTACLE_SENSORS
    )
    target_obstacle_sensor_name: str | None = TEACHER_TARGET_OBSTACLE_SENSOR
    physical_contact_force_threshold_n: float = 0.5
    robot_obstacle_clearance_m: float = 0.005

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # The teacher profiles choose their obstacle count explicitly; do not
        # let the legacy numeric curriculum overwrite it.
        self.active_obstacle_count = 2
        self.rewards.forbidden_region_contact.weight = -25.0

    def __post_init__(self) -> None:
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(DIRECTIONAL_CLUTTER_ARM_JOINT_POS)
        self.scene.robot.init_state.joint_pos = joint_pos
        super().__post_init__()
        self.actions.arm_action.scale = 0.03

        # Reuse the from-scratch, waypoint-free shaping that already solved
        # randomized hammer pushing.  The only new signals are typed safety
        # costs; strict XY/Z/SO(3)+dwell success remains unchanged.
        self.rewards.safe_region_distance.weight = -1.0
        self.rewards.safe_region_distance.params["normalization_distance_m"] = 0.30
        self.rewards.safe_region_progress.weight = 12.0
        self.rewards.safe_region_progress.params["normalization_distance_m"] = 0.01
        self.rewards.first_safe_region_contact.weight = 50.0
        self.rewards.joint_pose_progress = None
        self.rewards.joint_pose_tracking = None
        component_params = {
            "normalization_planar_distance_m": 0.01,
            "normalization_height_m": 0.005,
            "normalization_rotation_rad": 0.05,
            "command_name": "target_object_pose",
            **_contact_params(),
        }
        self.rewards.safe_contact_planar_progress = RewTerm(
            func=mdp.safe_contact_pose_component_progress_reward,
            params={"component": "planar", **component_params},
            weight=20.0,
        )
        self.rewards.safe_contact_height_progress = RewTerm(
            func=mdp.safe_contact_pose_component_progress_reward,
            params={"component": "height", **component_params},
            weight=4.0,
        )
        self.rewards.safe_contact_rotation_progress = RewTerm(
            func=mdp.safe_contact_pose_component_progress_reward,
            params={"component": "rotation", **component_params},
            weight=8.0,
        )
        self.rewards.near_goal_target_motion = RewTerm(
            func=mdp.near_goal_target_motion_penalty,
            params={
                "activation_pose_error": 1.5,
                "linear_speed_scale": 0.03,
                "angular_speed_scale": 0.30,
                **_joint_pose_reward_params(),
            },
            weight=-2.0,
        )
        self.rewards.action_magnitude.weight = -0.02
        self.rewards.action_rate.weight = -0.05


@configclass
class AffordanceHammerTeacherSoftEnvCfg(AffordanceHammerTeacherEnvCfg):
    """All typed costs are soft while the teacher learns useful interaction."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.terminations.forbidden_region_contact = None
        self.terminations.protected_region_collision = None
        self.terminations.robot_obstacle_collision = None


@configclass
class AffordanceHammerTeacherNoClutterSoftEnvCfg(AffordanceHammerTeacherSoftEnvCfg):
    """T0: learn a deployment-aligned safe push before clutter transfer."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # T0 is intentionally a soft curriculum phase.  Illegal contact no
        # longer receives the one-time safe-contact bonus, so a -5 weight is
        # sufficient to distinguish it while still allowing PPO to discover
        # contact-rich pushes.  Later hard profiles retain the -25 weight and
        # immediate termination.
        self.rewards.forbidden_region_contact.weight = -5.0
        self.active_obstacle_count = 0
        self.rewards.protected_region_clearance = None
        self.rewards.protected_region_collision = None
        self.rewards.robot_obstacle_clearance = None
        self.rewards.robot_obstacle_collision = None
        # The manifest still contains parked obstacle slots, but T0 has no
        # active clutter.  Do not let contacts with those inactive assets veto
        # the sparse goal reward.  C1 remains evaluated by task_success.
        self.rewards.task_success.params["evaluate_protected"] = False
        self.rewards.task_success.params[
            "require_physical_protected_contact"
        ] = False
        self.rewards.task_success.params["evaluate_robot_obstacle"] = False
        self.evaluate_protected_metric = False
        self.evaluate_robot_obstacle = False
        self.scene.target_obstacle_contacts = None
        for sensor_name in TEACHER_ROBOT_OBSTACLE_SENSORS:
            setattr(self.scene, sensor_name, None)


@configclass
class AffordanceHammerTeacherFrozenV7NoClutterSoftEnvCfg(
    AffordanceHammerTeacherNoClutterSoftEnvCfg
):
    """Freeze the accepted forward-v7 learning contract for distribution audits.

    The generic T0 profile later gained a dense 10--20 mm forbidden-region
    clearance cost.  That is a useful safety refinement, but it makes a new
    arm-div run incomparable with the archived forward-v7 checkpoints.  This
    profile removes only that post-v7 term.  The historical waypoint-free
    reaching, first-legal-contact, contact-gated XY/Z/SO(3), action, sparse
    success, and soft forbidden-contact terms remain unchanged.

    Inactive-obstacle C2/C3 checks stay disabled by the parent profile.  C1 is
    still enforced by ``forbidden_robot_contact`` in the sparse success term
    and by the soft forbidden-contact reward.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.forbidden_region_clearance = None


@configclass
class AffordanceHammerTeacherDAPLAlignedNoClutterSoftEnvCfg(
    AffordanceHammerTeacherNoClutterSoftEnvCfg
):
    """T0 affordance teacher with DAPL reward shaping and only a C1 delta.

    No contact side, waypoint, yaw half-space, wrench, component-progress, or
    action-regularization heuristic is active.  DAPL's target-centroid
    proximity is replaced by distance to the legal affordance set, and its
    0.1 m goal-reward gate uses that same distance.  The sparse success term
    retains this benchmark's explicitly requested strict XY/Z/SO(3)+dwell and
    C1 validity checks.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # DyWA paper/release robot reset: uniform sampling inside the exact
        # published Franka box.  This replaces the historical fixed
        # directional pose only for the paper-aligned task.
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_uniform_within_bounds,
            mode="reset",
            params={
                "position_lower": (
                    -0.3,
                    -0.4636,
                    -0.2,
                    -2.7432,
                    -0.3335,
                    1.5269,
                    -1.5707963267948966,
                ),
                "position_upper": (
                    0.3,
                    0.5432,
                    0.2,
                    -1.5237,
                    0.3335,
                    2.5744,
                    1.5707963267948966,
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["panda_joint.*"]
                ),
            },
        )
        self.rewards.contact_reward = RewTerm(
            func=mdp.safe_region_ee_distance_tanh,
            params={"std": 0.1, **_contact_params(evaluate_protected=False)},
            weight=1.0,
        )
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.safe_region_gated_object_goal_distance_tanh,
            params={
                "std": 0.6,
                "command_name": "target_object_pose",
                "safe_ee_distance_threshold": 0.1,
                **_contact_params(evaluate_protected=False),
            },
            weight=5.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.safe_region_gated_object_goal_distance_tanh,
            params={
                "std": 0.3,
                "command_name": "target_object_pose",
                "safe_ee_distance_threshold": 0.1,
                **_contact_params(evaluate_protected=False),
            },
            weight=16.0,
        )
        for term_name in (
            "safe_region_distance",
            "safe_region_progress",
            "goal_conditioned_safe_region_distance",
            "goal_conditioned_safe_region_progress",
            "semantic_corridor_approach",
            "semantic_corridor_progress",
            "semantic_geodesic_approach",
            "semantic_geodesic_progress",
            "semantic_vector_field_progress",
            "first_safe_region_contact",
            "safe_contact_push_progress",
            "safe_contact_joint_pose_progress",
            "safe_contact_planar_progress",
            "safe_contact_height_progress",
            "safe_contact_rotation_progress",
            "near_goal_target_motion",
            "joint_pose_progress",
            "joint_pose_tracking",
            "post_contact_joint_pose_tracking",
            "post_contact_joint_pose_improvement",
            "action_magnitude",
            "action_rate",
        ):
            setattr(self.rewards, term_name, None)


@configclass
class AffordanceHammerTeacherDAPLProgressNoClutterSoftEnvCfg(
    AffordanceHammerTeacherDAPLAlignedNoClutterSoftEnvCfg
):
    """No-waypoint T0 teacher driven only by observable transition progress.

    The paper-aligned run established that an absolute goal reward behind the
    10-cm gate can be harvested by remaining near the handle without pushing.
    This profile keeps the same assets, DyWA/DAPL pose distribution, actor
    observation, strict success predicate, and soft C1 contract.  It changes
    only the shaping signal: signed progress to the safe set before contact,
    followed by signed DAPL full-pose progress during C1-legal contact.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        semantic_contact = _contact_params(evaluate_protected=False)
        physical_c1 = _teacher_c1_params(evaluate_protected=False)

        # Absolute proximity/goal-state rewards created a stationary payoff at
        # the handle.  Transition potentials are zero when nothing improves.
        self.rewards.contact_reward = None
        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.rewards.safe_region_progress = RewTerm(
            func=mdp.safe_region_distance_progress_reward,
            params={
                "normalization_distance_m": 0.02,
                **semantic_contact,
            },
            weight=8.0,
        )
        self.rewards.first_safe_region_contact = RewTerm(
            func=mdp.first_safe_region_contact_reward,
            params=physical_c1,
            weight=50.0,
        )
        self.rewards.safe_contact_joint_pose_progress = RewTerm(
            func=mdp.legal_safe_contact_dapl_pose_progress_reward,
            params={
                "normalization_pose_error": 0.02,
                "command_name": "target_object_pose",
                **physical_c1,
            },
            weight=16.0,
        )

        # All T0 C1 consumers use the same physical contract and explicitly
        # skip protected-obstacle evaluation.  Besides matching the task, this
        # avoids separate contact-state cache entries for inactive clutter.
        self.rewards.forbidden_region_contact.params = dict(physical_c1)
        self.rewards.forbidden_region_clearance.params = {
            "activation_distance_m": 0.02,
            **physical_c1,
        }
        self.rewards.task_success.params = {
            "termination_term_name": "reached",
            "evaluate_robot_obstacle": False,
            **physical_c1,
        }


@configclass
class AffordanceHammerTeacherUnifiedProgressNoClutterSoftEnvCfg(
    AffordanceHammerTeacherDAPLProgressNoClutterSoftEnvCfg
):
    """Minimal waypoint-free T0 reward with no instantaneous contact gate.

    The failed DAPL-progress ablation made pose progress observable to PPO only
    on steps that simultaneously satisfied the legal-contact proxy.  An
    unstable first contact therefore removed the pushing signal precisely
    when the policy needed it.  This profile changes only that reward
    contract: safe-set distance progress trains reaching, while one signed
    XY/Z/SO(3) potential trains every object motion toward the strict goal.
    Protected contact and its local clearance band remain soft C1 costs.

    No waypoint, phase latch, desired contact side, absolute pose living
    reward, or actor-observation change is introduced.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # The safe-set progress term inherited above is the complete reaching
        # objective.  Remove the event bonus so contact itself is not a local
        # terminal objective.
        self.rewards.first_safe_region_contact = None

        # Object progress must remain learnable across intermittent contact.
        # Before the hammer moves this signed transition potential is exactly
        # zero, so it neither creates a pre-contact pose cost nor pays a living
        # bonus at the reset pose.
        self.rewards.safe_contact_joint_pose_progress = None
        self.rewards.joint_pose_progress = RewTerm(
            func=mdp.affordance_joint_pose_progress_reward,
            params=_joint_pose_reward_params(),
            weight=16.0,
        )


@configclass
class AffordanceHammerTeacherUnifiedDistanceNoClutterSoftEnvCfg(
    AffordanceHammerTeacherUnifiedProgressNoClutterSoftEnvCfg
):
    """Single-variable T0 ablation with continuous absolute reaching cost.

    Independent evaluation of the transition-progress profile showed that
    exploration noise occasionally reached the handle while the deterministic
    policy retained a large safe-set distance.  This profile changes exactly
    one reward: signed per-step safe-distance progress is replaced by an
    absolute linear distance excess.  Its 0.50-m normalization keeps a useful
    slope throughout the sampled workspace and reaches zero continuously at
    the contact boundary.

    The unified XY/Z/SO(3) object-progress term, strict success predicate,
    actor observations, soft C1 penalties, assets, and pose distribution are
    unchanged.  No waypoint, phase variable, contact latch, or event bonus is
    introduced.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        semantic_contact = _contact_params(evaluate_protected=False)

        self.rewards.safe_region_progress = None
        self.rewards.safe_region_distance = RewTerm(
            func=mdp.safe_region_distance_penalty,
            params={
                "normalization_distance_m": 0.50,
                **semantic_contact,
            },
            weight=-2.0,
        )


@configclass
class AffordanceHammerTeacherDistanceDAPLGoalNoClutterSoftEnvCfg(
    AffordanceHammerTeacherUnifiedDistanceNoClutterSoftEnvCfg
):
    """Continuous reaching plus DAPL current-state object-pose shaping.

    The absolute safe-distance profile learned deterministic reaching but its
    signed one-step pose potential did not turn intermittent contact into a
    stable push.  This controlled follow-up changes only the object-motion
    shaping family: the signed joint-pose progress term is replaced by DAPL's
    reported coarse/fine full-pose tracking kernels and weights.  Both terms
    use the current, actor-recoverable safe distance as the original 0.10-m
    gate.  The positive tracking scores have no entry penalty and make a pose
    improvement persist in the return instead of paying for one transition.

    The v5 absolute reaching cost remains the sole safe-approach objective, so
    DAPL's positive proximity living reward is deliberately not restored.
    There is no waypoint, contact event bonus, latch, or hidden phase.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        semantic_contact = _contact_params(evaluate_protected=False)

        self.rewards.joint_pose_progress = None
        self.rewards.object_goal_tracking = RewTerm(
            func=mdp.safe_region_gated_object_goal_distance_tanh,
            params={
                "std": 0.6,
                "command_name": "target_object_pose",
                "safe_ee_distance_threshold": 0.10,
                **semantic_contact,
            },
            weight=5.0,
        )
        self.rewards.object_goal_tracking_fine_grained = RewTerm(
            func=mdp.safe_region_gated_object_goal_distance_tanh,
            params={
                "std": 0.3,
                "command_name": "target_object_pose",
                "safe_ee_distance_threshold": 0.10,
                **semantic_contact,
            },
            weight=16.0,
        )


@configclass
class AffordanceHammerTeacherInitialRelativeDAPLGoalNoClutterSoftEnvCfg(
    AffordanceHammerTeacherDistanceDAPLGoalNoClutterSoftEnvCfg
):
    """Zero-centered persistent DAPL pose shaping for T0 pushing.

    v6 proved that the uncentered current-state score pays a positive living
    reward while the hand parks near a stationary hammer.  This profile makes
    the single conceptual correction: replace those two absolute terms by the
    identical weighted DAPL score minus its reset-time value.  Unchanged pose
    earns zero, sustained improvement earns a persistent positive return, and
    regression earns a persistent negative return.

    The reference is an episode-constant scalar, not a waypoint, phase latch,
    contact event, desired push side, or actor input.  Reaching, C1 costs,
    observations, PPO, assets, pose distribution, and strict success remain
    identical to v6.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        semantic_contact = _contact_params(evaluate_protected=False)

        self.rewards.object_goal_tracking = None
        self.rewards.object_goal_tracking_fine_grained = None
        self.rewards.initial_relative_dapl_pose_score = RewTerm(
            func=mdp.initial_relative_dapl_pose_score_reward,
            params={
                "command_name": "target_object_pose",
                "safe_ee_distance_threshold": 0.10,
                "coarse_standard_deviation": 0.6,
                "fine_standard_deviation": 0.3,
                "coarse_weight": 5.0,
                "fine_weight": 16.0,
                **semantic_contact,
            },
            weight=1.0,
        )


@configclass
class AffordanceHammerTeacherPositiveInitialRelativeDAPLGoalNoClutterSoftEnvCfg(
    AffordanceHammerTeacherInitialRelativeDAPLGoalNoClutterSoftEnvCfg
):
    """Rectify the v7 initial-relative score to preserve push exploration.

    v7 learned reaching and demonstrated legal force transfer, but most early
    pushes worsened the randomly sampled full pose.  Its signed score then
    made avoiding all contact preferable to exploratory pushing.  This
    controlled profile changes only that sign treatment: stationary and worse
    poses both earn zero from the goal term, while improvements retain the
    identical persistent DAPL score and scale.

    No reward term, observation, gate, waypoint, phase state, PPO parameter,
    manifest, success predicate, or C1 coefficient is added or changed.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score.params[
            "positive_only"
        ] = True


@configclass
class AffordanceHammerTeacherPositiveInitialRelativeJointGoalNoClutterSoftEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeDAPLGoalNoClutterSoftEnvCfg
):
    """Use one strict-aligned bottleneck error for persistent pose credit.

    v8 increased contact exploration but its DAPL ``position + rotation / 5``
    scalar still allowed orientation improvement to compensate for pushing XY
    away from the goal. This controlled profile changes only that score
    geometry: reset-relative positive credit now uses the normalized joint
    XY/Z/SO(3) smooth maximum shared with the strict task metrics.

    Reaching, the 0.10-m gate, reward count and weight, actor observations,
    PPO, C1 costs, manifests, and terminal predicate remain unchanged.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score.params[
            "coarse_weight"
        ] = -1.0


@configclass
class AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoClutterSoftEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalNoClutterSoftEnvCfg
):
    """Restore the declared 0.10 residual-action scale for the DAPL range.

    v9 retained the 0.03 scale introduced for the tightly initialized
    6--10-cm proof task even after switching to the wide DyWA joint reset and
    DAPL >=15-cm target displacement.  Its model-200 audit reaches a legal
    safe contact in only 5.47% of held-out episodes and never learns a push.
    This controlled profile changes only the simulator action scale back to
    the 0.10 value in the shared environment/source-of-truth contract.

    Rewards, PPO exploration noise, observations, reset distribution,
    manifests, C1 costs, and strict success are inherited unchanged.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action.scale = 0.10


@configclass
class AffordanceHammerTeacherSignedInitialRelativeJointGoalAction010NoClutterSoftEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoClutterSoftEnvCfg
):
    """Penalize wrong-direction joint-pose regression at the v10 action scale.

    v10 restores enough action authority to reach and move the hammer, but its
    positive-only reset-relative term gives an unchanged zero to the majority
    of pushes that worsen the strict-aligned joint pose error.  This controlled
    profile changes only that rectification: improvement remains positive,
    stationary state remains zero, and regression becomes bounded negative
    credit under the identical reset normalization.

    The action scale, five reward terms and weights, safe-distance gate,
    observations, PPO, manifests, C1 costs, and strict terminal predicate are
    inherited unchanged.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score.params[
            "positive_only"
        ] = False


@configclass
class AffordanceHammerTeacherLeakySignedInitialRelativeJointGoalAction010NoClutterSoftEnvCfg(
    AffordanceHammerTeacherSignedInitialRelativeJointGoalAction010NoClutterSoftEnvCfg
):
    """Retain push exploration with a 0.25 wrong-direction feedback slope.

    v10's zero regression slope preserves object motion but does not control
    its direction, whereas v11's unit slope converges toward less contact and
    less motion.  This profile changes only that scalar interpolation.  The
    positive-improvement branch and every other task/policy setting remain
    identical to v11.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score.params[
            "regression_scale"
        ] = 0.25


@configclass
class AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoClutterSoftEnvCfg
):
    """Test whether C1 learning costs prevent the base pushing skill.

    This is an ablation, not a selectable safe teacher.  It keeps v10's
    movement-preserving reward, 0.10 action scale, strict XY/Z/SO(3)+dwell
    termination, observations, PPO, reset distribution, and manifests.  The
    single conceptual factor switched off is C1 as a learning objective:
    both dense C1 costs are absent and sparse pose success is independent of
    the still-recorded violation history.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.forbidden_region_contact = None
        self.rewards.forbidden_region_clearance = None
        self.rewards.task_success = RewTerm(
            func=mdp.termination_success_reward,
            params={"termination_term_name": "reached"},
            weight=2000.0,
        )


@configclass
class AffordanceHammerTeacherWeightedComponentProgressAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoC1DiagnosticEnvCfg
):
    """Remove the strict-normalized smooth-max shaping bottleneck.

    v13 proves that removing C1 learning costs restores object interaction but
    not goal-directed pushing.  Its reset-relative smooth maximum exposes only
    the current worst normalized XY/Z/SO(3) component, so useful improvement
    in another component is nearly invisible at the wide DAPL pose range.

    This diagnostic changes only that goal-shaping scalar.  It reuses the
    accepted forward teacher's 20/4/8 signed component-progress weighting in
    one simultaneous term.  The strict joint terminal predicate is unchanged;
    there is no waypoint, contact latch, hidden phase, or actor-input change.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score = None
        self.rewards.weighted_component_pose_progress = RewTerm(
            func=mdp.affordance_weighted_component_pose_progress_reward,
            params={
                "safe_ee_distance_threshold": 0.10,
                "normalization_planar_distance_m": 0.01,
                "normalization_height_m": 0.005,
                "normalization_rotation_rad": 0.05,
                "planar_weight": 20.0,
                "height_weight": 4.0,
                "rotation_weight": 8.0,
                "command_name": "target_object_pose",
                **_contact_params(evaluate_protected=False),
            },
            weight=1.0,
        )


@configclass
class AffordanceHammerTeacherPositiveComponentImprovementAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoC1DiagnosticEnvCfg
):
    """Expose every goal component without penalizing exploratory contact.

    v13 preserves interaction but its joint smooth maximum masks improvements
    in non-maximum components. v14 exposes the components but its signed
    transition reward makes entering contact costly. This single-variable
    diagnostic combines the two supported properties: XY/Z/SO(3) are compared
    independently with their reset values, regressions receive zero, and the
    20/4/8 weighted average remains bounded in [0, 1].

    Every other T0 setting remains inherited from v13: no C1 learning cost,
    strict joint success, 0.10 action scale, one 10-cm current-distance gate,
    identical observations/PPO/manifests, and no waypoint, latch, or phase.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score = None
        self.rewards.positive_reference_component_pose_improvement = RewTerm(
            func=mdp.affordance_positive_reference_component_pose_improvement_reward,
            params={
                "safe_ee_distance_threshold": 0.10,
                "reference_planar_error_floor_m": 0.02,
                "reference_height_error_floor_m": 0.01,
                "reference_rotation_error_floor_rad": 0.10,
                "planar_weight": 20.0,
                "height_weight": 4.0,
                "rotation_weight": 8.0,
                "command_name": "target_object_pose",
                **_contact_params(evaluate_protected=False),
            },
            weight=1.0,
        )


@configclass
class AffordanceHammerTeacherParetoPoseImprovementAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoC1DiagnosticEnvCfg
):
    """Require joint goal improvement without a signed contact-entry cost.

    v15 confirms that summing independently rectified components rewards
    transient progress in one component while the joint pose deteriorates.
    This diagnostic replaces only that scalar combination by the minimum of
    reset-relative XY improvement, reset-relative full-SO(3) improvement, and
    the remaining strict Z margin. Thus all three task conditions must agree,
    while exploratory regressions still receive zero rather than a sustained
    negative contact cost.

    The reward remains one bounded scalar. The inherited actor, critic, PPO,
    reaching term, strict success, action scale, manifests, and no-C1 control
    are unchanged; there is no waypoint, latch, or hidden phase.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score = None
        self.rewards.positive_reference_pareto_pose_improvement = RewTerm(
            func=mdp.affordance_positive_reference_pareto_pose_improvement_reward,
            params={
                "safe_ee_distance_threshold": 0.10,
                "reference_planar_error_floor_m": 0.02,
                "reference_rotation_error_floor_rad": 0.10,
                "support_height_tolerance_m": 0.01,
                "command_name": "target_object_pose",
                **_contact_params(evaluate_protected=False),
            },
            weight=1.0,
        )


@configclass
class AffordanceHammerTeacherDyWAKeypointPotentialAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherPositiveInitialRelativeJointGoalAction010NoC1DiagnosticEnvCfg
):
    """Use DyWA's joint keypoint potential as the sole object-goal shaping.

    The previous no-C1 controls show that separate component credit either
    rewards partial regressions or becomes too sparse when combined with a
    hard minimum. DyWA instead transforms corresponding object keypoints at
    the current and goal poses, averages one exponential potential, and uses
    its discounted temporal difference. This exposes every pose component in
    one geometry without a waypoint, contact gate, latch, or hidden phase.

    The exponential constants and 0.16 per-step coefficient follow DyWA's
    released arm-diverse configuration. The shaping discount is matched to
    this PPO runner's 0.99 discount; RewardManager's 0.1-s dt therefore maps
    the configured weight 1.6 back to the 0.16 coefficient.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.initial_relative_dapl_pose_score = None
        self.rewards.dywa_keypoint_pose_potential = RewTerm(
            func=mdp.affordance_dywa_keypoint_pose_potential_reward,
            params={
                "potential_amplitude": 0.302,
                "potential_distance_rate": 243.12,
                "potential_exponential_base": 0.995,
                "potential_discount": 0.99,
                "command_name": "target_object_pose",
                "target_cfg": SceneEntityCfg("target"),
                "obstacles_cfg": SceneEntityCfg("obstacles"),
            },
            weight=1.6,
        )


@configclass
class AffordanceHammerTeacherDyWABBoxFullScaleAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherDyWAKeypointPotentialAction010NoC1DiagnosticEnvCfg
):
    """Port the released DyWA bbox-keypoint exponential branch at full scale.

    v17 proves reaching and object motion but its 512-surface-point, effective
    0.16-scale potential does not control the goal pose.  The released DyWA
    path instead uses canonical bounding-box keypoints and, in the exponential
    branch, does not apply the configured ``pot_coef=0.16``.  This diagnostic
    changes only that single object-goal shaping implementation: eight AABB
    corners, DyWA's 0.995 temporal factor, and an effective per-step scale of
    1.0 (weight 10 under RewardManager's 0.1-s dt).

    Actor/critic inputs, PPO, safe-distance reaching, strict success, action
    scale, manifests, and the no-C1 control are inherited unchanged.  There is
    still one object-goal scalar and no waypoint, gate, latch, or hidden phase.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        term = self.rewards.dywa_keypoint_pose_potential
        term.params["potential_discount"] = 0.995
        term.params["use_bounding_box_keypoints"] = True
        term.weight = 10.0


@configclass
class AffordanceHammerTeacherDyWAMatchedPotentialsAction010NoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherDyWABBoxFullScaleAction010NoC1DiagnosticEnvCfg
):
    """Match DyWA's temporal form for both reaching and object-goal shaping.

    v18 keeps an absolute safe-distance cost whose episode magnitude dominates
    its temporal object-goal potential. DyWA instead uses matched exponential
    temporal potentials for both relations, with hand-object amplitude 0.2 of
    object-goal. This diagnostic changes only the reaching term to that form,
    substituting safe-affordance distance for centroid distance. The signal is
    still continuous from the initial state and adds no waypoint or phase.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.safe_region_distance = None
        self.rewards.safe_region_progress = RewTerm(
            func=mdp.safe_region_dywa_distance_potential_reward,
            params={
                "potential_amplitude": 0.0604,
                "potential_distance_rate": 243.12,
                "potential_exponential_base": 0.995,
                "potential_discount": 0.995,
                **_contact_params(evaluate_protected=False),
            },
            weight=10.0,
        )


@configclass
class AffordanceHammerTeacherDyWAMatchedPotentialsCartesianNoC1DiagnosticEnvCfg(
    AffordanceHammerTeacherDyWAMatchedPotentialsAction010NoC1DiagnosticEnvCfg
):
    """Change only the policy control space to bounded Cartesian delta pose."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions = CartesianDeltaPoseActionsCfg()


@configclass
class AffordanceHammerTeacherGoalSideSoftEnvCfg(
    AffordanceHammerTeacherNoClutterSoftEnvCfg
):
    """T0 ablation with object-centric, goal-compatible safe-side shaping."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        params = {
            "contact_distance_m": 0.010,
            "minimum_safe_score": 0.25,
            "side_band_m": 0.015,
            "minimum_goal_displacement_m": 0.020,
            "command_name": "target_object_pose",
            "safe_radius_m": None,
            "protected_radius_m": None,
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.goal_conditioned_safe_region_distance = RewTerm(
            func=mdp.goal_conditioned_safe_region_distance_penalty,
            params={"normalization_distance_m": 0.20, **params},
            weight=-1.0,
        )
        self.rewards.goal_conditioned_safe_region_progress = RewTerm(
            func=mdp.goal_conditioned_safe_region_progress_reward,
            params={"normalization_distance_m": 0.020, **params},
            weight=8.0,
        )


@configclass
class AffordanceHammerTeacherGoalSideSafetySoftEnvCfg(
    AffordanceHammerTeacherGoalSideSoftEnvCfg
):
    """Goal-compatible safe-set shaping with the complete soft C1 cost."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # Keep exploration contact-rich and non-terminating, but make the
        # protected/neutral shortcut more expensive than the legal-contact
        # bonus.  This combines the two independently audited soft curricula;
        # it does not add a waypoint or change the actor observation contract.
        self.rewards.forbidden_region_contact.weight = -25.0


@configclass
class AffordanceHammerTeacherGoalSideExploreSoftEnvCfg(
    AffordanceHammerTeacherGoalSideSafetySoftEnvCfg
):
    """Endpoint discovery using one unambiguous goal-conditioned safe set."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Do not simultaneously optimize distance to an arbitrary safe point:
        # on side pushes that set's closest point is often on the wrong side of
        # the handle and conflicts with the trailing-side potential.  Contact
        # pose progress and the legal one-time contact bonus remain unchanged.
        self.rewards.safe_region_distance = None
        self.rewards.safe_region_progress = None


@configclass
class AffordanceHammerTeacherSemanticCorridorSoftEnvCfg(
    AffordanceHammerTeacherGoalSideSafetySoftEnvCfg
):
    """Waypoint-free safe-contact routing around semantic target obstacles."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Replace every Euclidean approach objective with one point-cloud
        # navigation potential.  The target set is still the trailing safe
        # surface, while the open route is penalized when non-safe target
        # points occlude it.  No route point enters the actor observation.
        self.rewards.safe_region_distance = None
        self.rewards.safe_region_progress = None
        self.rewards.goal_conditioned_safe_region_distance = None
        self.rewards.goal_conditioned_safe_region_progress = None
        params = {
            "normalization_distance_m": 0.20,
            "contact_distance_m": 0.010,
            "corridor_contact_clearance_m": 0.010,
            "corridor_activation_clearance_m": 0.030,
            "corridor_body_radius_m": 0.030,
            "corridor_barrier_floor": None,
            "obstruction_weight": 1.0,
            "corridor_samples": 9,
            "corridor_start_fraction": 0.10,
            "corridor_end_fraction": 0.85,
            "minimum_safe_score": 0.25,
            "side_band_m": 0.015,
            "minimum_goal_displacement_m": 0.020,
            "command_name": "target_object_pose",
            "safe_radius_m": None,
            "protected_radius_m": None,
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.semantic_corridor_approach = RewTerm(
            func=mdp.goal_conditioned_semantic_corridor_penalty,
            params=params,
            weight=-1.0,
        )
        self.rewards.semantic_corridor_progress = RewTerm(
            func=mdp.goal_conditioned_semantic_corridor_progress_reward,
            params={"normalization_potential": 0.05, **params},
            weight=12.0,
        )


@configclass
class AffordanceHammerTeacherSemanticCorridorBarrierSoftEnvCfg(
    AffordanceHammerTeacherSemanticCorridorSoftEnvCfg
):
    """Semantic corridor navigation with a finite near-contact log barrier."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # v32's linear obstruction saturated exactly where the path became
        # dangerous, so direct-distance progress could still pay an unsafe
        # shortcut.  The log barrier keeps the same geometric contract but
        # makes vanishing free margin dominate the shortcut incentive.
        self.rewards.semantic_corridor_approach.params[
            "corridor_barrier_floor"
        ] = 0.05
        self.rewards.semantic_corridor_progress.params[
            "corridor_barrier_floor"
        ] = 0.05


@configclass
class AffordanceHammerTeacherSemanticGeodesicSoftEnvCfg(
    AffordanceHammerTeacherSemanticCorridorSoftEnvCfg
):
    """Waypoint-free shortest legal route over the target semantic cloud."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.semantic_corridor_approach = None
        self.rewards.semantic_corridor_progress = None
        params = {
            "normalization_distance_m": 0.20,
            "contact_distance_m": 0.010,
            "route_contact_clearance_m": 0.010,
            "route_activation_clearance_m": 0.030,
            "route_body_radius_m": 0.030,
            "route_detour_margin_m": 0.020,
            "route_barrier_floor": 0.01,
            "obstruction_weight": 1.0,
            "route_candidates": 12,
            "route_segment_samples": 7,
            "route_obstacle_samples": 96,
            "minimum_safe_score": 0.25,
            "side_band_m": 0.015,
            "minimum_goal_displacement_m": 0.020,
            "command_name": "target_object_pose",
            "safe_radius_m": None,
            "protected_radius_m": None,
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.semantic_geodesic_approach = RewTerm(
            func=mdp.goal_conditioned_semantic_geodesic_penalty,
            params=params,
            weight=-1.0,
        )
        self.rewards.semantic_geodesic_progress = RewTerm(
            func=mdp.goal_conditioned_semantic_geodesic_progress_reward,
            params={"normalization_potential": 0.05, **params},
            weight=12.0,
        )


@configclass
class AffordanceHammerTeacherSemanticGeodesicConservativeSoftEnvCfg(
    AffordanceHammerTeacherSemanticGeodesicSoftEnvCfg
):
    """Geodesic shaping whose progress preserves route-quality differences."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # With 0.05, a small direct advance and a true lateral detour both hit
        # the clipped +1 progress ceiling.  A wider scale keeps the geodesic's
        # route-quality ordering visible to PPO without changing its geometry.
        self.rewards.semantic_geodesic_progress.params[
            "normalization_potential"
        ] = 0.20


@configclass
class AffordanceHammerTeacherSemanticVectorFieldSoftEnvCfg(
    AffordanceHammerTeacherSemanticGeodesicConservativeSoftEnvCfg
):
    """Geodesic teacher with local, reward-only semantic flow alignment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        params = dict(self.rewards.semantic_geodesic_approach.params)
        self.rewards.semantic_vector_field_progress = RewTerm(
            func=mdp.goal_conditioned_semantic_vector_field_progress_reward,
            params={"normalization_displacement_m": 0.010, **params},
            # This term is active only while the direct path is semantically
            # obstructed.  Ordinary directions therefore retain the accepted
            # baseline reward, while detours receive a dense signed signal.
            weight=8.0,
        )


@configclass
class AffordanceHammerTeacherSemanticVectorFieldExploreSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldSoftEnvCfg
):
    """Unambiguous detour discovery with a stronger local flow signal."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # v36 showed that the field was geometrically correct but contributed
        # only about one quarter of the simultaneous scalar-geodesic progress.
        # Remove that competing scalar and make a 5 mm aligned displacement a
        # full local-progress event.  This remains reward-only and does not
        # alter the actor observation contract.
        self.rewards.semantic_geodesic_progress = None
        self.rewards.semantic_vector_field_progress.params[
            "normalization_displacement_m"
        ] = 0.005
        self.rewards.semantic_vector_field_progress.weight = 40.0


@configclass
class AffordanceHammerTeacherSemanticVectorFieldScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldExploreSoftEnvCfg
):
    """From-scratch vector-field ablation with the baseline soft-C1 weight."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # GoalSideSafety raises this to -25 for behavior-preserving refinement.
        # That confounds a from-scratch comparison against T0, whose intended
        # exploratory soft-contact weight is -5.  Match T0 exactly here; only
        # the pre-contact approach potential is allowed to differ.
        self.rewards.forbidden_region_contact.weight = -5.0


@configclass
class AffordanceHammerTeacherSemanticVectorFieldBalancedScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldScratchSoftEnvCfg
):
    """From-scratch field with matched direct and stronger detour progress."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # The global field weight is 40 and displacement normalization is
        # 5 mm.  A 0.15 direct scale therefore matches the original Euclidean
        # progress slope (12 / 10 mm), while obstructed routes retain the
        # stronger lateral slope needed to escape the protected-head basin.
        self.rewards.semantic_vector_field_progress.params[
            "direct_route_scale"
        ] = 0.15


@configclass
class AffordanceHammerTeacherSemanticVectorFieldCommittedScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldBalancedScratchSoftEnvCfg
):
    """Keep detour-strength guidance through the final safe-contact edge."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # A route commonly becomes direct after the hand reaches the support
        # ring but before it reaches the safe surface.  Dropping immediately
        # to the 0.15 direct scale creates a discontinuous reward ridge there.
        # Latch only the reward strength for the remainder of the episode; no
        # route node, phase bit, or commitment enters the actor observation.
        self.rewards.semantic_vector_field_progress.params[
            "latch_detour_until_contact"
        ] = True


@configclass
class AffordanceHammerTeacherSemanticVectorFieldClearanceBlendScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldBalancedScratchSoftEnvCfg
):
    """Use an observation-recoverable smooth transition into direct motion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep full detour-strength guidance while the straight hand-to-safe
        # segment remains within 1--4 cm of the inflated protected region.
        # Above 4 cm of clearance the scale reaches the matched direct-route
        # baseline (0.15).  The blend depends only on the current semantic
        # geometry; it adds neither a waypoint nor an unobserved phase latch.
        self.rewards.semantic_vector_field_progress.params[
            "direct_route_activation_clearance_m"
        ] = 0.040


@configclass
class AffordanceHammerTeacherSemanticVectorFieldClearanceRecoveryScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldClearanceBlendScratchSoftEnvCfg
):
    """Recover feasibility whenever the sampled semantic route set is empty."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # If every direct/ring route is illegal, do not fall back to the
        # shortest (usually direct) illegal edge.  Reward the outward
        # clearance gradient until a legal route exists again.  This field is
        # fully recoverable from the current semantic cloud and hand state.
        self.rewards.semantic_vector_field_progress.params[
            "recover_illegal_route"
        ] = True


@configclass
class AffordanceHammerTeacherSemanticPotentialConsistentRecoveryScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldClearanceRecoveryScratchSoftEnvCfg
):
    """Use the local recovery field only when scalar potential also descends."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # v47's outward-plus-tangent direction resolves force cancellation but
        # is not itself a conservative field: PPO can circulate around the
        # protected geometry and collect positive alignment indefinitely.  In
        # this profile the local field only gates positive progress of a
        # bounded scalar navigation potential.  Potential ascent remains fully
        # negative, so every closed state-space loop has non-positive shaping
        # return.  The larger weight compensates for taking an exact bounded
        # potential difference instead of normalizing each 5 mm displacement.
        self.rewards.semantic_vector_field_progress.params[
            "require_potential_descent"
        ] = True
        self.rewards.semantic_vector_field_progress.params["potential_scale"] = 1.0
        self.rewards.semantic_vector_field_progress.params[
            "descent_gate_floor"
        ] = 0.25
        self.rewards.semantic_vector_field_progress.weight = 800.0


@configclass
class AffordanceHammerTeacherSemanticPotentialConsistentCalibratedScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticPotentialConsistentRecoveryScratchSoftEnvCfg
):
    """Match conservative semantic shaping to the accepted reaching baseline."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # On the identical known-success trajectory, the original Euclidean
        # terms integrate to -1.18 distance / +28.15 progress, whereas v48's
        # first conservative scale integrates to -43.02 / +7.47.  Rounded,
        # auditable weights below restore that accepted balance without
        # changing geometry or the no-positive-cycle invariant.  The known
        # v47 circulation remains strongly negative (about -59 total shaping).
        self.rewards.semantic_geodesic_approach.weight = -0.03
        self.rewards.semantic_vector_field_progress.weight = 3000.0


@configclass
class AffordanceHammerTeacherLexicographicPotentialScratchSoftEnvCfg(
    AffordanceHammerTeacherSemanticVectorFieldClearanceRecoveryScratchSoftEnvCfg
):
    """Shortest feasible safe-contact shaping without clearance trade-offs."""

    def __post_init__(self) -> None:
        super().__post_init__()
        lexicographic_params = {
            "lexicographic_feasibility": True,
            "lexicographic_length_scale_m": 0.20,
            "lexicographic_violation_scale_m": 0.01,
            # The route starts at an actual sampled hand-surface point.  A
            # second 3 cm spherical inflation falsely marked a known C1-clean
            # contact trajectory infeasible (0.7 mm reported clearance).  A
            # 2 cm sweep proxy removes that duplicated inflation while hard
            # C1 continues to use the full hand cloud plus PhysX whole-arm
            # contact.
            "route_body_radius_m": 0.02,
            # Match route feasibility to the audited semantic C1 contact
            # boundary.  The earlier 10 mm route threshold added an
            # undocumented 2 mm buffer on top of the 8 mm C1 predicate and
            # still rejected a known zero-C1 legal push by 0.7 mm.
            "route_contact_clearance_m": 0.008,
        }
        self.rewards.semantic_geodesic_approach.params.update(
            lexicographic_params
        )
        self.rewards.semantic_vector_field_progress.params.update(
            lexicographic_params
        )
        # Match the runner's gamma.  Unlike the earlier alignment-gated
        # difference, this term cannot prefer an early descent followed by a
        # late retreat over a trajectory with the same endpoint.  A separate
        # state cost remains after the shaping baseline, so parking at a high
        # cost is still worse than reaching the feasible safe-contact set.
        self.rewards.semantic_vector_field_progress.params[
            "potential_shaping_discount"
        ] = 0.99
        self.rewards.semantic_vector_field_progress.params[
            "require_potential_descent"
        ] = False
        self.rewards.semantic_geodesic_approach.weight = -12.0
        self.rewards.semantic_vector_field_progress.weight = 1000.0


@configclass
class AffordanceHammerTeacherWrenchLexicographicPotentialScratchSoftEnvCfg(
    AffordanceHammerTeacherLexicographicPotentialScratchSoftEnvCfg
):
    """Goal-wrench-aware safe contact manifold for joint XY+yaw pushing."""

    def __post_init__(self) -> None:
        super().__post_init__()
        wrench_params = {
            # Translation selects the trailing safe surface as before.  The
            # signed moment arm then breaks ties toward the side that can
            # generate the requested yaw, without ever admitting a non-safe
            # point or changing the actor observation contract.
            "yaw_moment_weight": 1.0,
            "yaw_activation_rad": 0.10,
        }
        self.rewards.semantic_geodesic_approach.params.update(wrench_params)
        self.rewards.semantic_vector_field_progress.params.update(wrench_params)
        # Normalize XY and SO(3) progress at roughly their strict-success
        # tolerances: 20 / 0.01 versus 20 / 0.05 gives a 5:1 metric ratio,
        # matching the 2 cm versus 0.1 rad terminal contract.
        self.rewards.safe_contact_rotation_progress.weight = 20.0


@configclass
class AffordanceHammerTeacherWrenchSeparatedContactGateScratchSoftEnvCfg(
    AffordanceHammerTeacherWrenchLexicographicPotentialScratchSoftEnvCfg
):
    """Contact-gated C1 teacher with a more discriminative wrench manifold."""

    def __post_init__(self) -> None:
        super().__post_init__()
        contact_gate_params = {
            # The 15 mm score band in v52 admitted contact points with
            # conflicting yaw moments.  A 10 mm band with a moderate 1.5x yaw
            # weight improves the actual hammer's signed moment distribution
            # without collapsing the manifold to a brittle single point.
            "side_band_m": 0.010,
            "yaw_moment_weight": 1.5,
            # Once any currently observable contact is both safe and legal,
            # stop moving the approach anchor and let object-pose progress
            # determine the push.  No hidden phase or waypoint is introduced.
            "gate_on_legal_safe_contact": True,
        }
        self.rewards.semantic_geodesic_approach.params.update(
            contact_gate_params
        )
        self.rewards.semantic_vector_field_progress.params.update(
            contact_gate_params
        )
        for term in (
            self.rewards.safe_contact_planar_progress,
            self.rewards.safe_contact_height_progress,
            self.rewards.safe_contact_rotation_progress,
        ):
            term.params["require_legal_contact"] = True
        # C1 is still a soft constraint in this discovery run, but its dense
        # barrier acts only in the 10--12 mm boundary layer.  It can no longer
        # buy arbitrary extra clearance by opposing a legal handle contact.
        self.rewards.forbidden_region_clearance.params[
            "activation_distance_m"
        ] = 0.012


@configclass
class AffordanceHammerTeacherFullSafeContactGateScratchSoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedContactGateScratchSoftEnvCfg
):
    """Contact-gated teacher whose approach objective is the full safe set.

    The v52/v53 weighted wrench subset coupled translation and yaw into one
    pre-contact anchor.  The real hammer audit shows that this single-push
    contact condition is asymmetric and sometimes infeasible.  Here the dense
    navigation reward has exactly one job: reach any semantically safe and
    C1-legal handle point.  Once contact is made it remains gated off, while
    the observable relative goal and signed point relations let PPO discover
    translation/rotation-producing contact changes from object-pose progress.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        safe_set_params = {
            "use_full_safe_region": True,
            # Avoid computing a goal-wrench preference that is deliberately
            # excluded from the reward-side approach set.
            "yaw_moment_weight": 0.0,
        }
        self.rewards.semantic_geodesic_approach.params.update(safe_set_params)
        self.rewards.semantic_vector_field_progress.params.update(
            safe_set_params
        )


@configclass
class AffordanceHammerTeacherFullSafeJointPoseCostScratchSoftEnvCfg(
    AffordanceHammerTeacherFullSafeContactGateScratchSoftEnvCfg
):
    """Full-safe teacher with a continuous simultaneous pose-state cost."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # v54 reaches a legal handle in every held-out scene but can remain in
        # a one-sign yaw mode because one-step pose progress provides no state
        # cost after that error has been accumulated.  Penalize the current
        # *joint* XY/Z/SO(3) error instead of adding separate objectives.  The
        # broad scales keep the signal dense over the 8 cm / +/-0.23 rad train
        # distribution; strict success remains 2 cm / 1 cm / 0.1 rad + dwell.
        self.rewards.joint_pose_tracking = RewTerm(
            func=mdp.affordance_joint_pose_tracking_cost,
            params={
                "planar_scale_m": 0.08,
                "height_scale_m": 0.01,
                "rotation_scale_rad": 0.20,
                "smooth_max_temperature": 0.25,
                "command_name": "target_object_pose",
                "target_cfg": SceneEntityCfg("target"),
            },
            weight=-5.0,
        )


@configclass
class AffordanceHammerTeacherFullSafePostContactPoseCostScratchSoftEnvCfg(
    AffordanceHammerTeacherFullSafeContactGateScratchSoftEnvCfg
):
    """Full-safe navigation followed by latched joint-pose control.

    v55 applied pose cost throughout the pre-contact phase and degraded both
    safe reaching and C1 behavior.  This profile preserves v54's proven
    full-safe approach.  The transition into the post-contact objective is
    the first observable legal safe contact, after which navigation stays off
    and the simultaneous XY/Z/SO(3) cost remains active while the hand changes
    contact side.  No waypoint or additional actor observation is introduced.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.semantic_vector_field_progress.params[
            "latch_after_legal_safe_contact"
        ] = True
        # Retain only a weak state cost toward the full safe set.  The strong
        # vector field supplies pre-contact exploration and is permanently
        # retired after first legal contact.
        self.rewards.semantic_geodesic_approach.weight = -1.0
        self.rewards.post_contact_joint_pose_tracking = RewTerm(
            func=mdp.post_legal_safe_contact_joint_pose_tracking_cost,
            params={
                "planar_scale_m": 0.08,
                "height_scale_m": 0.01,
                "rotation_scale_rad": 0.20,
                "smooth_max_temperature": 0.25,
                "command_name": "target_object_pose",
                **_teacher_c1_params(evaluate_protected=False),
                "obstacles_cfg": SceneEntityCfg("obstacles"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=-5.0,
        )


@configclass
class AffordanceHammerTeacherFullSafePostContactImprovementScratchSoftEnvCfg(
    AffordanceHammerTeacherFullSafeContactGateScratchSoftEnvCfg
):
    """Full-safe reaching with a contact-relative joint-pose objective.

    The first legal safe-contact pose defines zero improvement.  This keeps
    the contact transition attractive while retaining a bounded signed signal
    for simultaneous XY/Z/SO(3) improvement afterward.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.rewards.semantic_vector_field_progress.params[
            "latch_after_legal_safe_contact"
        ] = True
        self.rewards.semantic_geodesic_approach.weight = -1.0
        self.rewards.post_contact_joint_pose_improvement = RewTerm(
            func=mdp.post_legal_safe_contact_joint_pose_improvement_reward,
            params={
                "planar_scale_m": 0.08,
                "height_scale_m": 0.01,
                "rotation_scale_rad": 0.20,
                "smooth_max_temperature": 0.25,
                "normalization_cost": 0.25,
                "command_name": "target_object_pose",
                **_teacher_c1_params(evaluate_protected=False),
                "obstacles_cfg": SceneEntityCfg("obstacles"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=5.0,
        )


@configclass
class AffordanceHammerTeacherYawCompatiblePostContactImprovementScratchSoftEnvCfg(
    AffordanceHammerTeacherFullSafePostContactImprovementScratchSoftEnvCfg
):
    """State-conditioned safe contact set for bilateral XY+yaw control.

    The full-safe v57 objective restores reaching but leaves the initial
    contact side ambiguous, allowing PPO to collapse to one yaw sign.  While
    the observable yaw error is material, v58 retains all safe points within
    10 mm of the best compatible signed moment arm.  It switches back to the
    translation-support safe set only once yaw is nearly resolved.  This is a
    reward-side set potential derived from current geometry and relative goal,
    not a waypoint, action target, hidden phase, or relaxed success contract.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        yaw_safe_set_params = {
            "use_full_safe_region": False,
            "use_yaw_compatible_safe_region": True,
            "yaw_side_band_m": 0.010,
            "minimum_yaw_error_rad": 0.020,
            "yaw_activation_rad": 0.10,
            # Translation is used only as the below-threshold fallback.  The
            # active yaw set is lexicographic, so no weighted wrench sum can
            # recreate the v52/v53 translation--rotation cancellation basin.
            "yaw_moment_weight": 0.0,
        }
        self.rewards.semantic_geodesic_approach.params.update(
            yaw_safe_set_params
        )
        self.rewards.semantic_vector_field_progress.params.update(
            yaw_safe_set_params
        )


@configclass
class AffordanceHammerTeacherYawPositivePostContactImprovementScratchSoftEnvCfg(
    AffordanceHammerTeacherYawCompatiblePostContactImprovementScratchSoftEnvCfg
):
    """Broad correct-sign safe set that preserves contact reachability.

    v58's 10 mm near-best moment set is geometrically valid but narrows the
    asymmetric hammer handle enough to create a strong yaw-sign reachability
    imbalance.  v59 instead retains the full safe halfspace above a 2 mm
    positive signed-moment floor.  The exact same recoverable geometry and
    relative goal define the set; all success and hard-C1 predicates are
    unchanged.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        broad_yaw_safe_set_params = {
            "yaw_compatible_selection_mode": "positive_halfspace",
            "yaw_minimum_compatibility_m": 0.002,
        }
        self.rewards.semantic_geodesic_approach.params.update(
            broad_yaw_safe_set_params
        )
        self.rewards.semantic_vector_field_progress.params.update(
            broad_yaw_safe_set_params
        )


@configclass
class AffordanceHammerTeacherNoClutterSafetySoftEnvCfg(
    AffordanceHammerTeacherNoClutterSoftEnvCfg
):
    """Intermediate C1 curriculum with dense/binary costs but no termination."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.rewards.forbidden_region_contact.weight = -25.0


@configclass
class AffordanceHammerTeacherNoClutterHardEnvCfg(AffordanceHammerTeacherEnvCfg):
    """C1 audit: no clutter and hard whole-arm forbidden-contact failure."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.active_obstacle_count = 0
        self.terminations.protected_region_collision = None
        self.terminations.robot_obstacle_collision = None
        self.rewards.protected_region_clearance = None
        self.rewards.protected_region_collision = None
        self.rewards.robot_obstacle_clearance = None
        self.rewards.robot_obstacle_collision = None
        # This is a C1-only task.  Inactive obstacle slots must not suppress a
        # legal terminal reward merely because their contact sensors still
        # exist in the shared teacher scene schema.
        self.rewards.task_success.params["evaluate_protected"] = False
        self.rewards.task_success.params[
            "require_physical_protected_contact"
        ] = False
        self.rewards.task_success.params["evaluate_robot_obstacle"] = False
        self.evaluate_protected_metric = False
        self.evaluate_robot_obstacle = False
        self.scene.target_obstacle_contacts = None
        for sensor_name in TEACHER_ROBOT_OBSTACLE_SENSORS:
            setattr(self.scene, sensor_name, None)


@configclass
class AffordanceHammerTeacherOneObstacleSoftEnvCfg(
    AffordanceHammerTeacherSoftEnvCfg
):
    """Base one-obstacle transfer profile with every safety event kept soft."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.active_obstacle_count = 1


@configclass
class AffordanceHammerTeacherC2SoftEnvCfg(
    AffordanceHammerTeacherOneObstacleSoftEnvCfg
):
    """Soft C2 transfer without a simultaneous robot--clutter objective."""

    # Isolate protected-target-to-clutter contact from accidental blocker
    # settling. The target hammer remains fully dynamic.
    kinematic_active_obstacles: bool = True

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.rewards.robot_obstacle_clearance = None
        self.rewards.robot_obstacle_collision = None
        self.rewards.task_success.params["evaluate_robot_obstacle"] = False
        self.evaluate_robot_obstacle = False
        for sensor_name in TEACHER_ROBOT_OBSTACLE_SENSORS:
            setattr(self.scene, sensor_name, None)


@configclass
class AffordanceHammerTeacherC3SoftEnvCfg(
    AffordanceHammerTeacherOneObstacleSoftEnvCfg
):
    """Soft C3 routing transfer without a simultaneous protected sweep cost."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.rewards.protected_region_clearance = None
        self.rewards.protected_region_collision = None
        self.rewards.task_success.params["evaluate_protected"] = False
        self.rewards.task_success.params["require_physical_protected_contact"] = False
        self.evaluate_protected_metric = False
        self.scene.target_obstacle_contacts = None


@configclass
class AffordanceHammerTeacherOneObstacleHardEnvCfg(AffordanceHammerTeacherEnvCfg):
    """Base one-obstacle hard profile used by typed diagnostic tasks."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.active_obstacle_count = 1


@configclass
class AffordanceHammerTeacherC2EnvCfg(AffordanceHammerTeacherOneObstacleHardEnvCfg):
    """C2 audit: protected target parts must not sweep into the blocker."""

    kinematic_active_obstacles: bool = True

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # Keep C1 active (the robot must still push legally), but remove C3 so
        # the diagnostic's failure label is uniquely target-protected/clutter.
        self.terminations.robot_obstacle_collision = None
        self.rewards.robot_obstacle_clearance = None
        self.rewards.robot_obstacle_collision = None
        self.rewards.task_success.params["evaluate_robot_obstacle"] = False
        self.evaluate_robot_obstacle = False
        for sensor_name in TEACHER_ROBOT_OBSTACLE_SENSORS:
            setattr(self.scene, sensor_name, None)


@configclass
class AffordanceHammerTeacherC3EnvCfg(AffordanceHammerTeacherOneObstacleHardEnvCfg):
    """C3 audit: the whole Franka must route around the blocker."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # Keep C1 active, but remove C2 so a robot/blocker event cannot be
        # conflated with a protected-part sweep event in this ablation.
        self.terminations.protected_region_collision = None
        self.rewards.protected_region_clearance = None
        self.rewards.protected_region_collision = None
        self.rewards.task_success.params["evaluate_protected"] = False
        self.rewards.task_success.params["require_physical_protected_contact"] = False
        self.evaluate_protected_metric = False
        self.scene.target_obstacle_contacts = None


@configclass
class AffordanceHammerTeacherWrenchSeparatedC2SoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedContactGateScratchSoftEnvCfg
):
    """v53 shaping plus one active C2 blocker and no simultaneous C3 cost."""

    def _configure_curriculum_stage(self) -> None:
        # Bypass the no-clutter branch inherited by the v53 T0 profile while
        # retaining the common soft-termination contract.  The later v53
        # __post_init__ chain still installs the identical contact gate,
        # lexicographic route potential, narrow C1 barrier, and legal-contact
        # pose progress used to train the source checkpoint.
        AffordanceHammerTeacherSoftEnvCfg._configure_curriculum_stage(self)
        self.active_obstacle_count = 1
        self.rewards.robot_obstacle_clearance = None
        self.rewards.robot_obstacle_collision = None
        self.rewards.task_success.params["evaluate_robot_obstacle"] = False
        self.evaluate_robot_obstacle = False
        for sensor_name in TEACHER_ROBOT_OBSTACLE_SENSORS:
            setattr(self.scene, sensor_name, None)


@configclass
class AffordanceHammerTeacherWrenchSeparatedC2ClearanceProgressSoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedC2SoftEnvCfg
):
    """C2 soft task with one waypoint-free protected-clearance potential."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.rewards.protected_region_clearance_progress = RewTerm(
            func=mdp.protected_region_clearance_progress_reward,
            params={
                "activation_distance_m": 0.05,
                "potential_discount": 0.99,
                **_contact_params(),
            },
            weight=20.0,
        )


@configclass
class AffordanceHammerTeacherWrenchSeparatedC2ClearanceProgressHardEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedC2ClearanceProgressSoftEnvCfg
):
    """Keep the accepted C2 shaping and turn typed violations into failures.

    This is the second half of the declared soft-to-hard curriculum.  It does
    not replace the reward, manifest, observation, success predicate, or
    action space used by the soft stage; it only reinstates the C1 and C2
    termination predicates after the policy has learned legal safe contact and
    basic pushing.  C3 remains disabled for the isolated C2 proof.
    """

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.terminations.forbidden_region_contact = DoneTerm(
            func=mdp.forbidden_region_contact,
            params=_teacher_c1_params(),
        )
        self.terminations.protected_region_collision = DoneTerm(
            func=mdp.protected_region_collision,
            params=_teacher_c2_params(),
        )


@configclass
class AffordanceHammerTeacherWrenchSeparatedC2GeodesicProgressSoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedC2SoftEnvCfg
):
    """C2 soft task with one full-protected-body geodesic progress term."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.rewards.protected_region_geodesic_progress = RewTerm(
            func=mdp.protected_region_geodesic_progress_reward,
            params={
                "normalization_distance_m": 0.01,
                "route_contact_clearance_m": 0.005,
                "route_detour_margin_m": 0.020,
                "route_candidates": 12,
                "route_segment_samples": 9,
                "route_obstacle_samples": 128,
                "critical_sweep_samples": 5,
                "body_sweep_samples": 5,
                "body_aabb_clearance_m": 0.0005,
                **_contact_params(),
            },
            # Single-variable replacement for v67's equally weighted local
            # protected-clearance progress term.
            weight=20.0,
        )


@configclass
class AffordanceHammerTeacherWrenchSeparatedC2LateralEscapeProgressSoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedC2SoftEnvCfg
):
    """C2 soft task that rewards only obstruction-clearing lateral motion."""

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        self.rewards.protected_region_lateral_escape_progress = RewTerm(
            func=mdp.protected_region_geodesic_progress_reward,
            params={
                "progress_mode": "blocked_lateral_escape",
                "normalization_distance_m": 0.005,
                "route_contact_clearance_m": 0.005,
                "route_detour_margin_m": 0.020,
                "route_candidates": 12,
                "route_segment_samples": 9,
                "route_obstacle_samples": 128,
                "critical_sweep_samples": 5,
                "body_sweep_samples": 5,
                "body_aabb_clearance_m": 0.0005,
                **_contact_params(),
            },
            # Match v70's sole experimental term weight.  The only changed
            # variable is its projection: forward goal motion is exactly zero
            # while the protected sweep remains blocked.
            weight=20.0,
        )


@configclass
class AffordanceHammerTeacherWrenchSeparatedC3SoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedContactGateScratchSoftEnvCfg
):
    """v53 shaping plus one active C3 blocker and no simultaneous C2 cost."""

    def _configure_curriculum_stage(self) -> None:
        AffordanceHammerTeacherSoftEnvCfg._configure_curriculum_stage(self)
        self.active_obstacle_count = 1
        self.rewards.protected_region_clearance = None
        self.rewards.protected_region_collision = None
        self.rewards.task_success.params["evaluate_protected"] = False
        self.rewards.task_success.params[
            "require_physical_protected_contact"
        ] = False
        self.evaluate_protected_metric = False
        self.scene.target_obstacle_contacts = None


@configclass
class AffordanceHammerTeacherWrenchSeparatedCombinedSoftEnvCfg(
    AffordanceHammerTeacherWrenchSeparatedContactGateScratchSoftEnvCfg
):
    """v53 shaping with both C2 and C3 costs over two active blockers."""

    def _configure_curriculum_stage(self) -> None:
        AffordanceHammerTeacherSoftEnvCfg._configure_curriculum_stage(self)
        self.active_obstacle_count = 2


@configclass
class AffordanceHammerDirectionalClutterEnvCfg(AffordanceHammerPoseEnvCfg):
    """Balanced multi-direction pushes with two path-adjacent obstacles."""

    curriculum_stage: int = 2

    def _configure_curriculum_stage(self) -> None:
        super()._configure_curriculum_stage()
        # This is the second clutter-transfer phase: retain soft semantic
        # costs while requiring the policy to encode two real obstacle clouds.
        # A later task can harden these terms after deterministic success is
        # robust across every direction bin.
        self.active_obstacle_count = 2
        self.terminations.forbidden_region_contact = None

    def __post_init__(self) -> None:
        super().__post_init__()
        joint_pos = dict(self.scene.robot.init_state.joint_pos)
        joint_pos.update(DIRECTIONAL_CLUTTER_ARM_JOINT_POS)
        self.scene.robot.init_state.joint_pos = joint_pos
        self.rewards.safe_region_distance.params["normalization_distance_m"] = 0.30


@configclass
class AffordanceHammerAvoidEnvCfg(AffordanceAwareClutterEnvCfg):
    curriculum_stage: int = 2


@configclass
class AffordanceHammerClutterEnvCfg(AffordanceAwareClutterEnvCfg):
    curriculum_stage: int = 3


class AffordanceAwareClutterEnv(Clutter6DEnv):
    """Clutter environment that reports semantic constraint metrics."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.episode_affordance_violation_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_c1_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_c1_hand_semantic_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_c1_hand_neutral_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_c1_hand_protected_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_c1_arm_physical_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_c2_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_c3_violation_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.episode_constrained_reached_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_legal_safe_contact_buf = torch.zeros_like(
            self.episode_affordance_violation_buf
        )
        self.total_constrained_episodes = 0
        self.total_constrained_successes = 0
        self.total_affordance_violation_episodes = 0
        self.total_c1_violation_episodes = 0
        self.total_c1_hand_semantic_violation_episodes = 0
        self.total_c1_hand_neutral_violation_episodes = 0
        self.total_c1_hand_protected_violation_episodes = 0
        self.total_c1_arm_physical_violation_episodes = 0
        self.total_c2_violation_episodes = 0
        self.total_c3_violation_episodes = 0

    def _metric_contact_state(self):
        """Evaluate all constraints enabled by this task with one cache key."""

        return mdp.domino_affordance_contact_state(
            self,
            contact_distance_m=0.010,
            protected_clearance_m=0.005,
            robot_obstacle_clearance_m=float(
                getattr(self.cfg, "robot_obstacle_clearance_m", 0.005)
            ),
            physical_contact_force_threshold_n=float(
                getattr(self.cfg, "physical_contact_force_threshold_n", 0.5)
            ),
            evaluate_protected=bool(
                getattr(
                    self.cfg,
                    "evaluate_protected_metric",
                    self.cfg.curriculum_stage >= 2,
                )
            ),
            evaluate_robot_obstacle=bool(
                getattr(self.cfg, "evaluate_robot_obstacle", False)
            ),
            require_physical_protected_contact=bool(
                getattr(self.cfg, "require_physical_protected_contact", False)
            ),
            robot_target_sensor_name=getattr(
                self.cfg, "robot_target_sensor_name", None
            ),
            robot_obstacle_sensor_name=getattr(
                self.cfg, "robot_obstacle_sensor_name", None
            ),
            target_obstacle_sensor_name=getattr(
                self.cfg, "target_obstacle_sensor_name", None
            ),
        )

    def _reset_idx(self, env_ids):
        """Preserve terminal contact state before Isaac Lab auto-resets assets."""

        cached = getattr(self, "_domino_affordance_state_cache", None)
        reset_buf = getattr(self, "reset_buf", None)
        if cached is not None and reset_buf is not None:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            # ``_reset_idx`` is also called by explicit/initial resets.  Only
            # the auto-reset ids in reset_buf have a terminal state to retain.
            terminal_ids = ids[reset_buf[ids].bool()]
            if terminal_ids.numel() > 0 and not hasattr(
                self, "_pre_reset_affordance_violation"
            ):
                self._pre_reset_affordance_violation = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
                self._pre_reset_affordance_state_valid = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.bool
                )
            if terminal_ids.numel() > 0:
                # Reset events immediately advance each environment to its
                # next manifest task.  Preserve the scene/task identity here
                # so vectorized evaluation attributes the terminal episode to
                # the scene that actually produced it, not the post-reset one.
                if hasattr(self, "_clutter_scene_indices"):
                    if not hasattr(
                        self, "_episode_clutter_scene_indices_before_reset"
                    ):
                        self._episode_clutter_scene_indices_before_reset = (
                            torch.zeros(
                                self.num_envs,
                                device=self.device,
                                dtype=torch.long,
                            )
                        )
                    self._episode_clutter_scene_indices_before_reset[
                        terminal_ids
                    ] = self._clutter_scene_indices[terminal_ids]
                if hasattr(self, "_clutter_task_indices"):
                    if not hasattr(
                        self, "_episode_clutter_task_indices_before_reset"
                    ):
                        self._episode_clutter_task_indices_before_reset = (
                            torch.zeros(
                                self.num_envs,
                                device=self.device,
                                dtype=torch.long,
                            )
                        )
                    self._episode_clutter_task_indices_before_reset[
                        terminal_ids
                    ] = self._clutter_task_indices[terminal_ids]
                state = self._metric_contact_state()
                c1 = state["forbidden_robot_contact"]
                c1_hand_semantic = state["forbidden_hand_contact"]
                c1_hand_neutral = state["neutral_hand_contact"]
                c1_hand_protected = state["protected_hand_contact"]
                c1_arm_physical = state["arm_target_physical_contact"]
                c2 = state["protected_obstacle_collision"]
                c3 = state["robot_obstacle_collision"]
                violation = c1 | c2 | c3
                self._pre_reset_affordance_violation[terminal_ids] = violation[
                    terminal_ids
                ]
                if not hasattr(self, "_pre_reset_legal_safe_contact"):
                    self._pre_reset_legal_safe_contact = torch.zeros_like(
                        violation
                    )
                self._pre_reset_legal_safe_contact[terminal_ids] = state[
                    "legal_safe_robot_contact"
                ][terminal_ids]
                if not hasattr(self, "_pre_reset_c1_violation"):
                    self._pre_reset_c1_violation = torch.zeros_like(violation)
                    self._pre_reset_c1_hand_semantic_violation = torch.zeros_like(
                        violation
                    )
                    self._pre_reset_c1_hand_neutral_violation = torch.zeros_like(
                        violation
                    )
                    self._pre_reset_c1_hand_protected_violation = torch.zeros_like(
                        violation
                    )
                    self._pre_reset_c1_arm_physical_violation = torch.zeros_like(
                        violation
                    )
                    self._pre_reset_c2_violation = torch.zeros_like(violation)
                    self._pre_reset_c3_violation = torch.zeros_like(violation)
                self._pre_reset_c1_violation[terminal_ids] = c1[terminal_ids]
                self._pre_reset_c1_hand_semantic_violation[terminal_ids] = (
                    c1_hand_semantic[terminal_ids]
                )
                self._pre_reset_c1_hand_neutral_violation[terminal_ids] = (
                    c1_hand_neutral[terminal_ids]
                )
                self._pre_reset_c1_hand_protected_violation[terminal_ids] = (
                    c1_hand_protected[terminal_ids]
                )
                self._pre_reset_c1_arm_physical_violation[terminal_ids] = (
                    c1_arm_physical[terminal_ids]
                )
                self._pre_reset_c2_violation[terminal_ids] = c2[terminal_ids]
                self._pre_reset_c3_violation[terminal_ids] = c3[terminal_ids]
                self._pre_reset_affordance_state_valid[terminal_ids] = True
                target = self.scene["target"]
                goal = self.command_manager.get_command("target_object_pose")
                target_pos_env = (
                    target.data.root_pos_w[:, :3] - self.scene.env_origins
                )
                planar_error = torch.linalg.vector_norm(
                    goal[:, :2] - target_pos_env[:, :2], dim=1
                )
                height_error = torch.abs(goal[:, 2] - target_pos_env[:, 2])
                quat_dot = torch.sum(
                    target.data.root_quat_w * goal[:, 3:7], dim=1
                )
                rotation_error = 2.0 * torch.acos(
                    torch.clamp(torch.abs(quat_dot), max=1.0)
                )
                signed_yaw_error = mdp.affordance_signed_yaw_goal_error(self)
                if not hasattr(self, "_pre_reset_target_goal_planar_error"):
                    self._pre_reset_target_goal_planar_error = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    self._pre_reset_target_goal_rotation_error = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    self._pre_reset_target_goal_height_error = torch.zeros(
                        self.num_envs, device=self.device
                    )
                    self._pre_reset_target_goal_signed_yaw_error = torch.zeros(
                        self.num_envs, device=self.device
                    )
                self._pre_reset_target_goal_planar_error[terminal_ids] = (
                    planar_error[terminal_ids]
                )
                self._pre_reset_target_goal_rotation_error[terminal_ids] = (
                    rotation_error[terminal_ids]
                )
                self._pre_reset_target_goal_height_error[terminal_ids] = (
                    height_error[terminal_ids]
                )
                self._pre_reset_target_goal_signed_yaw_error[terminal_ids] = (
                    signed_yaw_error[terminal_ids]
                )
        super()._reset_idx(env_ids)

    def step(self, action):
        observations, reward, terminated, truncated, info = super().step(action)
        reached = self.termination_manager.get_term("reached")
        state = self._metric_contact_state()
        target = self.scene["target"]
        goal = self.command_manager.get_command("target_object_pose")
        target_pos_env = target.data.root_pos_w[:, :3] - self.scene.env_origins
        position_delta = goal[:, :3] - target_pos_env
        planar_error = torch.linalg.vector_norm(position_delta[:, :2], dim=1)
        height_error = torch.abs(position_delta[:, 2])
        quaternion_dot = torch.sum(target.data.root_quat_w * goal[:, 3:7], dim=1)
        rotation_error = 2.0 * torch.acos(
            torch.clamp(torch.abs(quaternion_dot), max=1.0)
        )
        joint_pose_error = mdp.affordance_joint_pose_error(
            self,
            **_joint_pose_reward_params(),
        )
        c1 = state["forbidden_robot_contact"]
        c1_hand_semantic = state["forbidden_hand_contact"]
        c1_hand_neutral = state["neutral_hand_contact"]
        c1_hand_protected = state["protected_hand_contact"]
        c1_arm_physical = state["arm_target_physical_contact"]
        c2 = state["protected_obstacle_collision"]
        c3 = state["robot_obstacle_collision"]
        violation = c1 | c2 | c3
        legal_safe_contact = state["legal_safe_robot_contact"]
        episode_ended = terminated | truncated
        valid_terminal_state = getattr(
            self,
            "_pre_reset_affordance_state_valid",
            torch.zeros_like(episode_ended),
        )
        if torch.any(episode_ended & valid_terminal_state):
            violation = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_affordance_violation,
                violation,
            )
            c1 = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c1_violation,
                c1,
            )
            c1_hand_semantic = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c1_hand_semantic_violation,
                c1_hand_semantic,
            )
            c1_hand_neutral = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c1_hand_neutral_violation,
                c1_hand_neutral,
            )
            c1_hand_protected = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c1_hand_protected_violation,
                c1_hand_protected,
            )
            c1_arm_physical = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c1_arm_physical_violation,
                c1_arm_physical,
            )
            c2 = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c2_violation,
                c2,
            )
            c3 = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_c3_violation,
                c3,
            )
            legal_safe_contact = torch.where(
                episode_ended & valid_terminal_state,
                self._pre_reset_legal_safe_contact,
                legal_safe_contact,
            )
        self.episode_affordance_violation_buf |= violation
        self.episode_c1_violation_buf |= c1
        self.episode_c1_hand_semantic_violation_buf |= c1_hand_semantic
        self.episode_c1_hand_neutral_violation_buf |= c1_hand_neutral
        self.episode_c1_hand_protected_violation_buf |= c1_hand_protected
        self.episode_c1_arm_physical_violation_buf |= c1_arm_physical
        self.episode_c2_violation_buf |= c2
        self.episode_c3_violation_buf |= c3
        self.episode_constrained_reached_buf |= reached
        self.episode_legal_safe_contact_buf |= legal_safe_contact
        if torch.any(episode_ended):
            constrained = (
                self.episode_constrained_reached_buf
                & ~self.episode_affordance_violation_buf
            )
            ended_count = int(episode_ended.sum().item())
            success_count = int((constrained & episode_ended).sum().item())
            violation_count = int(
                (self.episode_affordance_violation_buf & episode_ended).sum().item()
            )
            c1_count = int((self.episode_c1_violation_buf & episode_ended).sum().item())
            c1_hand_semantic_count = int(
                (
                    self.episode_c1_hand_semantic_violation_buf & episode_ended
                ).sum().item()
            )
            c1_hand_neutral_count = int(
                (
                    self.episode_c1_hand_neutral_violation_buf & episode_ended
                ).sum().item()
            )
            c1_hand_protected_count = int(
                (
                    self.episode_c1_hand_protected_violation_buf & episode_ended
                ).sum().item()
            )
            c1_arm_physical_count = int(
                (
                    self.episode_c1_arm_physical_violation_buf & episode_ended
                ).sum().item()
            )
            c2_count = int((self.episode_c2_violation_buf & episode_ended).sum().item())
            c3_count = int((self.episode_c3_violation_buf & episode_ended).sum().item())
            self.total_constrained_episodes += ended_count
            self.total_constrained_successes += success_count
            self.total_affordance_violation_episodes += violation_count
            self.total_c1_violation_episodes += c1_count
            self.total_c1_hand_semantic_violation_episodes += (
                c1_hand_semantic_count
            )
            self.total_c1_hand_neutral_violation_episodes += c1_hand_neutral_count
            self.total_c1_hand_protected_violation_episodes += (
                c1_hand_protected_count
            )
            self.total_c1_arm_physical_violation_episodes += c1_arm_physical_count
            self.total_c2_violation_episodes += c2_count
            self.total_c3_violation_episodes += c3_count
            self._episode_constrained_success_before_reset = constrained.clone()
            self._episode_affordance_violation_before_reset = (
                self.episode_affordance_violation_buf.clone()
            )
            self._episode_c1_violation_before_reset = self.episode_c1_violation_buf.clone()
            self._episode_c1_hand_semantic_violation_before_reset = (
                self.episode_c1_hand_semantic_violation_buf.clone()
            )
            self._episode_c1_hand_neutral_violation_before_reset = (
                self.episode_c1_hand_neutral_violation_buf.clone()
            )
            self._episode_c1_hand_protected_violation_before_reset = (
                self.episode_c1_hand_protected_violation_buf.clone()
            )
            self._episode_c1_arm_physical_violation_before_reset = (
                self.episode_c1_arm_physical_violation_buf.clone()
            )
            self._episode_c2_violation_before_reset = self.episode_c2_violation_buf.clone()
            self._episode_c3_violation_before_reset = self.episode_c3_violation_buf.clone()
            self._episode_legal_safe_contact_before_reset = (
                self.episode_legal_safe_contact_buf.clone()
            )
            self.episode_affordance_violation_buf[episode_ended] = False
            self.episode_c1_violation_buf[episode_ended] = False
            self.episode_c1_hand_semantic_violation_buf[episode_ended] = False
            self.episode_c1_hand_neutral_violation_buf[episode_ended] = False
            self.episode_c1_hand_protected_violation_buf[episode_ended] = False
            self.episode_c1_arm_physical_violation_buf[episode_ended] = False
            self.episode_c2_violation_buf[episode_ended] = False
            self.episode_c3_violation_buf[episode_ended] = False
            self.episode_constrained_reached_buf[episode_ended] = False
            self.episode_legal_safe_contact_buf[episode_ended] = False
            if hasattr(self, "_pre_reset_affordance_state_valid"):
                self._pre_reset_affordance_state_valid[episode_ended] = False
        log = self.extras.setdefault("log", {})
        log["Metrics/affordance/planar_error"] = planar_error
        log["Metrics/affordance/height_error"] = height_error
        log["Metrics/affordance/rotation_error"] = rotation_error
        log["Metrics/affordance/joint_pose_error"] = joint_pose_error
        log["Metrics/affordance/target_linear_speed"] = torch.linalg.vector_norm(
            target.data.root_lin_vel_w, dim=1
        )
        log["Metrics/affordance/target_angular_speed"] = torch.linalg.vector_norm(
            target.data.root_ang_vel_w, dim=1
        )
        log["Metrics/affordance/minimum_safe_distance"] = state[
            "minimum_safe_distance"
        ]
        yaw_set_count = getattr(
            self, "_affordance_yaw_compatible_selected_count", None
        )
        yaw_safe_count = getattr(
            self, "_affordance_yaw_compatible_safe_count", None
        )
        if yaw_set_count is not None and yaw_safe_count is not None:
            log["Metrics/affordance/yaw_contact_set_active_fraction"] = getattr(
                self,
                "_affordance_yaw_compatible_contact_active",
                torch.zeros_like(yaw_set_count, dtype=torch.bool),
            ).float()
            log["Metrics/affordance/yaw_contact_set_fraction_of_safe"] = (
                yaw_set_count.float()
                / torch.clamp(yaw_safe_count.float(), min=1.0)
            )
            log["Metrics/affordance/yaw_contact_set_fallback_fraction"] = getattr(
                self,
                "_affordance_yaw_compatible_selection_fallback",
                torch.zeros_like(yaw_set_count, dtype=torch.bool),
            ).float()
        log["Metrics/affordance/safe_contact_fraction"] = state[
            "safe_robot_contact"
        ].float()
        log["Metrics/affordance/legal_safe_contact_fraction"] = state[
            "legal_safe_robot_contact"
        ].float()
        log["Metrics/affordance/forbidden_contact_fraction"] = state[
            "forbidden_robot_contact"
        ].float()
        log["Metrics/affordance/forbidden_hand_contact_fraction"] = state[
            "forbidden_hand_contact"
        ].float()
        log["Metrics/affordance/neutral_hand_contact_fraction"] = state[
            "neutral_hand_contact"
        ].float()
        log["Metrics/affordance/protected_hand_contact_fraction"] = state[
            "protected_hand_contact"
        ].float()
        log["Metrics/affordance/protected_collision_fraction"] = state[
            "protected_obstacle_collision"
        ].float()
        log["Metrics/affordance/robot_obstacle_collision_fraction"] = state[
            "robot_obstacle_collision"
        ].float()
        log["Metrics/affordance/arm_target_physical_contact_fraction"] = state[
            "arm_target_physical_contact"
        ].float()
        log["Metrics/affordance/target_obstacle_physical_contact_fraction"] = state[
            "target_obstacle_physical_contact"
        ].float()
        log["Metrics/affordance/robot_obstacle_clearance"] = state[
            "robot_obstacle_clearance"
        ]
        log["Metrics/affordance/strict_pose_valid_fraction"] = (
            (planar_error < 0.02)
            & (height_error < 0.01)
            & (rotation_error < 0.10)
        ).float()
        log["Metrics/affordance/dwell_success_fraction"] = reached.float()
        log["Metrics/action/absolute_mean"] = torch.abs(action)
        log["Metrics/action/l2"] = torch.sum(torch.square(action), dim=1)
        log["Metrics/action/rate_l2"] = torch.sum(
            torch.square(
                self.action_manager.action - self.action_manager.prev_action
            ),
            dim=1,
        )
        log["Metrics/action/saturation_fraction"] = (
            torch.abs(action) >= 1.0
        ).float()
        denominator = max(self.total_constrained_episodes, 1)
        log["constrained_success_rate"] = (
            self.total_constrained_successes / denominator
        )
        log["affordance_violation_rate"] = (
            self.total_affordance_violation_episodes / denominator
        )
        log["c1_violation_rate"] = self.total_c1_violation_episodes / denominator
        log["c1_hand_semantic_violation_rate"] = (
            self.total_c1_hand_semantic_violation_episodes / denominator
        )
        log["c1_hand_neutral_violation_rate"] = (
            self.total_c1_hand_neutral_violation_episodes / denominator
        )
        log["c1_hand_protected_violation_rate"] = (
            self.total_c1_hand_protected_violation_episodes / denominator
        )
        log["c1_arm_physical_violation_rate"] = (
            self.total_c1_arm_physical_violation_episodes / denominator
        )
        log["c2_violation_rate"] = self.total_c2_violation_episodes / denominator
        log["c3_violation_rate"] = self.total_c3_violation_episodes / denominator
        return observations, reward, terminated, truncated, info
