"""Manifest-backed Clutter6D environment used for DAPL development."""

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from dapl.scene import ClutterScene, load_scene_manifest

import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.clutter import (
    build_clutter_rigid_assets,
    manifest_asset_resolver,
)
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env import (
    CurriculumCfg,
    NonPrehensileEnv,
    NonPrehensileEnvCfg,
    NonPrehensileSceneCfg,
    RelativeJointPositionActionsCfg,
)


@configclass
class Clutter6DSceneCfg(NonPrehensileSceneCfg):
    """Franka scene with one target and a collection of non-target objects."""

    # Remove the legacy single-object entity inherited from NonPrehensileSceneCfg.
    object: RigidObjectCfg | None = None
    target: RigidObjectCfg | None = None
    obstacles: RigidObjectCollectionCfg | None = None
    # Optional filtered contact reporters used by the affordance-teacher
    # tasks.  Keeping them disabled in the base clutter task avoids changing
    # the DAPL reproduction contract or its simulator cost.
    robot_target_contacts: ContactSensorCfg | None = None
    robot_obstacle_contacts: ContactSensorCfg | None = None
    target_obstacle_contacts: ContactSensorCfg | None = None


@configclass
class Clutter6DCommandsCfg:
    """Commands sourced from the task selected by the reset event."""

    target_object_pose = mdp.ManifestPoseCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        target_asset_name="target",
    )


@configclass
class Clutter6DObservationsCfg:
    """Privileged target-only observation used to establish a trainable baseline."""

    @configclass
    class PolicyCfg(ObsGroup):
        object_cloud = ObsTerm(
            func=mdp.get_object_pointcloud_in_env_frame,
            params={"object_cfg": SceneEntityCfg("target")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        hand_state = ObsTerm(
            func=mdp.hand_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        robot_state = ObsTerm(
            func=mdp.robot_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        previous_action = ObsTerm(func=mdp.last_action)
        rel_goal = ObsTerm(
            func=mdp.rel_pose_goal,
            params={
                "command_name": "target_object_pose",
                "object_cfg": SceneEntityCfg("target"),
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
        )
        phys_params = ObsTerm(
            func=mdp.phys_params,
            params={"object_cfg": SceneEntityCfg("target")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class WorldModelCfg(ObsGroup):
        scene = ObsTerm(
            func=mdp.dapl_physical_scene,
            params={
                "target_cfg": SceneEntityCfg("target"),
                "obstacles_cfg": SceneEntityCfg("obstacles"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    world_model: WorldModelCfg = WorldModelCfg()


@configclass
class Clutter6DDAPLObservationsCfg(Clutter6DObservationsCfg):
    """Paper policy observation: 1,280x7 scene followed by 44-D state."""

    @configclass
    class PolicyCfg(ObsGroup):
        physical_scene = ObsTerm(
            func=mdp.dapl_physical_scene_flattened,
            params={
                "target_cfg": SceneEntityCfg("target"),
                "obstacles_cfg": SceneEntityCfg("obstacles"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
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
        phys_params = ObsTerm(
            func=mdp.phys_params,
            params={"object_cfg": SceneEntityCfg("target")},
        )

        def __post_init__(self) -> None:
            # Stage-2 is the privileged simulation teacher. Sensor corruption is
            # introduced later for the sim-to-real student, not for this policy.
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Clutter6DEventCfg:
    """Reset robot, target, and obstacles before selecting a manifest task."""

    set_clutter_materials = EventTerm(
        func=mdp.set_clutter_material_properties_from_manifest,
        mode="startup",
        params={
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
        },
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_clutter = EventTerm(
        func=mdp.reset_clutter_from_manifest,
        mode="reset",
        params={
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
        },
    )
    # Enabled only by task profiles that explicitly reproduce DyWA's
    # published Franka joint-box initialization.
    reset_robot_joints = None


@configclass
class Clutter6DRewardsCfg:
    """DAPL appendix reward terms and reported weights."""

    task_success = RewTerm(
        func=mdp.clutter_task_success_reward,
        params={
            "command_name": "target_object_pose",
            "position_threshold": 0.05,
            "rotation_threshold": 0.1,
            # The appendix defines these constants but does not report values.
            # Keep the development choice explicit for reproducibility.
            "maximum_obstacle_translation": 0.2,
            "maximum_obstacle_rotation": math.pi,
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
        },
        weight=2000.0,
    )
    contact_reward = RewTerm(
        func=mdp.object_ee_distance_tanh,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("target"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=1.0,
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": 0.6,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": 0.1,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("target"),
        },
        weight=5.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_tanh,
        params={
            "std": 0.3,
            "command_name": "target_object_pose",
            "obj_ee_distance_threshold": 0.1,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("target"),
        },
        weight=16.0,
    )


@configclass
class Clutter6DTerminationsCfg:
    """Paper task completion, drop, and 300-control-step timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reached = DoneTerm(
        func=mdp.clutter_target_reached_goal,
        params={
            "command_name": "target_object_pose",
            "position_threshold": 0.05,
            "rotation_threshold": 0.1,
            "target_cfg": SceneEntityCfg("target"),
        },
    )
    object_dropped = DoneTerm(
        func=mdp.clutter_object_dropped_off_table,
        params={
            "minimum_height": -0.15,
            "target_cfg": SceneEntityCfg("target"),
            "obstacles_cfg": SceneEntityCfg("obstacles"),
        },
    )


@configclass
class Clutter6DEnvCfg(NonPrehensileEnvCfg):
    """Load a versioned manifest and instantiate its assets deterministically."""

    scene: Clutter6DSceneCfg = Clutter6DSceneCfg(num_envs=64, env_spacing=4.0)
    observations: Clutter6DObservationsCfg = Clutter6DObservationsCfg()
    actions: RelativeJointPositionActionsCfg = RelativeJointPositionActionsCfg()
    events: Clutter6DEventCfg = Clutter6DEventCfg()
    commands: Clutter6DCommandsCfg = Clutter6DCommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    rewards: Clutter6DRewardsCfg = Clutter6DRewardsCfg()
    terminations: Clutter6DTerminationsCfg = Clutter6DTerminationsCfg()

    clutter_manifest_path: str | None = None
    clutter_asset_source: str = "dgn"
    clutter_asset_root: str | None = None
    clutter_scenes: tuple[ClutterScene, ...] = ()
    clutter_scene_offset: int = 0
    enable_world_model_observation: bool = True
    activate_clutter_contact_sensors: bool = False
    # Fixed blockers remain visible, collidable and observed, while avoiding
    # unintended zero-action motion from mesh/support-pose discrepancies.
    kinematic_active_obstacles: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        manifest_value = self.clutter_manifest_path or os.environ.get(
            "DAPL_CLUTTER_MANIFEST"
        )
        if manifest_value is None:
            raise ValueError(
                "Clutter6D manifest is unset; set DAPL_CLUTTER_MANIFEST or "
                "cfg.clutter_manifest_path"
            )
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Clutter6D manifest not found: {manifest_path}")

        source = os.environ.get(
            "DAPL_CLUTTER_ASSET_SOURCE", self.clutter_asset_source
        ).lower()
        root = self.clutter_asset_root
        if root is None:
            root_variable = {
                "dgn": "DGN_DATA_ROOT",
                "dapl": "DAPL_DATA_ROOT",
                "domino": "DOMINO_ROOT",
            }.get(source)
            if root_variable is None:
                raise ValueError(
                    "clutter asset source must be 'dgn', 'dapl', or 'domino'"
                )
            root = os.environ.get(root_variable)
        scenes = tuple(load_scene_manifest(manifest_path))
        offset_override = os.environ.get("DAPL_CLUTTER_SCENE_OFFSET")
        if offset_override is not None:
            self.clutter_scene_offset = int(offset_override)
        if self.clutter_scene_offset < 0:
            raise ValueError("clutter_scene_offset must be non-negative")
        if scenes:
            self.clutter_scene_offset %= len(scenes)
        target, obstacles = build_clutter_rigid_assets(
            scenes,
            resolver=manifest_asset_resolver(source, root=root),
            active_obstacle_count=getattr(self, "active_obstacle_count", None),
            kinematic_active_obstacles=self.kinematic_active_obstacles,
            activate_contact_sensors=self.activate_clutter_contact_sensors,
        )
        self.scene.target = target
        self.scene.obstacles = obstacles
        self.clutter_manifest_path = str(manifest_path)
        self.clutter_asset_source = source
        self.clutter_asset_root = None if root is None else str(Path(root).expanduser().resolve())
        self.clutter_scenes = scenes

        observation_override = os.environ.get("DAPL_ENABLE_WORLD_MODEL_OBSERVATION")
        if observation_override is not None:
            self.enable_world_model_observation = observation_override.strip().lower() not in {
                "0",
                "false",
                "no",
            }
        if not self.enable_world_model_observation:
            self.observations.world_model = None

        # dt=1/80 and decimation=8 make 30 seconds exactly 300 policy steps.
        self.episode_length_s = 30.0
        self.visualize_current_object_pose = False


@configclass
class Clutter6DDAPLEnvCfg(Clutter6DEnvCfg):
    """Clutter6D configuration consumed by the frozen-encoder DAPL policy."""

    observations: Clutter6DDAPLObservationsCfg = Clutter6DDAPLObservationsCfg()
    # The physical scene already lives in the policy group, so avoid computing
    # the separate collection-only group a second time on every simulator step.
    enable_world_model_observation: bool = False


class Clutter6DEnv(NonPrehensileEnv):
    """Clutter6D environment with the baseline success-rate logging wrapper."""
