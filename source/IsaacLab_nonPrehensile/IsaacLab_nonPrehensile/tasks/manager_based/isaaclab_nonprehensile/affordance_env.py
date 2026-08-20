"""DOMINO-backed, part-aware non-prehensile clutter task."""

from __future__ import annotations

import torch

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.clutter_env import (
    Clutter6DEnv,
    Clutter6DEnvCfg,
    Clutter6DObservationsCfg,
    Clutter6DRewardsCfg,
    Clutter6DTerminationsCfg,
)


def _semantic_params() -> dict:
    return {
        "safe_radius_m": None,
        "protected_radius_m": None,
        "target_cfg": SceneEntityCfg("target"),
    }


def _contact_params() -> dict:
    return {
        "contact_distance_m": 0.008,
        "minimum_safe_score": 0.25,
        "minimum_protected_score": 0.25,
        "protected_point_count": 64,
        "protected_clearance_m": 0.005,
        "safe_radius_m": None,
        "protected_radius_m": None,
        "target_cfg": SceneEntityCfg("target"),
        "obstacles_cfg": SceneEntityCfg("obstacles"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    }


@configclass
class AffordanceAwareObservationsCfg(Clutter6DObservationsCfg):
    """Target geometry plus point-aligned DOMINO semantic scores."""

    @configclass
    class PolicyCfg(Clutter6DObservationsCfg.PolicyCfg):
        target_affordance = ObsTerm(
            func=mdp.domino_target_affordance,
            params=_semantic_params(),
        )

        def __post_init__(self) -> None:
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class WorldModelCfg(Clutter6DObservationsCfg.WorldModelCfg):
        # Keep semantics separate from the paper-fixed [B, 1280, 7] physical
        # tensor so existing DAPL checkpoints and collectors remain valid.
        target_affordance = ObsTerm(
            func=mdp.domino_target_affordance,
            params=_semantic_params(),
        )

        def __post_init__(self) -> None:
            super().__post_init__()

    world_model: WorldModelCfg = WorldModelCfg()


@configclass
class AffordanceAwareRewardsCfg(Clutter6DRewardsCfg):
    """Goal tracking with semantic-contact shaping and safety penalties."""

    safe_region_contact = RewTerm(
        func=mdp.safe_region_contact_reward,
        params=_contact_params(),
        weight=2.0,
    )
    forbidden_region_contact = RewTerm(
        func=mdp.forbidden_region_contact_penalty,
        params=_contact_params(),
        weight=-25.0,
    )
    protected_region_collision = RewTerm(
        func=mdp.protected_region_collision_penalty,
        params=_contact_params(),
        weight=-25.0,
    )


@configclass
class AffordanceAwareTerminationsCfg(Clutter6DTerminationsCfg):
    """Treat both semantic-contact violations as hard failures."""

    forbidden_region_contact = DoneTerm(
        func=mdp.forbidden_region_contact,
        params=_contact_params(),
    )
    protected_region_collision = DoneTerm(
        func=mdp.protected_region_collision,
        params=_contact_params(),
    )


@configclass
class AffordanceAwareClutterEnvCfg(Clutter6DEnvCfg):
    """Sparse Clutter6D task whose target annotations come from DOMINO."""

    observations: AffordanceAwareObservationsCfg = AffordanceAwareObservationsCfg()
    rewards: AffordanceAwareRewardsCfg = AffordanceAwareRewardsCfg()
    terminations: AffordanceAwareTerminationsCfg = AffordanceAwareTerminationsCfg()
    clutter_asset_source: str = "domino"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.clutter_asset_source != "domino":
            raise ValueError(
                "AffordanceAwareClutterEnvCfg requires DAPL_CLUTTER_ASSET_SOURCE=domino"
            )


class AffordanceAwareClutterEnv(Clutter6DEnv):
    """Clutter environment that reports constrained success separately."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.episode_affordance_violation_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_constrained_reached_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.total_constrained_episodes = 0
        self.total_constrained_successes = 0
        self.total_affordance_violation_episodes = 0

    def step(self, action):
        observations, reward, terminated, truncated, info = super().step(action)
        reached = self.termination_manager.get_term("reached")
        violation = self.termination_manager.get_term(
            "forbidden_region_contact"
        ) | self.termination_manager.get_term("protected_region_collision")
        self.episode_affordance_violation_buf |= violation
        self.episode_constrained_reached_buf |= reached
        episode_ended = terminated | truncated
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
            self.total_constrained_episodes += ended_count
            self.total_constrained_successes += success_count
            self.total_affordance_violation_episodes += violation_count
            self._episode_constrained_success_before_reset = constrained.clone()
            self._episode_affordance_violation_before_reset = (
                self.episode_affordance_violation_buf.clone()
            )
            self.episode_affordance_violation_buf[episode_ended] = False
            self.episode_constrained_reached_buf[episode_ended] = False
            log = self.extras.setdefault("log", {})
            log["constrained_success_rate"] = (
                self.total_constrained_successes / self.total_constrained_episodes
            )
            log["affordance_violation_rate"] = (
                self.total_affordance_violation_episodes
                / self.total_constrained_episodes
            )
        return observations, reward, terminated, truncated, info
