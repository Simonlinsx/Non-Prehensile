"""Bounded differential-IK action used by the DyWA control-space audit."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class ClippedDifferentialInverseKinematicsAction(
    DifferentialInverseKinematicsAction
):
    """Clamp normalized policy actions before the standard IsaacLab DLS IK."""

    cfg: "ClippedDifferentialInverseKinematicsActionCfg"

    def process_actions(self, actions: torch.Tensor) -> None:
        limit = float(self.cfg.raw_action_clip)
        super().process_actions(torch.clamp(actions, min=-limit, max=limit))


@configclass
class ClippedDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    """Configuration for bounded relative Cartesian pose control."""

    class_type: type[ActionTerm] = ClippedDifferentialInverseKinematicsAction
    raw_action_clip: float = 1.0

    def __post_init__(self) -> None:
        if float(self.raw_action_clip) <= 0.0:
            raise ValueError("raw_action_clip must be positive")
