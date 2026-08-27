"""Relative joint-position action with one target latched per policy step."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class LatchedRelativeJointPositionAction(JointAction):
    """Apply ``q_target = q_policy_step + scale * action`` for all substeps.

    Isaac Lab's built-in relative joint-position action recomputes its target
    from the latest joint positions every physics substep.  With action
    decimation this turns one policy action into a moving target.  This term
    instead computes the target once in :meth:`process_actions` and keeps it
    fixed until the next policy action arrives.
    """

    cfg: "LatchedRelativeJointPositionActionCfg"

    def __init__(
        self, cfg: "LatchedRelativeJointPositionActionCfg", env
    ) -> None:
        super().__init__(cfg, env)
        if cfg.use_zero_offset:
            self._offset = 0.0
        if cfg.raw_action_clip <= 0.0:
            raise ValueError("raw_action_clip must be positive")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative")

        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_position_target = current_joint_pos.clone()
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._joint_lower = limits[..., 0] + cfg.joint_limit_margin
        self._joint_upper = limits[..., 1] - cfg.joint_limit_margin
        if bool(torch.any(self._joint_lower >= self._joint_upper).item()):
            raise RuntimeError("joint_limit_margin leaves an empty joint range")

    @property
    def joint_position_target(self) -> torch.Tensor:
        """The absolute joint target held across physics substeps."""

        return self._joint_position_target

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"expected actions with shape {tuple(self._raw_actions.shape)}, "
                f"got {tuple(actions.shape)}"
            )

        self._raw_actions[:] = actions
        bounded_actions = torch.clamp(
            actions, -self.cfg.raw_action_clip, self.cfg.raw_action_clip
        )
        self._processed_actions = bounded_actions * self._scale + self._offset

        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        target = current_joint_pos + self._processed_actions
        self._joint_position_target[:] = torch.maximum(
            torch.minimum(target, self._joint_upper), self._joint_lower
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._joint_position_target, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_position_target[env_ids] = current_joint_pos[env_ids]


@configclass
class LatchedRelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for :class:`LatchedRelativeJointPositionAction`."""

    class_type: type[ActionTerm] = LatchedRelativeJointPositionAction
    use_zero_offset: bool = True
    raw_action_clip: float = 1.0
    joint_limit_margin: float = 0.01
