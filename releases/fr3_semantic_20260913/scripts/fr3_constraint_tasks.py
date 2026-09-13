import gymnasium as gym
import os
from isaaclab.utils import configclass
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.affordance_env import AffordanceHammerTeacherEnvCfg
@configclass
class TypedCfg(AffordanceHammerTeacherEnvCfg):
    def _configure_curriculum_stage(self):
        super()._configure_curriculum_stage()
        self.active_obstacle_count = int(os.environ['M1_ACTIVE_OBSTACLES'])
        self.kinematic_active_obstacles = False
gym.register(id='Isaac-FR3-Typed-Validation-v0',
    entry_point='IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.affordance_env:AffordanceAwareClutterEnv',
    disable_env_checker=True, kwargs={'env_cfg_entry_point': TypedCfg})
