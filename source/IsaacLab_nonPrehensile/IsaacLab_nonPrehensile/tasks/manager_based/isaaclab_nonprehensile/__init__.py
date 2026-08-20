import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-nonPrehensile-Franka-v0",
    entry_point=f"{__name__}.env:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:NonPrehensilePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Clutter6D-Franka-v0",
    entry_point=f"{__name__}.clutter_env:Clutter6DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clutter_env:Clutter6DEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.config.rsl_rl_ppo_cfg:Clutter6DPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Clutter6D-DAPL-Franka-v0",
    entry_point=f"{__name__}.clutter_env:Clutter6DEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clutter_env:Clutter6DDAPLEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.config.rsl_rl_ppo_cfg:Clutter6DDAPLPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-AffordanceClutter6D-Franka-v0",
    entry_point=f"{__name__}.affordance_env:AffordanceAwareClutterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.affordance_env:AffordanceAwareClutterEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.config.rsl_rl_ppo_cfg:Clutter6DPPORunnerCfg"
        ),
    },
)
