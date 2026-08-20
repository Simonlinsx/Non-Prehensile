# from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoAlgorithmCfg,
# )


# @configclass
# class PointNetActorCriticCfg:
#     """Config for PointNet-based Actor-Critic used in non-prehensile task."""

#     class_name: str = "ActorCriticPointNet"
#     # class_name: str = "ActorCritic"
#     pointnet_point_dim: int = 3
#     pointnet_num_points: int = 512
#     fuser_hidden_dims: list[int] = [256, 512, 256]
#     actor_hidden_dims: list[int] = [128, 64]
#     critic_hidden_dims: list[int] = [128, 64]
#     pointnet_output_dim: int = 128
#     activation: str = "elu"
#     init_noise_std: float = 1.0
#     noise_std_type: str = "scalar"


# @configclass
# class NonPrehensilePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     """RSL-RL PPO configuration for the non-prehensile pushing task."""

#     # Training parameters
#     num_steps_per_env = 8
#     max_iterations = 1000000
#     save_interval = 500

#     # Logging / experiment identifiers
#     experiment_name = "franka_nonprehensile"

#     # Observation normalization
#     empirical_normalization = False

#     # Policy network
#     policy = PointNetActorCriticCfg()

#     # PPO algorithm hyper-parameters
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=0.5,         
#         use_clipped_value_loss=True,
#         clip_param=0.3,                
#         entropy_coef=0.006,               
#         num_learning_epochs=8,      
#         num_mini_batches=8,            
#         learning_rate=5.0e-5,        
#         schedule="adaptive",  
#         gamma=0.99,               
#         lam=0.95,                 
#         desired_kl=0.016, 
#         max_grad_norm=1.0,
#     )


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import field


_USE_RANDOM_ICP = os.environ.get("DAPL_USE_RANDOM_ICP") == "1"
_ICP_WEIGHTS_PATH = None if _USE_RANDOM_ICP else os.environ.get(
    "DAPL_ICP_WEIGHTS", "./ckpts/512-32-balanced-SAM-wd-5e-05-920"
)


@configclass
class ICPActorCriticCfg:
    """Config for ICP-based Actor-Critic used in manipulation tasks."""

    class_name: str = "ActorCriticICP"
    
    # ICP pretrained weights path
    icp_weights_path: str | None = _ICP_WEIGHTS_PATH
    freeze_icp: bool = not _USE_RANDOM_ICP  # Never freeze a random smoke-test encoder.
    
    icp_point_dim: int = 3  # Only xyz coordinates
    icp_num_points: int = 512  # Number of points in point cloud
    
    # Network architecture
    fusion_hidden_dims: list[int] = [512, 256, 128]  # Feature fusion MLP
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])
    
    # Activation and noise configuration
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class NonPrehensilePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the non-prehensile pushing task."""

    # Training parameters
    num_steps_per_env = 8
    max_iterations = 1000000
    save_interval = 500

    # Logging / experiment identifiers
    experiment_name = "franka_nonprehensile"

    # Observation normalization
    empirical_normalization = False

    # Policy network
    policy = ICPActorCriticCfg()

    # PPO algorithm hyper-parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,         
        use_clipped_value_loss=True,
        clip_param=0.3,                
        entropy_coef=0.006,               
        num_learning_epochs=8,      
        num_mini_batches=8,            
        learning_rate=5.0e-5,        
        schedule="adaptive",  
        gamma=0.99,               
        lam=0.95,                 
        desired_kl=0.016, 
        max_grad_norm=1.0,
    )


@configclass
class Clutter6DPPORunnerCfg(NonPrehensilePPORunnerCfg):
    """Privileged target-cloud PPO baseline for Clutter6D integration."""

    experiment_name = "franka_clutter6d"


@configclass
class DAPLActorCriticCfg:
    """Paper-aligned frozen dynamics encoder and cross-attention policy."""

    class_name: str = "ActorCriticDAPL"
    world_model_checkpoint_path: str | None = os.environ.get(
        "DAPL_WORLD_MODEL_CHECKPOINT"
    )
    scene_num_points: int = 1280
    scene_point_dim: int = 7
    environment_state_dim: int = 44
    policy_attention_heads: int = 8
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class Clutter6DDAPLPPORunnerCfg(NonPrehensilePPORunnerCfg):
    """DAPL stage-2 PPO configuration from the paper supplement."""

    experiment_name = "franka_clutter6d_dapl"
    policy: DAPLActorCriticCfg = DAPLActorCriticCfg()
