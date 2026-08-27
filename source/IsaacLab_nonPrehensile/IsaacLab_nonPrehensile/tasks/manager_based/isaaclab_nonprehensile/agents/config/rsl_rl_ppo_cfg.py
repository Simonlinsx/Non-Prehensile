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
class AffordancePPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO options used by controlled teacher curriculum transfers."""

    actor_loss_coef: float = 1.0


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


@configclass
class AffordanceActorCriticCfg:
    """Joint target-affordance and obstacle PointNet/attention policy."""

    class_name: str = "ActorCriticAffordance"
    target_num_points: int = 512
    target_point_dim: int = 5
    obstacle_num_points: int = 512
    obstacle_point_dim: int = 3
    environment_state_dim: int = 50
    point_feature_dim: int = 64
    attention_heads: int = 4
    attention_queries: int = 1
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    actor_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    max_noise_std: float | None = None


@configclass
class AffordancePPORunnerCfg(NonPrehensilePPORunnerCfg):
    """PPO runner shared by all four affordance curriculum stages."""

    experiment_name = "franka_affordance_curriculum"
    save_interval = 250
    policy: AffordanceActorCriticCfg = AffordanceActorCriticCfg()


@configclass
class AffordanceProofActorCriticCfg(AffordanceActorCriticCfg):
    """Conservative exploration for the tightly initialized hammer proof task."""

    # A unit Gaussian together with 0.1-rad relative joint actions produced
    # violent random impacts before PPO had seen a successful push.  The proof
    # task starts close enough to the handle that substantially less noise is
    # sufficient for exploration.
    init_noise_std: float = 0.35
    noise_std_type: str = "log"


@configclass
class AffordanceProofPPORunnerCfg(AffordancePPORunnerCfg):
    """PPO runner used only by the single-hammer, no-obstacle proof task."""

    policy: AffordanceProofActorCriticCfg = AffordanceProofActorCriticCfg()
    num_steps_per_env: int = 16


@configclass
class AffordanceRefineActorCriticCfg(AffordanceProofActorCriticCfg):
    """Cap resumed exploration while refining strict pose and dwell control."""

    init_noise_std: float = 0.10
    max_noise_std: float | None = 0.10


@configclass
class AffordanceRefinePPORunnerCfg(AffordanceProofPPORunnerCfg):
    policy: AffordanceRefineActorCriticCfg = AffordanceRefineActorCriticCfg()


@configclass
class AffordanceScratchActorCriticCfg(AffordanceProofActorCriticCfg):
    """Exploration budget for learning safe-region approach from random weights."""

    init_noise_std: float = 0.40
    max_noise_std: float | None = 0.40


@configclass
class AffordanceScratchPPORunnerCfg(AffordanceProofPPORunnerCfg):
    policy: AffordanceScratchActorCriticCfg = AffordanceScratchActorCriticCfg()
    save_interval: int = 50


@configclass
class AffordanceTeacherActorCriticCfg(AffordanceScratchActorCriticCfg):
    """Recoverable actor with an independent privileged training critic."""

    environment_state_dim: int = 45
    critic_environment_state_dim: int = 50


@configclass
class AffordanceTeacherPPORunnerCfg(AffordanceProofPPORunnerCfg):
    """Deployment-aligned oracle-affordance teacher training."""

    experiment_name = "franka_affordance_teacher"
    policy: AffordanceTeacherActorCriticCfg = AffordanceTeacherActorCriticCfg()
    save_interval: int = 50


@configclass
class AffordanceTeacherFrozenV7GoalWrenchActorCriticCfg(
    AffordanceTeacherActorCriticCfg
):
    """Frozen-v7 actor plus one explicit recoverable goal-point relation.

    The external observation remains exactly 512 target points with
    ``[x, y, z, safe, protected]``, 512 obstacle XYZ points, and the 45-value
    deployable state.  The only representational delta is an internal
    per-target-point residual containing point-to-hand, object-local,
    translation-support, and signed-yaw-moment relations derived from those
    same values.  Translation and yaw channels stay separate so full-pose
    goals are not collapsed into a single ambiguous scalar.
    """

    use_relation_features: bool = True
    use_wrench_relation_features: bool = True
    separate_wrench_relation_features: bool = True
    zero_initialize_relation_output: bool = False
    wrench_relation_yaw_moment_weight: float = 1.5
    wrench_relation_yaw_activation_rad: float = 0.10


@configclass
class AffordanceTeacherFrozenV7GoalWrenchPPORunnerCfg(
    AffordanceTeacherPPORunnerCfg
):
    """Single-variable goal-conditioned point-attention comparison."""

    policy: AffordanceTeacherFrozenV7GoalWrenchActorCriticCfg = (
        AffordanceTeacherFrozenV7GoalWrenchActorCriticCfg()
    )


@configclass
class AffordanceTeacherDAPLActorCriticCfg(AffordanceTeacherActorCriticCfg):
    """Affordance observation contract with the DAPL baseline exploration."""

    init_noise_std: float = 1.0
    max_noise_std: float | None = None


@configclass
class AffordanceTeacherDAPLPPORunnerCfg(AffordanceTeacherPPORunnerCfg):
    """DAPL PPO hyperparameters for the minimally modified C1 teacher."""

    policy: AffordanceTeacherDAPLActorCriticCfg = (
        AffordanceTeacherDAPLActorCriticCfg()
    )
    num_steps_per_env: int = 8
    save_interval: int = 50


@configclass
class AffordanceTeacherDAPLProgressActorCriticCfg(
    AffordanceTeacherActorCriticCfg
):
    """Bounded from-scratch exploration for transition-progress shaping."""

    init_noise_std: float = 0.40
    max_noise_std: float | None = 0.40


@configclass
class AffordanceTeacherDAPLProgressPPORunnerCfg(
    AffordanceTeacherDAPLPPORunnerCfg
):
    """Keep the baseline PPO contract while fixing clipped exploration."""

    policy: AffordanceTeacherDAPLProgressActorCriticCfg = (
        AffordanceTeacherDAPLProgressActorCriticCfg()
    )


@configclass
class AffordanceTeacherMultiQueryDAPLProgressActorCriticCfg(
    AffordanceTeacherDAPLProgressActorCriticCfg
):
    """DyWA-style multi-query point attention with no new observations."""

    attention_queries: int = 16


@configclass
class AffordanceTeacherMultiQueryDAPLProgressPPORunnerCfg(
    AffordanceTeacherDAPLProgressPPORunnerCfg
):
    """Single-variable multi-query comparison for the no-C1 baseline."""

    policy: AffordanceTeacherMultiQueryDAPLProgressActorCriticCfg = (
        AffordanceTeacherMultiQueryDAPLProgressActorCriticCfg()
    )


@configclass
class AffordanceTeacherCartesianDAPLProgressActorCriticCfg(
    AffordanceTeacherDAPLProgressActorCriticCfg
):
    """Single-query actor for the six-dimensional Cartesian action audit."""

    environment_state_dim: int = 44
    critic_environment_state_dim: int = 49


@configclass
class AffordanceTeacherCartesianDAPLProgressPPORunnerCfg(
    AffordanceTeacherDAPLProgressPPORunnerCfg
):
    policy: AffordanceTeacherCartesianDAPLProgressActorCriticCfg = (
        AffordanceTeacherCartesianDAPLProgressActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationScratchActorCriticCfg(
    AffordanceTeacherActorCriticCfg
):
    """From-scratch teacher with only recoverable point relations added."""

    # Keep the original scratch exploration and PPO contract.  Unlike the
    # short v38 continuation, all encoders and the policy head are optimized
    # together from random initialization, so the relation branch is not
    # forced to undo a converged one-sided policy through a residual alone.
    use_relation_features: bool = True


@configclass
class AffordanceTeacherRelationScratchPPORunnerCfg(
    AffordanceTeacherPPORunnerCfg
):
    """Standard from-scratch PPO for the recoverable relation policy."""

    policy: AffordanceTeacherRelationScratchActorCriticCfg = (
        AffordanceTeacherRelationScratchActorCriticCfg()
    )
    save_interval: int = 25


@configclass
class AffordanceTeacherRelationWrenchScratchActorCriticCfg(
    AffordanceTeacherRelationScratchActorCriticCfg
):
    """Recoverable relation encoder with signed goal-wrench compatibility."""

    use_wrench_relation_features: bool = True
    wrench_relation_yaw_moment_weight: float = 1.0
    wrench_relation_yaw_activation_rad: float = 0.10


@configclass
class AffordanceTeacherRelationWrenchScratchPPORunnerCfg(
    AffordanceTeacherRelationScratchPPORunnerCfg
):
    policy: AffordanceTeacherRelationWrenchScratchActorCriticCfg = (
        AffordanceTeacherRelationWrenchScratchActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationWrenchSeparatedScratchActorCriticCfg(
    AffordanceTeacherRelationWrenchScratchActorCriticCfg
):
    """From-scratch policy that keeps translation and yaw wrench cues distinct."""

    separate_wrench_relation_features: bool = True
    zero_initialize_relation_output: bool = False
    wrench_relation_yaw_moment_weight: float = 1.5


@configclass
class AffordanceTeacherRelationWrenchSeparatedScratchPPORunnerCfg(
    AffordanceTeacherRelationWrenchScratchPPORunnerCfg
):
    policy: AffordanceTeacherRelationWrenchSeparatedScratchActorCriticCfg = (
        AffordanceTeacherRelationWrenchSeparatedScratchActorCriticCfg()
    )


@configclass
class AffordanceTeacherHardActorCriticCfg(AffordanceTeacherActorCriticCfg):
    """Deployment-aligned teacher with bounded hard-constraint exploration."""

    # Resuming restores the soft policy's learned log_std parameter.  The
    # independent cap is therefore required to keep the first hard episodes
    # useful instead of terminating on random C1/C2/C3 impacts.
    init_noise_std: float = 0.005
    max_noise_std: float | None = 0.005


@configclass
class AffordanceTeacherHardPPORunnerCfg(AffordanceTeacherPPORunnerCfg):
    """Low-noise runner for C1/C2/C3 and combined hard fine-tuning."""

    policy: AffordanceTeacherHardActorCriticCfg = (
        AffordanceTeacherHardActorCriticCfg()
    )
    # Hard-contact adaptation can improve rapidly and then drift toward the
    # locally attractive no-contact/no-push solution.  Keep both the update
    # budget and checkpoint spacing conservative so the useful transient is
    # measurable on a disjoint constrained evaluation set.
    save_interval: int = 2
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class AffordanceTeacherRelationHardActorCriticCfg(
    AffordanceTeacherHardActorCriticCfg
):
    """Hard-constraint policy retaining the recoverable relation encoder."""

    use_relation_features: bool = True


@configclass
class AffordanceTeacherRelationHardPPORunnerCfg(
    AffordanceTeacherHardPPORunnerCfg
):
    policy: AffordanceTeacherRelationHardActorCriticCfg = (
        AffordanceTeacherRelationHardActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationWrenchHardActorCriticCfg(
    AffordanceTeacherRelationHardActorCriticCfg
):
    """Hard-C1 evaluation architecture for wrench-aware checkpoints."""

    use_wrench_relation_features: bool = True
    wrench_relation_yaw_moment_weight: float = 1.0
    wrench_relation_yaw_activation_rad: float = 0.10


@configclass
class AffordanceTeacherRelationWrenchHardPPORunnerCfg(
    AffordanceTeacherRelationHardPPORunnerCfg
):
    policy: AffordanceTeacherRelationWrenchHardActorCriticCfg = (
        AffordanceTeacherRelationWrenchHardActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationWrenchSeparatedHardActorCriticCfg(
    AffordanceTeacherRelationWrenchHardActorCriticCfg
):
    """Hard-C1 architecture matching separated-wrench scratch checkpoints."""

    separate_wrench_relation_features: bool = True
    zero_initialize_relation_output: bool = False
    wrench_relation_yaw_moment_weight: float = 1.5


@configclass
class AffordanceTeacherRelationWrenchSeparatedHardPPORunnerCfg(
    AffordanceTeacherRelationWrenchHardPPORunnerCfg
):
    policy: AffordanceTeacherRelationWrenchSeparatedHardActorCriticCfg = (
        AffordanceTeacherRelationWrenchSeparatedHardActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationWrenchSeparatedC2HardActorCriticCfg(
    AffordanceTeacherRelationWrenchSeparatedHardActorCriticCfg
):
    """C2 continuation noise that keeps PPO ratios numerically useful.

    The 0.005 standard deviation used for deterministic-like hard audits makes
    the on-policy Gaussian too narrow for adaptation: a tiny actor-mean update
    produces a large likelihood-ratio change.  C2 training retains a small
    0.02 exploration budget; deterministic evaluation remains unchanged.
    """

    init_noise_std: float = 0.02
    max_noise_std: float | None = 0.02


@configclass
class AffordanceTeacherRelationWrenchSeparatedC2HardPPORunnerCfg(
    AffordanceTeacherRelationWrenchSeparatedHardPPORunnerCfg
):
    policy: AffordanceTeacherRelationWrenchSeparatedC2HardActorCriticCfg = (
        AffordanceTeacherRelationWrenchSeparatedC2HardActorCriticCfg()
    )


@configclass
class AffordanceTeacherProtectedObstacleC2HardActorCriticCfg(
    AffordanceTeacherRelationWrenchSeparatedC2HardActorCriticCfg
):
    """Hard-C2 audit policy with one recoverable protected–blocker residual."""

    use_protected_obstacle_relation_features: bool = True
    zero_initialize_protected_obstacle_relation_output: bool = True


@configclass
class AffordanceTeacherProtectedObstacleC2HardPPORunnerCfg(
    AffordanceTeacherRelationWrenchSeparatedC2HardPPORunnerCfg
):
    policy: AffordanceTeacherProtectedObstacleC2HardActorCriticCfg = (
        AffordanceTeacherProtectedObstacleC2HardActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationWrenchSeparatedC2CriticWarmupPPORunnerCfg(
    AffordanceTeacherRelationWrenchSeparatedC2HardPPORunnerCfg
):
    """Adapt the critic to hard-C2 returns without changing the source actor."""

    algorithm = AffordancePPOAlgorithmCfg(
        actor_loss_coef=0.0,
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=8,
        # An actor-frozen rollout has KL ~= 0 by construction.  An adaptive
        # KL schedule would therefore raise the critic learning rate to the
        # 1e-2 ceiling even though the critic is the only trainable branch.
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class AffordanceTeacherSafetyRefineActorCriticCfg(AffordanceTeacherActorCriticCfg):
    """Small exploration budget for behavior-preserving soft C1 adaptation."""

    init_noise_std: float = 0.02
    max_noise_std: float | None = 0.02


@configclass
class AffordanceTeacherSafetyRefinePPORunnerCfg(AffordanceTeacherPPORunnerCfg):
    """Conservative PPO used between the pushing and hard-contact phases."""

    policy: AffordanceTeacherSafetyRefineActorCriticCfg = (
        AffordanceTeacherSafetyRefineActorCriticCfg()
    )
    save_interval: int = 5
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class AffordanceTeacherRelationSafetyRefineActorCriticCfg(
    AffordanceTeacherSafetyRefineActorCriticCfg
):
    """Low-noise soft transfer without changing relation-policy structure."""

    use_relation_features: bool = True


@configclass
class AffordanceTeacherRelationSafetyRefinePPORunnerCfg(
    AffordanceTeacherSafetyRefinePPORunnerCfg
):
    policy: AffordanceTeacherRelationSafetyRefineActorCriticCfg = (
        AffordanceTeacherRelationSafetyRefineActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationWrenchSeparatedSafetyRefineActorCriticCfg(
    AffordanceTeacherSafetyRefineActorCriticCfg
):
    """Low-noise transfer architecture matching the v53 teacher exactly."""

    use_relation_features: bool = True
    use_wrench_relation_features: bool = True
    separate_wrench_relation_features: bool = True
    zero_initialize_relation_output: bool = False
    wrench_relation_yaw_moment_weight: float = 1.5
    wrench_relation_yaw_activation_rad: float = 0.10


@configclass
class AffordanceTeacherRelationWrenchSeparatedSafetyRefinePPORunnerCfg(
    AffordanceTeacherSafetyRefinePPORunnerCfg
):
    """Behavior-preserving C2/C3 transfer for separated-wrench checkpoints."""

    policy: AffordanceTeacherRelationWrenchSeparatedSafetyRefineActorCriticCfg = (
        AffordanceTeacherRelationWrenchSeparatedSafetyRefineActorCriticCfg()
    )


@configclass
class AffordanceTeacherProtectedObstacleSafetyRefineActorCriticCfg(
    AffordanceTeacherRelationWrenchSeparatedSafetyRefineActorCriticCfg
):
    """Soft-C2 transfer with one deployable protected–blocker relation."""

    use_protected_obstacle_relation_features: bool = True
    zero_initialize_protected_obstacle_relation_output: bool = True


@configclass
class AffordanceTeacherProtectedObstacleSafetyRefinePPORunnerCfg(
    AffordanceTeacherRelationWrenchSeparatedSafetyRefinePPORunnerCfg
):
    policy: AffordanceTeacherProtectedObstacleSafetyRefineActorCriticCfg = (
        AffordanceTeacherProtectedObstacleSafetyRefineActorCriticCfg()
    )


@configclass
class AffordanceTeacherProtectedObstacleAdapterActorCriticCfg(
    AffordanceTeacherProtectedObstacleSafetyRefineActorCriticCfg
):
    """Preserve the competent source actor while learning only the C2 adapter."""

    freeze_base_actor_for_protected_obstacle_transfer: bool = True


@configclass
class AffordanceTeacherProtectedObstacleAdapterPPORunnerCfg(
    AffordanceTeacherProtectedObstacleSafetyRefinePPORunnerCfg
):
    """Fixed-rate PPO for the small protected–blocker residual and critic."""

    policy: AffordanceTeacherProtectedObstacleAdapterActorCriticCfg = (
        AffordanceTeacherProtectedObstacleAdapterActorCriticCfg()
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class AffordanceTeacherProtectedObstacleExplorationAdapterActorCriticCfg(
    AffordanceTeacherProtectedObstacleAdapterActorCriticCfg
):
    """Broaden only soft-stage exploration around the frozen source mean."""

    init_noise_std: float = 0.10
    max_noise_std: float | None = 0.10


@configclass
class AffordanceTeacherProtectedObstacleExplorationAdapterPPORunnerCfg(
    AffordanceTeacherProtectedObstacleAdapterPPORunnerCfg
):
    policy: AffordanceTeacherProtectedObstacleExplorationAdapterActorCriticCfg = (
        AffordanceTeacherProtectedObstacleExplorationAdapterActorCriticCfg()
    )


@configclass
class AffordanceTeacherDirectionalSafetyActorCriticCfg(
    AffordanceTeacherActorCriticCfg
):
    """Moderate exploration for discovering a new legal contact side."""

    init_noise_std: float = 0.10
    max_noise_std: float | None = 0.10


@configclass
class AffordanceTeacherDirectionalSafetyPPORunnerCfg(
    AffordanceTeacherPPORunnerCfg
):
    """Standard PPO updates with bounded directional-contact exploration."""

    policy: AffordanceTeacherDirectionalSafetyActorCriticCfg = (
        AffordanceTeacherDirectionalSafetyActorCriticCfg()
    )
    save_interval: int = 5


@configclass
class AffordanceTeacherGeodesicActorCriticCfg(AffordanceTeacherActorCriticCfg):
    """Intermediate exploration for semantic-route adaptation."""

    init_noise_std: float = 0.05
    max_noise_std: float | None = 0.05


@configclass
class AffordanceTeacherGeodesicPPORunnerCfg(AffordanceTeacherPPORunnerCfg):
    """Behavior-preserving PPO for full-direction plus endpoint replay."""

    policy: AffordanceTeacherGeodesicActorCriticCfg = (
        AffordanceTeacherGeodesicActorCriticCfg()
    )
    save_interval: int = 5
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.5,
    )


@configclass
class AffordanceTeacherVectorFieldActorCriticCfg(AffordanceTeacherActorCriticCfg):
    """Exploration budget for discovering a lateral semantic detour."""

    init_noise_std: float = 0.15
    max_noise_std: float | None = 0.15


@configclass
class AffordanceTeacherVectorFieldPPORunnerCfg(
    AffordanceTeacherGeodesicPPORunnerCfg
):
    """Conservative PPO with enough action diversity to sample a detour."""

    policy: AffordanceTeacherVectorFieldActorCriticCfg = (
        AffordanceTeacherVectorFieldActorCriticCfg()
    )


@configclass
class AffordanceTeacherRelationVectorFieldActorCriticCfg(
    AffordanceTeacherVectorFieldActorCriticCfg
):
    """PointNet/attention teacher with explicit recoverable point relations."""

    use_relation_features: bool = True


@configclass
class AffordanceTeacherRelationVectorFieldPPORunnerCfg(
    AffordanceTeacherVectorFieldPPORunnerCfg
):
    policy: AffordanceTeacherRelationVectorFieldActorCriticCfg = (
        AffordanceTeacherRelationVectorFieldActorCriticCfg()
    )


@configclass
class AffordanceHardProofActorCriticCfg(AffordanceProofActorCriticCfg):
    """Low-noise exploration after the deterministic push has been learned."""

    # The 0.03-rad action scale makes 0.05 policy noise large enough to move
    # the fingers across the narrow safe-mask boundary in one step.  Keep only
    # a small residual perturbation during hard-constraint fine-tuning.
    init_noise_std: float = 0.005
    max_noise_std: float | None = 0.005


@configclass
class AffordanceHardProofPPORunnerCfg(AffordanceProofPPORunnerCfg):
    """Hard-contact fine-tuning without restoring unsafe checkpoint noise."""

    policy: AffordanceHardProofActorCriticCfg = AffordanceHardProofActorCriticCfg()
