# DAPL reproduction status

This repository is the development target for reproducing **Dynamics-Aware
Policy Learning (DAPL)** in Isaac Lab. The pre-existing single-object task and
CORN/ICP policy remain available as a legacy baseline; they are not yet a DAPL
reproduction. DyWA is reference material only.

## Reproducibility contract

The implementation records the paper's environment-facing contract in
simulator-independent code and exposes it through a manifest-backed Isaac Lab
task:

- `dapl.data` resolves the public `Steve3zz/DAPL-dataset` layout through an
  explicit path or `DAPL_DATA_ROOT`. It contains no developer-home defaults.
- `dapl.scene` defines versioned JSONL manifests for Sparse (4 objects),
  Moderate (8), and Dense (12) Clutter6D scenes, including physical properties
  and the 16 target initial/goal pairs.
- `dapl.representation` constructs the physical scene tensor
  `[x, y, z, M/N, vx, vy, vz]`: 512 target points, the 512 obstacle points
  nearest to the target centroid, and 256 end-effector points.
- Obstacle source indices are part of the output. Reusing them for the future
  frame preserves point correspondence for the `t -> t + 0.1 s` world-model
  target.
- Tensor value checks can be enabled with `validate_values=True` for fixtures.
  They are disabled by default because checking CUDA values would synchronize
  every vectorized RL observation step; tensor shapes are always checked.
- `Isaac-Clutter6D-Franka-v0` deterministically maps `env_id % scene_count`
  onto manifest assets, cycles through the 16 tasks in each scene, and keeps
  the legacy single-object task unchanged.
- Its privileged ICP policy observation remains 1,580-dimensional for the
  runnable baseline. A separate, non-concatenated `world_model.scene` group
  exposes `[B, 1280, 7]` without changing the actor input.
- `dapl.models` implements semantic FPS/kNN tokenization with 16 target,
  16 obstacle, and 8 end-effector patches (`K=32`), the paper's 12-block,
  128-dimensional, 8-head dynamics encoder, 3-D end-effector-flow-conditioned
  decoder, patch-index scatter/averaging, and position/velocity/variance
  objectives with weights `1 / 1 / 100`.
- `Isaac-Clutter6D-DAPL-Franka-v0` exposes the paper policy layout as a
  flattened 8,960-D physical scene followed by the 44-D hand/robot/goal/
  physics/previous-action state. `ActorCriticDAPL` restores and freezes the
  world-model encoder, cross-attends a state query over its 40 dynamics tokens,
  and uses the reported `[512, 256, 128]` fusion and `[64]` policy/value heads.

The public Hugging Face dataset was inspected on 2026-08-10. Its relevant
layout is:

```text
<DAPL_DATA_ROOT>/
  flattened_usds/<asset_id>/_<asset_id>.usd
  usds/<asset_id>/<asset_id>_geometry.obj
  usds/<asset_id>/<asset_id>_geometry_wo_coacd.obj
  embodiments/hand_merged.obj
  embodiments/pc_npy_cache/hand_merged.npy
```

It currently exposes assets and the hand point cache, but no train/eval scene
manifests. Therefore Clutter6D scene generation and split manifests must be
reconstructed rather than assumed to be present in the download.

## Local validation

The foundation tests require only PyTorch, not Isaac Sim:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
  python -m unittest discover -s tests -v
```

The development environment validated on 2026-08-10 is:

- Conda environment: `dapl-isaaclab` (Python 3.11)
- Isaac Sim 5.0.0.0 and Isaac Lab v2.2.0
- PyTorch 2.7.0+cu126
- FlashAttention 2.8.3
- PyTorch3D v0.7.9, built for CUDA architecture 8.9

Isaac Sim pins `packaging==23.0`, `click==8.1.7`, `psutil==5.9.8`, and
`typing_extensions==4.12.2`. To keep those pins satisfiable with packages
resolved after the Isaac Lab 2.2 release, this environment uses
`wheel==0.45.1`, `setuptools==80.9.0`, `onnx==1.18.0`,
`transformers==4.52.4`, `huggingface-hub==0.33.4`, and `ipython==8.37.0`.
`python -m pip check` reports no broken requirements.

The repository README previously requested PyTorch3D `v0.7.8`, but that tag
does not exist upstream (the official tags jump from `v0.7.7` to `v0.7.9`).
The validated build therefore uses `v0.7.9`.

## Validated Isaac Sim baselines

The local `dapl-isaaclab` Conda environment stores the accepted EULA flag and
the baseline asset/weight paths, so a new shell only needs:

```bash
source /data1/linsixu/miniconda3/bin/activate dapl-isaaclab
cd /data1/linsixu/IsaacLab-nonPrehensile
```

The public DGN baseline release was downloaded and validated at
`/data1/linsixu/datasets/DGN/DGN`. Its `yes.json` contains 323 entries and all
323 referenced USD and OBJ assets are present. The released ICP encoder is at
`ckpts/512-32-balanced-SAM-wd-5e-05-920`; its SHA-256 is
`f4c72e764fe1cfe4e6f08add3e24c53001032338c3088669f09e8840883fea7a`.

The following integration layers passed on one NVIDIA L40:

- headless Isaac Sim startup, one update, and clean shutdown;
- Isaac Lab and project task registration/config parsing;
- one-environment reset and five physics/control steps with the opt-in cube;
- one PPO iteration with 8 cube environments and a random trainable ICP;
- one PPO iteration with 8 DGN objects and the released frozen ICP encoder.

The final baseline smoke run produced 64 timesteps at approximately 19
steps/s and saved `model_0.pt` under
`logs/rsl_rl/dgn_pretrained_icp_smoke/2026-08-10_20-00-05_one_iteration_retry/`.
It validates the pipeline only; one iteration is not a performance result.

For a longer legacy-baseline run:

```bash
python scripts/train.py \
  --task=Isaac-nonPrehensile-Franka-v0 \
  --num_envs=256 \
  --max_iterations=1000 \
  --headless \
  --device=cuda:0 \
  --experiment_name=dgn_pretrained_icp
```

This runnable single-object baseline is still not the DAPL paper
reproduction.

### Manifest-backed Clutter6D integration

A deterministic two-scene DGN Sparse development manifest is checked in at
`data/manifests/dgn_sparse_smoke_seed17.jsonl`. Each scene contains one target,
one large obstacle, two small obstacles, and 16 task pairs. Its SHA-256 is
`582f18ea75c92705177bf84f228ba187196cf928a6ad9a47c9d06ab6b91dbbc7`.
This DGN adapter is an integration fixture, not the official Clutter6D
benchmark: DGN and DAPL use different asset normalization and the public DAPL
release does not include the official scene manifests.

The following paths have passed on an NVIDIA L40:

- Gym registration and configuration parsing for `Isaac-Clutter6D-Franka-v0`;
- 2- and 8-environment reset/step tests with one target plus three obstacles;
- exact `(B, 1580)` privileged policy observation and non-concatenated
  `(B, 1280, 7)` physical world-model observation;
- paper thresholds/weights for contact, coarse/fine goal tracking, planar 6D
  success, obstacle-motion discount, target/any-obstacle drop, and 300 policy
  steps;
- manifest mass, inertia, static/dynamic friction, and restitution applied to
  each spawned rigid body;
- one PPO update with the released frozen ICP, both before and after adding the
  world-model observation group.

The post-world-model regression checkpoint is:

```text
logs/rsl_rl/franka_clutter6d/
  2026-08-10_20-41-20_dgn_sparse_world_model_obs_one_iteration/model_0.pt
```

A longer target-cloud privileged baseline completed 500 PPO iterations with
128 environments (512,000 transitions) at:

```text
logs/rsl_rl/franka_clutter6d/
  2026-08-10_20-43-04_dgn_sparse_privileged_128env_500iter_s17/model_499.pt
```

Its mean training reward rose from `0.0157` over the first 50 iterations to
`50.2038` over the last 50, and goal-tracking reward rose from `0.00036` to
`0.92501`. However, it recorded zero successes. This shows that the baseline
learns contact and partial tracking, but it is not an effective DAPL result
and should not be reported as one.

The paper defines `d_max` and `theta_max` for normalizing obstacle motion but
does not publish their numerical values. The development task therefore keeps
the provisional choices `0.2 m` and `pi rad` explicit in `Clutter6DRewardsCfg`.
They must not be presented as released paper constants.

Run the bounded simulator smoke test with:

```bash
export DGN_DATA_ROOT=/data1/linsixu/datasets/DGN/DGN
export DAPL_CLUTTER_MANIFEST=$PWD/data/manifests/dgn_sparse_smoke_seed17.jsonl
export DAPL_CLUTTER_ASSET_SOURCE=dgn
python scripts/smoke_clutter6d.py --headless --num_envs 8 --steps 8
```

Collect aligned world-model transitions (one control step is exactly `0.1 s`):

```bash
python scripts/collect_dapl_transitions.py \
  --headless --num-envs 8 --steps 128 \
  --output-dir outputs/dapl_transitions/smoke_seed17
```

The collector stores `scene_t`, normalized relative-joint `action`, the
paper-defined 3-D `end_effector_flow`, aligned `scene_tp1`, the reused obstacle
source indices, goal/task metadata, and reward in bounded `.pt` shards. It
filters terminal steps so a transition never crosses an automatic environment
reset.

Policy-driven collection and the released hand cache are also supported:

```bash
export DAPL_HAND_POINTS=/path/to/embodiments/pc_npy_cache/hand_merged.npy
python scripts/collect_dapl_transitions.py \
  --headless --num-envs 128 --steps 469 \
  --action-mode policy --policy-action sample \
  --checkpoint /path/to/model.pt \
  --output-dir outputs/dapl_transitions/policy_seed17
```

The released cache is validated as finite `float32 [256, 3]` data and
transformed from the `panda_hand` frame through the configured Franka TCP.
Its SHA-256 is
`2d498477c0f886cec4f42fdfebc03910b7a56a5ed0d45034b7477b140b6a9539`.
The collector records the actual hand source, checkpoint, action sampling
mode, and initial-task mode in every shard. Its default distributed start
covers the manifest's 16 task pairs across parallel environments.

Train the paper-aligned dynamics model with the shard-streaming trainer:

```bash
python scripts/train_dapl_world_model.py \
  --data-dir outputs/dapl_transitions/train \
  --output-dir outputs/dapl_world_model/train_seed17 \
  --device cuda:0 --batch-size 32 --max-steps 500000
```

The trainer fits 7-D feature normalization on the training shards, scales the
3-D flow with the position standard deviation, uses AdamW, validates on a
deterministic shard split, and atomically saves model/optimizer/normalizer
state. A GPU smoke run using the paper-default 12-block model completed 10
finite optimization steps and wrote:

```text
outputs/dapl_world_model/smoke_seed17_20260810/
  world_model_step_0000010.pt
  summary.json
```

That run used only eight random-action transitions. It verifies execution and
checkpoint loading, not prediction quality or generalization.

The trainer now evaluates and logs the no-change persistence baseline at the
start of every run, atomically maintains `world_model_best.pt`, and supports
optimizer/model/normalizer continuation via `--resume`. A 2,048-transition
policy-rollout pilot, evenly covering all 16 tasks and using the released hand
cache, was trained for 10,500 steps. On its fixed 512-transition held-out
temporal shard:

- persistence: total `434.559`, position `0.0579`, velocity `3.8350`, variance
  `4.3067`;
- step 10,000: total `409.877`, position `0.3007`, velocity `1.9095`, variance
  `4.0767`;
- step 10,500: total `405.734`, position `0.3292`, velocity `1.8786`, variance
  `4.0353`.

Thus the pilot learns a measurable velocity-dynamics signal and beats the
weighted persistence total by about 6.6% at step 10,500, but its absolute
position head remains worse than persistence. This is not yet a satisfactory
world model or a DAPL reproduction. The dataset is also only the two-scene DGN
integration fixture, not the official Clutter6D distribution.

`scripts/evaluate_dapl_world_model.py` restores the serialized architecture and
normalizer and evaluates every held-out shard, rather than only the trainer's
bounded validation window. It reports training-compatible normalized losses,
component-wise position/velocity RMSE in physical units, the persistence
baseline, a zero-action ablation, action-conditioned prediction deltas, and a
checkpoint SHA-256. A bounded `--max-batches` mode exists only for smoke tests.

The stage-2 integration path has also passed a real Isaac Sim/RSL-RL smoke:

```text
task: Isaac-Clutter6D-DAPL-Franka-v0
observation: (9004,) = (1280 * 7) + 44
world-model checkpoint: step 30000
parallel environments: 2
PPO iterations/timesteps: 1 / 16
result: finite actor/critic update and clean exit
log: logs/rsl_rl/franka_clutter6d_dapl/
     2026-08-11_06-54-29_dapl_actor_integration_smoke2_s17/
```

This validates the simulator-to-policy plumbing, not task performance. Formal
stage-2 training must use the accepted final/best checkpoint from the 500k
world-model run.

For the target-only privileged PPO baseline, the unused world-model group can
be disabled without changing the task physics:

```bash
export DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0
python scripts/train.py \
  --task Isaac-Clutter6D-Franka-v0 \
  --headless --num_envs 128 --max_iterations 500 --seed 17
```

Without an explicit released cache, the analytical 256-point two-finger cloud
remains available as a development fallback. Reported runs must use
`embodiments/pc_npy_cache/hand_merged.npy` and retain the shard metadata.

## Implementation sequence

1. **Done for the DGN integration fixture:** deterministic Clutter6D manifests,
   named target/obstacle assets and physical parameters, task reset, rewards,
   paper-aligned terminations, and the `[B, 1280, 7]` physical scene
   observation.
2. **Done for the development collector:** aligned `t -> t + 0.1 s` transition
   shards that preserve obstacle point identity and exclude reset crossings.
3. **Done for the development model/trainer:** semantic FPS/kNN tokenization
   (16/16/8 patches, K=32), the 12-block 128-dimensional dynamics encoder,
   action-conditioned decoder, position/velocity/variance losses, feature
   normalization, validation, TensorBoard logging, and atomic checkpoints.
4. **Done for the development path:** validate and transform the released hand
   cache, load PPO checkpoints for sampled or mean policy rollouts, distribute
   parallel collection across all manifest tasks, and record provenance.
5. **In progress on the DGN proxy distribution:** collect approximately 60k
   policy interaction steps, train the world model for 500k optimization steps,
   and evaluate held-out one-step position/velocity prediction before selecting
   its encoder. The complete run must not be labeled an official Clutter6D
   result because official scene manifests/assets are unavailable.
6. **Implemented and integration-smoked:** connect the frozen encoder to the
   paper's cross-attention actor-critic. Formal stage-2 PPO training and the
   alternating policy/world-model curriculum remain to be run.
7. Build the sim2real student path: noisy perception inputs, latent/action
   distillation, EKF velocities, action-scale curriculum, Cartesian clipping,
   and impedance execution.

The privileged simulation path should be evaluated first. Camera perception
and student distillation depend on the same scene and policy contracts, so
they come after the multi-object environment and world model are stable.
