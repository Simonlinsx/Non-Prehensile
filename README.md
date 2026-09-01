# Affordance-Aware Non-Prehensile Manipulation

This repository develops safety-aware non-prehensile manipulation of tools in
Isaac Lab.  The current frozen result is **C1**: a Franka must place a DOMINO
hammer while contacting only its safe handle region.  The functional hammer
head and claw are protected.

The accepted source snapshot is tagged `c1-accepted-v30`.  The exact protocol,
checkpoint provenance, and limitations are recorded in
[the accepted C1 snapshot](docs/C1_ACCEPTED_SNAPSHOT.md).  Later clutter,
C2/C3, 360-degree, world-model, and RGB-D student experiments are research
work in progress and are not part of the accepted claim.

## Current result: C1

### Task and safety contract

- Target: DOMINO `020_hammer:0`, one fixed stable support orientation.
- Goal displacement: `6.5--9.5 cm`, with planar direction in
  `[-45 deg, +45 deg]`.
- Initial target XY: `x=0.46--0.50 m`, `y=-0.015--0.015 m`.
- Goal yaw delta: zero in the accepted split.  Full SO(3) is still checked, so
  tipping or rotating the hammer away from its support pose fails.
- Success: XY error `< 2 cm`, height error `< 1 cm`, full SO(3) error
  `< 0.1 rad`, held for five policy steps.
- C1: the hand may contact only the semantic safe region; hand contact with
  neutral/protected target points or proximal-arm physical contact invalidates
  the entire episode.
- Active clutter: none.  C2 and C3 are explicitly outside this release.

The actor jointly encodes 512 target points as
`[x,y,z,safe,protected]`, an inactive 512-point obstacle block, robot/hand
state, previous action, relative goal, and noisy target twist.  Exact dynamics
parameters are critic-only.  The accepted controller uses seven-dimensional
relative joint actions; this is a privileged oracle-affordance teacher, not the
RGB-D student.

### Frozen deterministic evaluation

Evaluation assigns exactly one terminal episode to each of 128 disjoint
manifest scenes.  It does not repeatedly count fast-resetting easy scenes.

| Seed | Checkpoint | Constrained success | C1 violations |
| ---: | :--- | ---: | ---: |
| 17 | v30 `model_359.pt` | 127/128 (99.22%) | 0/128 |
| 23 | v29 `model_498.pt` | 121/128 (94.53%) | 0/128 |
| 41 | v29 `model_498.pt` | 125/128 (97.66%) | 0/128 |
| **Total** | selected set | **373/384 (97.14%)** | **0/384** |

Legal safe-region contact occurs in 383/384 episodes.  These checkpoints are
short curriculum continuations of direction-competent policies; the table is
not a claim that all three selected policies were learned in one uninterrupted
run from random weights.

### Reproduce the C1 evaluation

Install Isaac Sim 5.0 and Isaac Lab 2.2.0, install this repository in editable
mode, then provide the external DOMINO and DAPL hand-point assets:

```bash
python -m pip install -e source/IsaacLab_nonPrehensile

export DOMINO_ROOT=/absolute/path/to/DOMINO
export DOMINO_USD_ROOT=$PWD/data/domino_usd
export DAPL_HAND_POINTS=/absolute/path/to/DAPL-dataset/embodiments/pc_npy_cache/hand_merged.npy

PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/prepare_domino_affordance_assets.py --headless \
  --manifest data/manifests/teacher_direction_curriculum_v10/hammer_teacher_dir45_eval128_seed9833.jsonl
```

Checkpoints are generated artifacts and are not committed to Git.  With one of
the selected checkpoints available locally, run:

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 \
PROFILE=c1_frozenv7_goalwrench_dir45 \
CHECKPOINT=/absolute/path/to/model_359.pt \
NUM_ENVS=128 NUM_EPISODES=128 SEED=17 \
bash scripts/evaluate_affordance_teacher.sh
```

Generate close-up videos with a translucent cyan goal and green/red semantic
overlays using:

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 \
CHECKPOINT=/absolute/path/to/model_359.pt \
OUTPUT_ROOT=outputs/teacher_demos/c1_accepted_randomized \
bash scripts/render_c1_randomized_demos.sh
```

## Contact-planner branch (M1/M2/M3)

The alternative VLM/perception + motion-planning route now has an M1 oracle
contact planner.  It explicitly samples contacts on the safe handle, separates
surface approach from push direction, solves Pinocchio IK, rejects unsafe
swept paths, performs short event-triggered pushes, and replans from observed
state.  No RL checkpoint is used.

Algorithmically this is **semantic contact sampling with receding-horizon
execution**, between the Sampling and SCSP rows of our method comparison.  It
is not yet full SCSP/CI-MPC: the current one-step translation/yaw proxy is not
reliable enough for accepted simultaneous XY+yaw convergence.  Contact/IK/C1
are the M1 claims; full-pose robustness, clutter, C2, C3, and RGB-D perception
remain later milestones.  See the exact interface, defaults, evidence, and
commands in [Contact Planner M1](docs/CONTACT_PLANNER_M1.md).

An experimental M2 now ranks contact-direction-distance candidates with
restored Isaac physics rollouts and a short shooting horizon.  Its C1 safety
checks pass, but strict simultaneous XY+orientation success is **not yet
accepted**; current evidence points to the straight-push primitive rather than
another scalar scoring problem.  See [Contact Planner M2](docs/CONTACT_PLANNER_M2.md)
for exact negative results and the contact-trajectory optimization next step.

M3 adopts the open-source Push Anything sampling + C3+ controller as that
contact-trajectory backbone instead of extending M2's fixed straight-push
family.  The upstream `dairlib`, C3, and Drake revisions are pinned; a
conservative exporter partitions DOMINO triangles into mutually-exclusive
safe/protected/neutral meshes, and a repository-owned upstream patch adds a
safe-only global sampling mesh without deleting physical protected geometry.
The real hammer export contains 2,233 safe, 2,229 protected, and 172 neutral
faces.  The three required Push Anything binaries now build and start through
a fully user-local Bazel 8.4.0 toolchain, with neither Docker, sudo, nor
Gurobi.  This validates the controller backbone but is not yet a strict-pose
or semantic-safety success claim.  See
[Contact Planner M3](docs/CONTACT_PLANNER_M3_C3PLUS.md).

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=8 \
  bash scripts/run_contact_planner_m1.sh

OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 VIDEO=1 \
  bash scripts/run_contact_planner_m1.sh

OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=1 \
  bash scripts/run_contact_planner_m2.sh
```

Training and evaluation outputs belong under ignored `logs/` and `outputs/`
directories.  Do not commit machine-local paths, converted USD assets,
checkpoints, W&B state, videos, or OptiX caches.

## Status and roadmap

| Component | Status |
| --- | --- |
| Oracle DOMINO safe/protected annotation | Implemented and audited |
| Single-hammer C1 teacher | Accepted, three seeds |
| C1 with physical clutter and wider directions | Experimental; not accepted |
| C2 clutter-to-protected safety | Experimental; not accepted |
| C3 robot-to-clutter avoidance | Experimental; not accepted |
| Semantic C3+ planner integration | Face-semantic bridge and native C3+ build verified; task reproduction pending |
| RGB-D affordance predictor and deployable student | Planned |
| DAPL-style dynamics model | Implemented and smoke-tested; not in C1 |

The planning route now first reproduces Push Anything on one hammer, adds C1
over the entire optimized contact trajectory, and only then adds C2 protected-
part and C3 whole-arm constraints in clutter.  RGB-D affordance prediction is
the final perception front-end.  Geometric success and C1/C2/C3 constrained
success must always be reported separately.

## Broader DAPL development

The repository also contains development code for reproducing
[DAPL](https://pku-epic.github.io/DAPL/) and components adapted from
[DyWA](https://pku-epic.github.io/DyWA/).  The physical world model, transition
collector, streaming trainer, evaluator, and frozen-encoder policy are
implemented and smoke-tested.  This is not yet a complete DAPL reproduction:
full policy training, the iterative curriculum, official benchmark splits,
and sim-to-real student remain in progress.  See
[the DAPL development status](docs/DAPL_DEVELOPMENT.md).

Legacy baseline preview:

![Legacy training video preview](asset/video.gif)

[Download / view the legacy full video](asset/video.mp4)


## Repository Structure

- `scripts/`: Training, evaluation, play, and shared CLI args
  - `train.py`: Training entrypoint
  - `eval.py`: Evaluation entrypoint (reports success rate and per-object stats)
  - `play.py`: Playback and export to JIT/ONNX
  - `cli_args.py`: Common RSL-RL CLI arguments
- `rsl_rl/`: RSL-RL components/utilities (aligned with Isaac Lab ecosystem)
- `source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/`: Python package (task registration, env definition, etc.)
- `logs/`: Training/eval logs, videos, and exported models
- `outputs/`: Optional script outputs


## Prerequisites

- Install Isaac-sim 5.0 (pip install recommended) and Isaac Lab 2.2.0 (Install from source code recommended). Official guide: `https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/source_installation.html`


## Install This Project (editable)

```bash
# From the repository root
python -m pip install -e source/IsaacLab_nonPrehensile
```

After installation, the Gym task is registered as follows:

```9:16:source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/isaaclab_nonprehensile/__init__.py
gym.register(
    id="Isaac-nonPrehensile-Franka-v0",
    entry_point=f"{__name__}.env:NonPrehensileEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env:NonPrehensileEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.config.rsl_rl_ppo_cfg:NonPrehensilePPORunnerCfg",
    },
)
```


## Data/Assets Paths (Important)

New DAPL code resolves assets from an explicit root rather than hard-coded
developer paths:

```bash
export DAPL_DATA_ROOT=/absolute/path/to/DAPL-dataset
```

The public asset release is
[`Steve3zz/DAPL-dataset`](https://huggingface.co/datasets/Steve3zz/DAPL-dataset).
It is roughly 153 GB, so do not download every directory unless the full asset
library is required. The release currently does not include Clutter6D scene
manifests; this repository defines a versioned manifest schema for generated
train/eval scenes.

The legacy single-object baseline uses the pre-converted
[DGN assets](https://huggingface.co/datasets/Steve3zz/DGN_usd). Point the
environment at the extracted directory instead of editing source paths:

```bash
export DGN_DATA_ROOT=/absolute/path/to/DGN
# Expected entries: yes.json, coacd_usd_convexhull/, coacd_normalized/
```

The checked-in DGN-backed Clutter6D integration fixture additionally uses:

```bash
export DAPL_CLUTTER_MANIFEST=$PWD/data/manifests/dgn_sparse_smoke_seed17.jsonl
export DAPL_CLUTTER_ASSET_SOURCE=dgn
python scripts/smoke_clutter6d.py --headless --num_envs 8 --steps 8
```

This fixture validates implementation plumbing only; it is not the released
Clutter6D benchmark split.

The four-stage affordance hammer curriculum imports DOMINO/RoboTwin tool assets
and maps their sparse contact/functional annotations to safe-contact and
protected-functional point regions. Its policy jointly encodes per-point
`[x,y,z,safe,protected]`, obstacle geometry, and robot/goal state; the default
runner uses 1,024 environments and includes a three-seed workflow. See
[the DOMINO affordance integration](docs/DOMINO_AFFORDANCE.md) for asset
conversion, manifest generation, constraints, and the approximation boundary.
The frozen single-hammer result, exact three-seed acceptance table, and
reproduction commands are summarized in
[the accepted C1 snapshot](docs/C1_ACCEPTED_SNAPSHOT.md).

Before downloading assets, the Isaac Lab environment and training pipeline can
be tested with a bundled procedural cube. This mode is intentionally opt-in and
must not be used for reported baseline or DAPL results:

```bash
export DAPL_USE_SMOKE_ASSET=1
```



## Quickstart

### 1. Environment Dependencies

After installing Isaac Lab, run the following commands to install specialized dependencies for the PTV3 encoder:

```bash
# Install Flash Attention (avoiding build isolation for CUDA compatibility)
pip install flash_attn==2.8.3 --no-build-isolation

# Install debugging tools
pip install icecream

# Build and install PyTorch3D from source (required for KNN operations).
# The upstream repository has no v0.7.8 tag; use the next official tag.
FORCE_CUDA=1 pip install \
  "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" \
  --no-build-isolation
```

Download the pretrained ICP encoder weights **`512-32-balanced-SAM-wd-5e-05-920`** from [Hugging Face (`imm-unicorn/corn-public`)](https://huggingface.co/imm-unicorn/corn-public/tree/main), then update `icp_weights_path` in `source/IsaacLab_nonPrehensile/IsaacLab_nonPrehensile/tasks/manager_based/isaaclab_nonprehensile/agents/config/rsl_rl_ppo_cfg.py` (default: `./ckpts/512-32-balanced-SAM-wd-5e-05-920`) to point to your local download directory.

The path can be supplied without editing source:

```bash
export DAPL_ICP_WEIGHTS=/absolute/path/to/512-32-balanced-SAM-wd-5e-05-920
```

For a pipeline-only smoke test, a randomly initialized, trainable ICP encoder
can be selected explicitly. It is not a replacement for the released weights:

```bash
export DAPL_USE_SMOKE_ASSET=1
export DAPL_USE_RANDOM_ICP=1
```

### 2. Path Configuration

To utilize the customized version of `rsl_rl` included in this repository, export the project root to your `PYTHONPATH`:

```bash
export PYTHONPATH=$HOME/IsaacLab_nonPrehensile:$PYTHONPATH
```

### Train (RSL-RL / PPO)

```bash
python scripts/train.py \
  --task=Isaac-nonPrehensile-Franka-v0 \
  --experiment_name=franka_nonprehensile \
  --num_envs=256 \
  --video --headless
```

Common options:
- `--video`, `--video_length`, `--video_interval`: record training videos
- `--seed`: random seed (`-1` to sample randomly)
- `--distributed`: multi-GPU/multi-node
- See `scripts/cli_args.py` for shared RSL-RL args (e.g., `--logger`, `--run_name`)

Training logs are saved under: `logs/rsl_rl/<experiment_name>/<time>[_run]`.

To train the current target-cloud privileged Clutter6D baseline:

```bash
export DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0
python scripts/train.py \
  --task=Isaac-Clutter6D-Franka-v0 \
  --num_envs=128 --max_iterations=500 --seed=17 --headless
```

Omit `DAPL_ENABLE_WORLD_MODEL_OBSERVATION=0` when collecting or inspecting the
non-concatenated physical scene tensor for world-model development.

Collect aligned `t -> t + 0.1 s` world-model transitions in bounded PyTorch
shards with:

```bash
python scripts/collect_dapl_transitions.py \
  --headless --num-envs=8 --steps=128 \
  --output-dir=outputs/dapl_transitions/smoke_seed17
```

For every transition, the future frame reuses the obstacle canonical-point
indices chosen at the current frame. The shard stores both the raw 7-D
relative-joint control and the paper-defined 3-D end-effector flow between the
two frames. Terminal/reset-crossing samples are excluded. The default
temporally correlated random actions are for data-pipeline development;
policy rollouts should replace them for the full world-model dataset.

For paper-aligned collection, point the scene builder at the released 256-point
Franka hand cache and load an RSL-RL policy checkpoint. Parallel environments
are distributed across all 16 manifest tasks by default:

```bash
export DAPL_HAND_POINTS=/path/to/embodiments/pc_npy_cache/hand_merged.npy
python scripts/collect_dapl_transitions.py \
  --headless --num-envs=128 --steps=469 \
  --action-mode=policy --policy-action=sample \
  --checkpoint=/path/to/model.pt \
  --output-dir=outputs/dapl_transitions/policy_seed17
```

If neither `DAPL_HAND_POINTS` nor a valid cache below `DAPL_DATA_ROOT` is
available, the scene uses an analytical two-finger fallback and records that
fact in each shard's `rollout.hand_point_source` metadata.

Train the dynamics model from the generated shards with:

```bash
python scripts/train_dapl_world_model.py \
  --data-dir=outputs/dapl_transitions/train \
  --output-dir=outputs/dapl_world_model/train_seed17 \
  --device=cuda:0 --batch-size=32 --max-steps=500000
```

The defaults match the paper specification: semantic FPS/kNN patches
`16 target / 16 obstacle / 8 end-effector` with `K=32`, 128-dimensional
tokens, a 12-block 8-head transformer, 3-D end-effector-flow conditioning,
and position/velocity/variance loss weights `1 / 1 / 100`. Training streams
one shard at a time and saves normalization statistics in each checkpoint.
The 10-step, eight-transition smoke test only validates the pipeline; a
full experiment requires policy rollouts at the paper's data scale.

Training writes an atomically replaced `world_model_best.pt` and a periodic
step checkpoint. It also logs a no-change persistence baseline. Continue an
interrupted run into a new output directory with `--resume=/path/to/checkpoint`;
`--max-steps` remains the absolute target step.

Evaluate the selected checkpoint on every transition in the checkpoint's
held-out shard split (including physical-unit, component-wise, persistence,
and zero-action diagnostics) with:

```bash
python scripts/evaluate_dapl_world_model.py \
  --checkpoint=outputs/dapl_world_model/train_seed17/world_model_best.pt \
  --device=cuda:0 --batch-size=32
```

After the world model is accepted, launch the paper-shaped frozen-encoder PPO
policy by exposing the checkpoint explicitly:

```bash
export DAPL_WORLD_MODEL_CHECKPOINT=$PWD/outputs/dapl_world_model/train_seed17/world_model_best.pt
python scripts/train.py \
  --task=Isaac-Clutter6D-DAPL-Franka-v0 \
  --num_envs=128 --seed=17 --headless
```

This policy consumes an 8,960-D flattened physical scene followed by the
paper's 44-D environment state. It cross-attends the state query over 40 frozen
dynamics tokens, uses fusion dimensions `[512, 256, 128]`, actor/critic hidden
dimension `[64]`, and the reported PPO hyperparameters.


### Evaluate (success rate + per-object stats)

A trained model is provided for testing: [Hugging Face (Steve3zz/DGN_usd)](https://huggingface.co/datasets/Steve3zz/DGN_usd).

```bash
# Automatically resolves checkpoint from the experiment folder
# (use --load_run/--checkpoint for precise selection)
python scripts/eval.py \
  --task=Isaac-nonPrehensile-Franka-v0 \
  --experiment_name=franka_nonprehensile \
  --num_envs=64 \
  --num_episodes=1000 \
  --load_run "your ckpt dir"
```

- Supports `--video` (only when `num_envs=1`) and `--real_time`
- Results are written to the run directory:
  - `eval_summary.json`: overall success rate
  - `eval_per_object.csv`: per-object success breakdown


### Play & Export (JIT/ONNX)

```bash
python scripts/play.py \
  --task=Isaac-nonPrehensile-Franka-v0 \
  --experiment_name=franka_nonprehensile \
  --num_envs 64 \
  --video --headless
```

- Exports to `logs/.../exported/`: `policy.pt` (JIT) and `policy.onnx`


### Random Agent (baseline/connectivity check)

```bash
python scripts/random_agent.py --task=Isaac-nonPrehensile-Franka-v0
```

> Note: A zero-action agent is not provided. You can adapt from `random_agent.py` if needed.




## Legacy upstream baseline results

The media and hardware/runtime statement below belong to the original
non-affordance baseline.  They are retained for upstream compatibility and are
not evidence for the accepted C1 result reported at the top of this README.

**Training video** — Demonstrates the legacy policy performing non-prehensile manipulation:

[Download / view the full video](asset/video.mp4)

**Training curve** — Reward and success rate vs. environment steps:

![Training curve](asset/curve.png)

These results were obtained on a **single RTX 4090D** with **4096 parallel environments**, trained for approximately **48 hours**.

## Environment Highlights (excerpt)

- Franka Panda joint workspace and initial pose are customizable
- Observations include: point cloud, hand state, robot state, last action, relative goal pose, physical params
- Rewards/terminations target goal tracking, contact shaping, and success completion
- Built-in success statistics and recent-window success rate


## Logging & Visualization

- Training/Eval logs: `logs/rsl_rl/<experiment_name>/<time>...`
- Videos: `logs/.../videos/{train|eval|play}`
- Exported models: `logs/.../exported/{policy.pt, policy.onnx}`

## License

Files in this repository include BSD-3-Clause license headers. Please use and distribute under the corresponding terms.


## References

```bibtex
@misc{zheng2026emergingextrinsicdexteritycluttered,
      title={Emerging Extrinsic Dexterity in Cluttered Scenes via Dynamics-aware Policy Learning},
      author={Yixin Zheng and Jiangran Lyu and Yifan Zhang and Jiayi Chen and Mi Yan and Yuntian Deng and Xuesong Shi and Xiaoguang Zhao and Yizhou Wang and Zhizheng Zhang and He Wang},
      year={2026},
      eprint={2603.09882},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.09882},
}
@article{cho2024corn,
  title={Corn: Contact-based object representation for nonprehensile manipulation of general unseen objects},
  author={Cho, Yoonyoung and Han, Junhyek and Cho, Yoontae and Kim, Beomjoon},
  journal={arXiv preprint arXiv:2403.10760},
  year={2024}
}
@inproceedings{lyu2025dywa,
  title={Dywa: Dynamics-adaptive world action model for generalizable non-prehensile manipulation},
  author={Lyu, Jiangran and Li, Ziming and Shi, Xuesong and Xu, Chaoyi and Wang, Yizhou and Wang, He},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11058--11068},
  year={2025}
}
```
