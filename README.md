## Project Overview

This repository is the development codebase for reproducing [DAPL](https://pku-epic.github.io/DAPL/) (Dynamics-Aware Policy Learning) in Isaac Lab. It now contains both the legacy single-object task and a runnable manifest-backed `Isaac-Clutter6D-Franka-v0` integration task. The latter spawns one target plus multiple obstacles, applies the paper's task/reward contract, and exposes a separate `[B, 1280, 7]` physical-scene observation for world-model development.

The existing baseline borrows components from [DyWA](https://pku-epic.github.io/DyWA/). The paper-aligned physical world model, transition collector, streaming trainer, full-split evaluator, and frozen-encoder cross-attention actor-critic are implemented and smoke-tested. This should still not be confused with a complete DAPL reproduction: full policy training, the iterative curriculum, official benchmark assets/scenes, and the sim2real student remain in progress. See [the DAPL development status](docs/DAPL_DEVELOPMENT.md) for the exact contract and roadmap.

**Demo (preview)**:

![Training video preview](asset/video.gif)

[Download / view the full video](asset/video.mp4)

## Possible Extensions

This codebase is intended as a flexible template for contact-rich non-prehensile manipulation.  
On top of this implementation, it should be straightforward to:

- Reproduce several recent non-prehensile manipulation papers based on this CORN-style pipeline (specific papers to be listed here).
- Swap in alternative point-cloud encoders (e.g., PointNet, Point Transformer, MAE-style encoders) while reusing the same Isaac Lab task, reward, and evaluation pipeline.
- Prototype new RL / IL algorithms that operate on the same observation and command interface.

We plan to add concrete pointers to specific papers and corresponding configuration files in future updates.


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




## Training Results

**Training video** — Demonstrates the learned policy performing non-prehensile manipulation:

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
