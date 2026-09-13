# FR3 semantic pushing simulation release — 2026-09-13

Start with [the deployment handoff](../../docs/FR3_SIM_TO_REAL_HANDOFF_20260913.md).

This directory contains the exact tested controller/environment source, scene manifests, cached semantic rules, collision/render assets and independent audits. `source_manifest.json` pins 79 source files. `run.py` is a new simulation-only launcher; it never opens hardware. Root-level historical experiments are not the release entry point.

## Prerequisites

- Git LFS assets: run `git lfs pull` after checkout. `assets/legacy` preserves the previous scenes; `assets/sitting` contains the upright 22.7 cm teddy.
- The existing IsaacLab/Isaac Sim environment, PyTorch, Pinocchio/hppfcl, trimesh, NumPy/SciPy, and renderer dependencies. Exact observed package versions are in `evidence/environment.json`.
- The DOMINO dataset and converted USD assets. These external datasets/caches are not included. On the experiment server use `/data1/linsixu/DOMINO` and `outputs/fr3_semantic_map_20260913/source_choice_eval_v1/data/domino_usd`.
- FR3 stock-hand model: `data/robot_models/fr3_stock_20260909/robot_model_manifest.json`. Its checked URDFs and mesh paths currently use the original absolute checkout location. On another machine, regenerate the model/URDF paths and hashes with the repository's FR3 preparation workflow; do not disable the hash checks. Relocation to another machine has not been tested.

## Run on the existing simulation server

From the repository root, with `dapl-isaaclab` activated:

```bash
python releases/fr3_semantic_20260913/run.py \
  --example sitting_doll --gpu 0 \
  --usd-root outputs/fr3_semantic_map_20260913/source_choice_eval_v1/data/domino_usd \
  --output outputs/fr3_release/my_sitting_demo --execute
```

The output directory must not contain a previous `process.log`. Omit `--execute` to validate inputs and write a reviewable `launch.json`. Use `--no-video` to avoid recording. `--weight 0` runs the paired hard-only controller; `--weight 0.5` is the frozen semantic cost setting. Other examples: `optional_contact`, `doll_side`, `doll_front`, `bowl_sponge`.

Exact held-out cases are included as manifests. To replay case 3:

```bash
python releases/fr3_semantic_20260913/run.py \
  --holdout-case 3 --weight 0.5 --no-video --gpu 0 \
  --usd-root outputs/fr3_semantic_map_20260913/source_choice_eval_v1/data/domino_usd \
  --output outputs/fr3_release/holdout003_soft --execute
```

The default budget remains 60 simulation seconds / 10 replans / 900 wall seconds. `--sim-seconds` is only a diagnostic budget override; a shortened trial is not part of the success-rate cohort. Every executed attempt is independently audited, and Python exceptions/missing artifacts cannot silently count as task outcomes.

## Verify and preview

```bash
cd releases/fr3_semantic_20260913
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests
python -m http.server 8766 --bind 127.0.0.1
```

For VS Code Remote SSH, forward port **8766** in the **Ports** panel and click **Open in Browser** on that row. Open `evidence/index.html`. Use the local address shown by VS Code; it may choose a different local port. Opening a server filesystem path or the HTML source editor on your laptop does not create port forwarding.

The original server gallery remains `outputs/fr3_semantic_map_20260913/index.html`, served on port 8765 during this session. Uploaded evidence is a separate smaller gallery with relative links; full physics/query logs remain on the experiment server.

## Interpretation

The original 20/20 per group is a development regression over four known layouts and five workspace translations. New relative layouts/goal angles are reported separately in `evidence/holdout20`; do not combine the cohorts. The sitting-teddy success is one development demonstration. Semantic inference was provided by assistant review/cached rules; geometry came from simulation truth. Real RGB-D pose tracking, hardware contact evidence, and the pushing-to-Franka session adapter still need integration.
