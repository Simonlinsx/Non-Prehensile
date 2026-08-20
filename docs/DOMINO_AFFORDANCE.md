# DOMINO affordance-aware pushing

This integration adds a DOMINO-backed task without changing the paper-fixed
DAPL physical tensor. It maps DOMINO/RoboTwin annotations as follows:

| DOMINO field | Task interpretation | Runtime representation |
| --- | --- | --- |
| `contact_points_pose` | robot-safe contact anchor | target point safe score |
| `functional_matrix` | protected functional/tool-end anchor | target point protected score and protected-surface clearance |
| `scale` | raw mesh-to-metre scale | Isaac Lab USD spawn scale and annotation scale |

Both fields contain sparse 4x4 poses in the unscaled object frame. DOMINO does
not release a per-triangle semantic mask for these assets. Consequently, the
current implementation expands each anchor into a metric region and evaluates
point-cloud proximity. It is an explicit approximation, not an exact
part-specific PhysX contact sensor.

## Available tool assets

The local DOMINO/RoboTwin release contains annotated hammer, screwdriver,
knife, fork, drill, small shovel, wooden mallet, skillet, and related objects.
It contains no object directory named scissors, so scissors need a new asset
and annotation before they can be included. The default smoke task uses
`020_hammer:0` as the target.

Portable asset ids use `<object-directory>:<model-id>`, for example:

```text
020_hammer:0
032_screwdriver:0
034_knife:0
```

## Prepare and run

Set paths explicitly. `DOMINO_ROOT` can point at the DOMINO checkout or its
`assets/objects` directory. Converted USD files are kept outside source asset
directories.

```bash
export DOMINO_ROOT=/data1/linsixu/DOMINO
export DOMINO_USD_ROOT=$PWD/data/domino_usd
export DAPL_CLUTTER_MANIFEST=$PWD/data/manifests/domino_hammer_sparse_smoke_seed23.jsonl
export DAPL_CLUTTER_ASSET_SOURCE=domino
```

Generate a new deterministic manifest if needed:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/generate_domino_affordance_manifest.py \
  --output data/manifests/domino_hammer_sparse_smoke_seed23.jsonl \
  --scene-count 2 --seed 23 --track sparse
```

The generator keeps the target on the same stable support face between its
initial and goal poses. This avoids producing push-only tasks that require the
tool to flip onto another face.

Convert all GLB/OBJ assets referenced by the manifest. Isaac Sim may require
the user to accept NVIDIA's Omniverse EULA on first use; acceptance must be
provided by the user.

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/prepare_domino_affordance_assets.py --headless \
  --manifest "$DAPL_CLUTTER_MANIFEST"
```

Then run the bounded task check:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/smoke_domino_affordance.py --headless --num_envs 2 --steps 8
```

The registered Gym task is `Isaac-AffordanceClutter6D-Franka-v0`.

## Observation and constraint contract

The policy receives the original target cloud and a flattened 1,024-D semantic
term. Reshape the latter to `[B, 512, 2]`; the last dimension is
`[safe_contact_score, protected_functional_score]` and uses exactly the same
canonical point ordering as the target point cloud.

The world-model observation remains `[B, 1280, 7]`. The semantic term is a
separate `world_model.target_affordance` entry, preserving existing DAPL model
and checkpoint compatibility.

The task defines two hard failure terms:

1. `forbidden_region_contact`: the end-effector point cloud is in target
   contact proximity, but the closest target point has insufficient safe score.
2. `protected_region_collision`: one of the 64 highest-scoring protected target
   surface points violates the required clearance from any obstacle point cloud.

The corresponding reward penalties are also enabled for learning signal. The
environment reports `constrained_success_rate` and
`affordance_violation_rate`; ordinary geometric `success_rate` remains
available for comparison.

Default geometric parameters live in `affordance_env.py`:

```text
robot-target contact distance: 0.008 m
minimum safe score:             0.25
protected obstacle clearance:   0.005 m
anchor region radius:           max(0.015 m, 10% of annotated max extent)
```

For exact part-level contact in a later version, segment each tool into fixed
safe/functional collision sub-bodies and attach filtered PhysX contact sensors.
The current sparse annotations cannot supply those collision subsets by
themselves.
