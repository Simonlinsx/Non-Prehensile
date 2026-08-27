# DOMINO affordance-aware pushing

This integration adds a DOMINO-backed task without changing the paper-fixed
DAPL physical tensor. It maps DOMINO/RoboTwin annotations as follows:

| DOMINO field | Task interpretation | Runtime representation |
| --- | --- | --- |
| `contact_points_pose` | robot-safe contact anchor | target point safe score |
| `functional_matrix` | protected functional/tool-end anchor | target point protected score and protected-surface clearance |
| `scale` | raw mesh-to-metre scale | Isaac Lab USD spawn scale and annotation scale |

Both fields contain sparse 4x4 poses in the unscaled object frame. DOMINO does
not release a per-triangle semantic mask for these assets. The audited
`020_hammer:0` therefore uses a deterministic canonical part mask: the main
handle is safe, the complete head and claw are protected, and the narrow neck
is neutral. Because this rule is evaluated from coordinates rather than saved
point indices, it remains aligned after every fresh surface-point sample. Other
assets still fall back to metric regions expanded from their sparse anchors.
This remains a point-cloud approximation, not an exact part-specific PhysX
contact sensor.

Reproduce the hammer mask audit and the old/new comparison figure with:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/audit_domino_affordance_annotation.py \
  --domino-root /data1/linsixu/DOMINO \
  --output-prefix outputs/affordance_annotation_audit/hammer_020_part_mask_v2
```

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
  --curriculum-stage 0 --scene-count 2 --seed 23 --track sparse
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

The primary proof task is `Isaac-AffordanceHammer-Pose-Franka-v0`.  Its success
definition is fixed for the entire run:

```text
XY error:                < 0.02 m
absolute height error:   < 0.01 m
full SO(3) error:        < 0.10 rad
required dwell:          5 consecutive policy steps
```

The full quaternion distance includes yaw, roll, and pitch.  The older task ids
remain as checkpoint-compatible aliases; their pose threshold is no longer
stage-dependent.  Their numeric level now changes only contact/clutter
constraints:

| Level | Gym task | Pose objective | Constraints |
| --- | --- | --- | --- |
| 0 | `Isaac-AffordanceHammer-Pose-Franka-v0` | strict pose + dwell | single target, soft illegal-contact penalty |
| 1 | `Isaac-AffordanceHammer-Yaw-Franka-v0` | strict pose + dwell | single target, soft illegal-contact penalty |
| 2 | `Isaac-AffordanceHammer-Avoid-Franka-v0` | strict pose + dwell | hard illegal robot contact, soft protected-part clearance |
| 3 | `Isaac-AffordanceHammer-Clutter-Franka-v0` | strict pose + dwell | hard illegal robot and protected-part contact |

All default to 1,024 parallel environments. The legacy task id
`Isaac-AffordanceClutter6D-Franka-v0` is retained as an alias for the final
stage.

## Observation and constraint contract

The policy observation is a fixed 4,140-D vector in every stage:

```text
target:    512 x [x, y, z, safe, protected] = 2560
obstacles: 512 x [x, y, z]                  = 1536
state:     hand + robot + action + goal + physics = 44
```

`ActorCriticAffordance` applies a shared PointNet to each point before pooling,
so geometry and semantics cannot become detached. The 50-D state (including
target linear and angular velocity) queries both
the target and obstacle tokens with cross-attention; max-pooled features retain
a global geometry summary. This replaces the earlier ICP layout, which encoded
target XYZ before the flattened semantic scores were appended.

The world-model observation remains `[B, 1280, 7]`. The semantic term is a
separate `world_model.target_affordance` entry, preserving existing DAPL model
and checkpoint compatibility.

The final constraint level defines two hard failure terms:

1. `forbidden_region_contact`: any end-effector surface point is in target
   contact proximity and its nearest target point has insufficient safe score.
2. `protected_region_collision`: any protected target-surface point violates
   the required clearance from an obstacle point cloud.

The corresponding soft reward penalties are introduced before their hard
terminations. Safe-region approach uses a bounded linear distance score plus
signed per-step distance progress, avoiding the far-field saturation of the
former `1 - tanh(distance / 0.1)` term. A continuous protected-clearance cost
starts at stage 2. The environment reports `constrained_success_rate` and
`affordance_violation_rate`; ordinary geometric `success_rate` remains
available for comparison.

Default geometric parameters live in `affordance_env.py`:

```text
robot-target contact distance: 0.010 m
minimum safe score:             0.25
protected obstacle clearance:   0.005 m
anchor region radius:           max(0.015 m, 10% of annotated max extent)
```

For exact part-level contact in a later version, segment each tool into fixed
safe/functional collision sub-bodies and attach filtered PhysX contact sensors.
The current sparse annotations cannot supply those collision subsets by
themselves.

## Generated scenes and three-seed training

The checked-in curriculum contains one deliberately simple stage-0 scene and
128 scenes in each later stage. Every target is `020_hammer:0`; each initial and
goal pose uses the same stable support face. The obstacle pool is restricted to
the six DOMINO assets already converted under `data/domino_usd`; regenerating
with the script defaults expands the pool to twelve assets after USD conversion.

```text
data/manifests/domino_hammer_stage0_xy_seed1701.jsonl
data/manifests/domino_hammer_stage1_yaw_128_seed1702.jsonl
data/manifests/domino_hammer_stage2_avoid_128_seed1703.jsonl
data/manifests/domino_hammer_stage3_clutter_128_seed1704.jsonl
data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl
```

The first single-hammer proof deliberately uses 128 identical entries from the
last manifest above, so every vectorized environment receives one controlled
task: 8 cm translation along +X and -0.15 rad yaw on the same support face.
This isolates policy learning from scene-distribution difficulty; it is not the
final 128-scene diversity benchmark. Runtime obstacles are disabled by the Pose
task.

The Pose task uses trimesh's highest-probability hammer support face.  A prior
manifest copied a narrow candidate that tipped in PhysX from root Z 6.69 cm to
1.29 cm before contact, making the strict height/orientation goal impossible.
The v3 manifest stays on the 1.29 cm support face and places the longitudinal
safe handle across the robot's push direction so an off-centre safe push can
create the requested yaw.  The Franka reset is a verified vertical pre-contact
pose with TCP `[0.387, 0.000, 0.025]` m. The strict success definition remains
XY + height + full SO(3) + five-step dwell.
The proof-only controller uses a 0.03-rad relative joint-action scale and 0.35
initial policy noise, preventing random exploration from launching the hammer.
Its continuous safe-distance cost has weight -2, signed contact-distance
progress has weight 8, the first safe contact pays a one-time event bonus, and
one joint XY/Z/SO(3) potential has weight 16. Illegal contact has a soft weight
of -15; stage 0 still does not terminate on illegal contact.

Early controlled runs revealed two local optima. First, absolute contact and
pose scores paid reward while the hammer remained stationary. Second, the old
positive distance score paid more at 1.6 cm from the safe surface than at
contact, where it was abruptly gated to zero. The current shaping uses signed,
reset-safe progress and a zero-at-contact distance excess penalty, so there is
no positive reward cliff to hover beside. It also uses a one-shot contact event,
joint XY/Z/SO(3) tracking, and action magnitude/rate costs. The strict success
thresholds are unchanged.

Regenerate the controlled manifest with:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
DOMINO_ROOT=/data1/linsixu/DOMINO \
python scripts/generate_domino_hammer_proof_manifest.py \
  --base-manifest data/manifests/domino_hammer_stage1_yaw_128_seed1702.jsonl \
  --output data/manifests/domino_hammer_joint_pose_proof_128_v3_stable.jsonl
```

Run seeds 17, 23, and 41 sequentially on one GPU:

```bash
GPU_ID=7 NUM_ENVS=1024 MAX_ITERATIONS=5000 \
  OMNI_KIT_ACCEPT_EULA=YES \
  bash scripts/train_domino_affordance_joint_pose.sh
```

Set `OMNI_KIT_ACCEPT_EULA=YES` only after reviewing and accepting NVIDIA's
Omniverse license. The training script intentionally does not accept it on the
user's behalf.

Iteration counts and seeds can be overridden without editing the script, for
example `SEEDS="17" MAX_ITERATIONS=2` for a bounded pipeline check.
