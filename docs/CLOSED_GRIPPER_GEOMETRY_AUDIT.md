# Closed Franka collision audit — 2026-09-08

Follow-up: the identified contact-geometry mismatches are now corrected in
the [shared PhysX contact model](SHARED_PHYSX_CONTACT_MODEL.md). This document
records the original diagnosis; its final pending-work paragraph describes
the state before that implementation.

The single 8 mm sphere omits contact surfaces of the actual closed fingers.
This is supported by the live PhysX collision geometry at 19 historical force
samples, after separating coordinate conversion, contact margins, and target
mesh differences. It does not establish that geometry is the only cause of
the unsuccessful task rollouts.

## Evidence

Artifacts are under `outputs/contact_planner_m3/closed_gripper_geometry_audit_20260908/`:

- `live_collision.json`: composed source meshes, cooked hulls, runtime
  offsets, initial joint/base states, and reconstructed historical poses.
- `comparison.json`: per-sample sphere gaps, convex-set distances, force
  magnitudes, closest finger, nearest points, and coordinate residuals.
- `contact_gap_comparison.png`: comparison across all 19 samples.
- `trace_smoke/result.json`: 1 s instrumentation smoke, not task acceptance.

| Quantity | Median | Maximum |
| --- | ---: | ---: |
| Sphere clearance to C3 collision mesh | 6.882 mm | 12.394 mm |
| Sphere clearance to actual PhysX target convexes | 5.717 mm | 9.921 mm |
| Closed finger convex distance to actual PhysX target convexes | 0.000106 mm | 0.033643 mm |
| Reconstructed proxy coordinate error | — | 0.000074 mm |

All nearest reconstructed contacts are on the left finger, matching the
filtered force channels in the historical logs. Convex distance is zero for
interpenetrating sets; it is not a penetration-depth measurement. Positions
are reconstructed from measured arm joints with both finger joints at zero.
Historical finger motion was not logged, so a small nonzero distance cannot
be interpreted as the exact gap during the original force event.

## What was checked

The local robot spawn override in `env.py` imports the packaged
`panda_arm_hand.urdf`, with fixed joints retained and normal independent finger
joints. The loaded finger collision meshes match the packaged STL bounds.
Both use `convexHull`; their runtime contact offsets are 0.420496 mm and rest
offsets are zero. The target has 16 `convexDecomposition` shapes, each with
contact offset 0.590199 mm and rest offset zero. Schema defaults are not used
as substitutes for runtime offsets.

The exporter traverses instance proxies and strips only the rigid-body pose
from composed USD transforms, preserving target scale 0.079. It queries the
PhysX cooking interface for actual convex geometry. It never relies on USD
world transforms for measured articulation motion when Fabric is enabled.
Measured body poses come from the physics tensor buffers. The live base has
identity orientation, position (0, 0, 0.029) m, and zero environment origin;
this validates the legacy proxy audit's frame assumption for these runs.

`prepare_domino_affordance_assets.py` passes `source.visual_mesh` to the USD
MeshConverter, while `export_domino_semantic_mesh.py` exports
`source.collision_mesh`. Thus Isaac and C3 share scale/support transforms but
use different source surfaces. Across USD source vertices, unsigned distance
to the C3 mesh is median 1.372 mm, p95 3.159 mm, maximum 11.995 mm. This is a
one-direction sampled surface comparison over the full object, not a
Hausdorff bound or a contact-specific error. Comparing the sphere directly
to PhysX convexes still leaves up to 9.921 mm of separation, independently of
that source-mesh difference.

The teacher ContactSensor has `history_length=0`, `update_period=0` and uses
`get_contact_force_matrix(dt=physics_dt)`. The executor reads the audit after
`env.step`; the three reference runs use audit stride 1 and trace stride 10.
The force belongs to the final 2.5 ms physics interval and the pose to the
end of that interval. There is no evidence of a stale history window here,
but within-interval contact location and normal cannot be reconstructed from
the historical trace. The new explicit contact timestamp lets the offline
proxy audit exclude stale audit samples when a future run uses a wider
audit stride.

## Reproduction

Use the Isaac Python environment and a free GPU for
`scripts/inspect_closed_gripper_collision.py`. It accepts the executor's
`--manifest`, `--output`, `--headless`, `--device`, and controller arguments,
plus `--reference-results` followed by the v2/v3/v4 result paths. Use
`--no-use-push-anything-end-effector`, `--local-controller cartesian_impedance`,
`--osc-inertial-dynamics-decoupling`, and `--osc-track-orientation`, as in
the reference runs. It builds the same environment without launching C3 or
executing a task rollout.

Run the CPU comparison with a Python containing NumPy, SciPy, trimesh and
rtree (the local `domino` environment):

```bash
/data1/linsixu/miniconda3/envs/domino/bin/python scripts/analyze_closed_gripper_geometry.py \
  outputs/contact_planner_m3/closed_gripper_geometry_audit_20260908/live_collision.json \
  --stage-manifest outputs/contact_planner_m3/gate1_closed_gripper_clock0_20260908_v2/scene007/stage_manifest.json \
  --mesh data/push_anything_semantics/020_hammer_0/full.obj \
  --output outputs/contact_planner_m3/closed_gripper_geometry_audit_20260908/comparison.json \
  --figure outputs/contact_planner_m3/closed_gripper_geometry_audit_20260908/contact_gap_comparison.png
```

The convex distance solver checks feasibility and the convex first-order
optimality condition. Tests cover separated, diagonal and overlapping hulls
after a common rotation/translation, signed inside/outside distance, and
nontrivial base/environment/support coordinate transforms. Twelve combined
geometry, effect-clock, and tracking tests passed. The 1 s live smoke confirms
the new trace fields and timestamp agreement; its maximum-time task failure
is expected and is excluded from all acceptance statistics.

The baseline proxy dimensions and physics remain unchanged. Its previous
description as a certified inscribed sphere was too strong and has been
corrected. Next model work must align the target asset contract and validate
finger contact positions/normals before another frozen-scene task trial.
