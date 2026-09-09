# Shared PhysX contact model

Implemented 2026-09-08. The C1 Isaac evaluator now uses the actual PhysX target
and closed-finger collision convexes in C3. This fixes the previously observed
millimetre-scale contact-geometry discrepancy. The task success gate remains
open.

## Implementation

`data/contact_models/closed_franka_hammer_physx_v1.json` contains the 16 target
convexes and two finger convexes extracted from the live Isaac simulation,
with source-export and asset hashes. The robot remains the stock fully
closed Franka gripper. Isaac physics assets are unchanged.

`scripts/stage_shared_physx_contact_model.py` stages an isolated runtime:

- Target convex vertices are rotated into the same support-baked object frame
  already used by C3. Both physical and controller SDF body volumes reference
  those exact convexes with `drake:declare_convex`.
- Finger vertices are expressed in hand axes about the existing task point,
  0.104279112 m from `panda_hand`.
- Original semantic sampling meshes, goal, objective, and solver settings are
  preserved. Unrelated runtime assets are symlinked; modified assets/configs
  are copied so the original runtime is not overwritten.

Patch `0007-closed-gripper-convex-contact.patch` adds optional convex fingers
to the translation-only C3 actor. They are massless welded children of its
existing point mass, with a context-parameter orientation frame. Franka FK
supplies measured hand orientation on each replan, and both numerical and
AutoDiff contexts receive that orientation. The state/action dimensions and
C3+ optimization remain unchanged. Orientation is held within the prediction
horizon; rotational contact dynamics are not optimized.

Both fingers are paired with every target convex (32 candidate contact
pairs). The upstream closest-contact resolver still selects the configured
number of active pairs. Mesh-normal sampling offsets use the oriented
fingers' support function instead of the old sphere radius; the sampled
semantic face and outward normal are unchanged. Candidate convex overlap is
checked directly. Point-query results explicitly exclude the added fingers,
preventing their classification as obstacles by the old index-based sampler.

`evaluate_push_anything_isaaclab_c1.py` selects this model by default and
records its hash. A model change cannot silently resume an old result set.
The executor records the shared contract and checks target asset/scale,
robot source, and task-frame offset. Changing assets requires a new export.
The current development export is v4: the same convexes plus measured
inertia/COM, bearing support points and effective pair friction. The evaluator
also selects the simulation clock, unified task-point OSC and velocity tracking.
See [the follow-up evidence](SIMULATION_DYNAMICS_ALIGNMENT.md).

For historical sphere/hand reproduction use `--legacy-sphere-contact-model`,
`--planner-clock-mode wall`, `--osc-control-point hand` and
`--no-osc-track-trajectory-velocity`, plus the original run settings. The
physical attached-sphere baseline retains its own geometry and also requires
an explicit wall-clock setting.

## Validation

- Native online build passed with the existing v151 Bazel cache.
- The patch applies to the canonical source and passes a dry-run against the
  existing extended checkout. Only the canonical native build was validated;
  the optional extended experiment stack is outside these rollouts.
- 38 tests passed: shared-shape staging/source preservation, geometric
  distance and frame checks, normal-validation rejection cases, protocol,
  bridge mapping, effect clock, and tracking diagnostics.
- 19 historical arm/object poses were queried in the compiled Drake model
  and probed for one PhysX step from zero velocity. All produced PhysX contact
  manifolds. Maximum positive Drake gap was 0.03369 mm, compared with the old
  sphere's maximum 9.921 mm against the same PhysX target convexes.
- Drake/PhysX minimum separation differed by at most 0.01970 mm. After the
  fixed reporter-normal sign conversion, the closest manifold normal differed
  by at most 4.2778 degrees. Hand orientation agreed within 1.151e-6 rad.
  Normal matching uses the closest normal in a multi-point manifold, not every
  point's normal. This does not reproduce the historical force trajectory.
- The default evaluator completed its full staging/bridge/native/Isaac path
  in a 1 s smoke, with 16 target convexes, two fingers, and all 10 trace samples
  carrying matching contact-audit timestamps. A maximum-time task failure is
  expected for this instrumentation smoke.

Artifacts are in `outputs/contact_planner_m3/shared_physx_contact_20260908/`.
`geometry_validation.json` contains the 19 per-pose checks;
`physx_contact_normals.json` contains raw signed forces, contact positions,
normals, and separations. `default_entry_smoke/` validates the default entry.

The corrected 35 s regression is `scene007_corrected_short/result.json`:
legal contact first at 18.87 s, final XY 59.944 mm, SO(3) 10.162 degrees,
C1 pass and task fail. It must not be reported as successful simulation
acceptance. Its wrapper's postprocessing was interrupted by an in-session
script edit after the result was written; the effect audit was regenerated
explicitly at the correct 0.075 s interval. The final shell script passes
syntax checking and the subsequent default-entry smoke completed normally.
`scene007_short/` is the earlier integration attempt with the self-query bug,
and is superseded by the corrected regression.

The 35 s velocity-reference comparison is `scene007_velocity_short/result.json`:
legal contact first at 3.01 s, final XY 38.121 mm, SO(3) 28.989 degrees,
C1 pass and task fail. Fresh-C3 governed tracking error has median 4.890 mm
and p95 8.132 mm. Position improved while rotation moved away from the goal;
this comparison does not establish pose convergence. Its staging used the
original full export in `runtime_v2/`, with the same convexes as the default
trimmed asset, but without the subsequently added asset-hash contract.

The complete default-entry velocity-reference regression is
`velocity_full180/scene007/result.json`. It completed all 180 s with zero
infrastructure failures, first legal contact at 3.01 s, final XY **38.000 mm**,
height **0.988 mm**, and SO(3) **43.968 degrees**. C1 passed; the strict pose
task failed, with zero dwell steps inside all pose thresholds. All 1800 trace
samples have matching measurement and contact-audit timestamps. Fresh-C3
governed tracking error has median 3.729 mm and p95 7.556 mm. Sampled contact
fractions describe trace rows, not physical contact duration.

![Scene007 pose error regressions](../outputs/contact_planner_m3/shared_physx_contact_20260908/scene007_pose_regressions.png)

These are individual trials on a frozen scene, not a statistical performance
comparison. The geometry fix does not establish strict task acceptance.
The machine-readable evidence and artifact hashes are in
[`contact_planner_m3_shared_physx_contact_20260908.json`](evidence/contact_planner_m3_shared_physx_contact_20260908.json).

Remaining approximations include the original target table-support spheres,
original C3 target dynamics parameters, fixed finger closure, and orientation
held within each horizon. Surface agreement is established; effective pushing
and joint-pose success still require validation.

`remaining_dynamics_comparison.json` records a separate parameter check:
both target masses are 0.05 kg, but C3 retains a zero body-frame COM while the
Isaac COM in that same support-baked frame is
[13.0954, 4.9780, 0.9120] mm. The COM disagreement is 14.0393 mm. This turn
does not change those dynamics, and the discrepancy alone does not establish
the cause of rollout failure.

## Run

Apply/build the pinned native checkout with the canonical profile:

```bash
bash scripts/apply_push_anything_patches.sh /data1/linsixu/dairlib-push-anything-canonical canonical
PUSH_ANYTHING_ROOT=/data1/linsixu/dairlib-push-anything-canonical \
PUSH_ANYTHING_PATCH_PROFILE=canonical PUSH_ANYTHING_BUILD_PROFILE=online \
bash scripts/build_push_anything_native.sh --build

/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python \
  scripts/evaluate_push_anything_isaaclab_c1.py \
  --manifest data/manifests/contact_planner_m3/hammer_c1_canonical_front180_screen8_seed20260907.jsonl \
  --scene-id scene007 --output-root outputs/contact_planner_m3/shared_contact_scene007_eval \
  --gpu 7 --base-port 8700 --max-sim-time-s 180
```

Use a free GPU and unused ports. Add `--osc-track-trajectory-velocity` for
the separately recorded velocity-reference comparison; it remains off by
default. Use a new output directory for each model/controller comparison.
