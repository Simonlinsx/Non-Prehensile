# Contact Planner M1: Oracle Safe Contact

## Scope

M1 is the first milestone of the VLM/perception + motion-planning route.  It
isolates the contact-planning problem from RGB-D perception and learned
dynamics:

- one DOMINO hammer on a planar support;
- no clutter;
- oracle target pose, metric target point cloud, and per-point
  `safe`/`protected` labels;
- a Franka hand may contact only the safe handle (C1);
- strict task pose remains XY `< 2 cm`, height `< 1 cm`, full SO(3)
  `< 0.1 rad`, held for five simulator steps.

M1 does not use an RL checkpoint, a VLM, an affordance predictor, or a learned
world model.  It is deliberately the oracle upper-bound interface that those
modules will feed later.

## What kind of planner is this?

The implementation is best described as **semantic contact sampling plus
kinematic/swept-path validation and receding-horizon execution**.  In the
comparison used in this project, it lies between Sampling and SCSP:

| Property | M1 implementation |
| --- | --- |
| Contact choice | Explicit samples on the oracle-safe object surface |
| Semantic constraint | Hard rejection of neutral/protected contact |
| Hand pose | Sampled wrist yaw, then Pinocchio endpoint IK |
| Motion prediction | One-step planar translation + moment-arm yaw proxy |
| Path safety | Sampled hand swept-volume clearance against target and support |
| Execution | Approach, event-triggered contact, short push, outward retreat, replan |

It is **not** a full SCSP contact-implicit optimizer, CI-MPC, trajectory
optimizer, or DyWA learned policy.  The current one-step physics proxy is the
main reason it cannot yet robustly make XY and yaw simultaneously converge.

## Interface

`OraclePlanningScene` receives batched tensors:

- target points `[B,N,3]` in the environment frame;
- `safe_scores` and `protected_scores` `[B,N]`;
- current target position, goal position, TCP position, and optional yaw error;
- hand surface points in the TCP frame and the live TCP rotation.

`OracleContactCandidateBatch` returns ranked, padded candidates containing:

- semantic contact point and target/hand point indices;
- contact, pre-contact, and short-push TCP endpoints;
- independent approach and push directions;
- wrist rotation/yaw mode;
- safe, protected, support, and approach clearances;
- predicted planar/yaw residual and contact moment arm.

No-safe-point and no-safe-path cases fail closed: the planner returns no valid
candidate and the executor does not move.

## Planning and execution sequence

1. Sample push directions around the target-to-goal direction.
2. Sample contacts only from points where `safe >= 0.25` and
   `protected < 0.25`.
3. Approach from the contact surface's outward normal; optimize the push
   direction independently.
4. Sample wrist-yaw modes and reject contact configurations that violate the
   target protected clearance or support-plane clearance.
5. Rank candidates with predicted XY/yaw residual, travel, direction change,
   and clearance.
6. Solve pre-contact, contact, push, and post-push outward-retreat endpoints
   with Pinocchio IK.
7. Recheck every joint-space segment against semantic and support clearances.
8. Move to pre-contact, close until the first legal safe contact, perform a
   short push, retreat away from the translated object, observe, and replan.

The Pinocchio `panda_hand` frame is composed with the 0.1034 m local TCP
offset during forward kinematics.  This avoids the orientation-dependent
9--10 cm error caused by treating that offset as a fixed world translation.

## Reproducible execution profiles

The shell entry point exposes two profiles while keeping the C1 gate, IK/path
validation, and the accepted control timing identical:

| Parameter | `adaptive` (default) | `fixed5mm` baseline |
| --- | ---: | ---: |
| Exact C1 contact distance | 0.010 m | 0.010 m |
| Protected/neutral clearance | 0.020 m | 0.020 m |
| Pre-contact approach clearance | 0.015 m | 0.015 m |
| Support clearance | 0.002 m | 0.002 m |
| Push distance hypotheses | 8 / 11.5 / 15 mm | 5 mm |
| Closed-loop replans | 30 | 30 |
| Candidate outputs | 32 | 32 |
| Online scalar dynamics adaptation | disabled | disabled |

On the fixed `scene000`, seed 486317 A/B, the adaptive profile reached the
unchanged strict pose gate in 13 pushes and 1,347 simulator steps, versus 21
pushes and 2,164 steps for `fixed5mm` (37.8% fewer pushes and 37.8% less
simulated task time).  Final errors were 19.87 mm / 4.79 degrees for adaptive
and 19.68 mm / 3.58 degrees for fixed5mm; both had 100% legal contact-gate
passes and zero C1 violations.

A separate attempt to shorten approach/contact/push/retreat interpolation
failed the strict pose gate after 30 pushes despite 30/30 legal contacts.  It is
therefore rejected: the faster profile changes push distance only and does not
speed up the contact servo.

Extending the distance set to 5 / 10 / 15 / 20 mm was also rejected on the
same scene.  It reached a 3.36 mm minimum XY error but failed to make XY and
SO(3) valid simultaneously after 30 legal pushes (final 23.96 mm / 8.71
degrees, zero C1 violations).  The accepted 15 mm cap is therefore empirical,
not an arbitrary speed limit.

The contact event gate is also 0.010 m by default.  A looser 0.013 m gate was
tested as a diagnostic for point-cloud/mesh mismatch, but it is not the safe
default because it can spend pushes without a physical contact while strict C1
accounting still uses 0.010 m.

## Run

Quantitative eight-scene run:

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=8 \
  bash scripts/run_contact_planner_m1.sh
```

Conservative 5 mm control baseline:

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=8 \
  EXECUTION_PROFILE=fixed5mm \
  bash scripts/run_contact_planner_m1.sh
```

Persistent-contact execution (experimental efficiency profile):

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=8 \
  EXECUTION_PROFILE=persistent \
  bash scripts/run_contact_planner_m1.sh
```

This profile keeps the first measured C1-legal contact, reads the live target
pose every three control steps, and advances the TCP in 2 mm increments. It
recomputes a bounded XY/yaw contact wrench from the live pressure-center
estimate and releases when contact is lost, the pose regresses, the requested
moment is not physically realizable, or the motion ceases to advance toward
the goal. A rejected first micro-step falls back to the accepted short macro
push, so the original executor remains available on difficult contact sides.

Single-scene video:

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 VIDEO=1 \
  RUN_LABEL=m1_oracle_c1_video \
  bash scripts/run_contact_planner_m1.sh
```

The video uses a cyan translucent hammer for the goal, green points for the
safe handle, red points for the protected tool end, and yellow/orange/purple
points for the selected contact/pre-contact/push endpoints.  All overlays are
non-physical.

## Acceptance boundary

M1 has two deliberately separate decisions:

| Claim | Status |
| --- | --- |
| Pure contact candidate generation and fail-closed checks | Passed (9 unit tests) |
| Correct TCP IK and endpoint tracking | Passed in Isaac smoke tests |
| Candidate + IK path available | Passed on the accepted single-scene run |
| Reach a legal safe contact without C1 violation | 13/13 adaptive pushes, zero C1 violations |
| Strict simultaneous XY + yaw task success | Accepted on one deterministic scene; randomized rate pending |
| Persistent-contact efficiency | Same scene: 2 contacts / 318 steps versus 13 contacts / 1347 steps; zero C1 violations |
| Clutter, C2, and C3 | Out of M1 scope |

Earlier regressions used an incomplete end-effector envelope and a 0.5 N
physical-contact threshold, so legal contact was either rejected or silently
missed.  The accepted run instead separates the full Franka hand/finger
collision envelope from the finger-only deliberate-contact surface and uses a
0.02 N measured-contact threshold.  Those geometry/contact fixes, rather than
a weaker pose or C1 gate, produced the first simultaneous XY+yaw success.

The current acceptance claim remains intentionally narrow: the deterministic
scene succeeds with both execution profiles, while the fixed 50-scene
direction/distance/yaw evaluation is still required before calling M1 robust.
On the first four fixed scenes and a 15-contact budget, persistent execution
matched the macro baseline's 2/4 success count while reducing the successful
scenes from 14/4 contacts to 2/1. It is therefore an opt-in efficiency profile,
not yet the default executor.
