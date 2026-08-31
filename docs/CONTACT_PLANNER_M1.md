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

## Conservative default contract

| Parameter | Default |
| --- | ---: |
| Exact C1 contact distance | 0.010 m |
| Protected/neutral clearance | 0.020 m |
| Pre-contact approach clearance | 0.015 m |
| Support clearance | 0.002 m |
| Short push distance | 0.008--0.015 m |
| Closed-loop replans | 12 |
| Candidate outputs | 16 |
| Online scalar dynamics adaptation | disabled |

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
| Candidate + IK path available | 8/8 in the default seed-17 regression |
| Reach a legal safe contact without C1 violation | 8/8 contacts, 0/8 C1 violations |
| Robust simultaneous XY + yaw task success | **Not accepted** |
| Clutter, C2, and C3 | Out of M1 scope |

In a representative closed-loop diagnostic, the planner reached a minimum XY
error of 2.45 mm and a minimum rotation error of 0.011 rad with zero C1
violations, but those minima did not occur simultaneously; final rotation was
0.434 rad.  Larger pushes also passed through good XY/yaw states and then
diverged.  This is evidence that contact generation and execution work, but a
single scalar translation/yaw proxy is not a sufficiently predictive dynamics
model for full-pose convergence.

The finalized-default eight-scene regression reached a legal safe contact in
all 8 scenes with 0 C1 violations.  The strict contact gate passed 67/96
planned approaches (69.8%); 2/8 trajectories individually entered the XY
threshold and 7/8 individually entered the rotation threshold, but 0/8 held
both thresholds together.  These separate minima must not be reported as task
success.

The next milestone should keep this M1 contact/path safety layer and replace
candidate ranking with short Isaac physics rollouts or a learned local
contact-conditioned dynamics model.  That is a contained model upgrade, not a
reason to weaken C1 or add ad-hoc waypoints.
