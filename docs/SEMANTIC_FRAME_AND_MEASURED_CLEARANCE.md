# Semantic frame and measured clearance follow-up

Gate 1 remains open. No strict debug success or frozen-50 acceptance result
has been established. All runs below are diagnostic scene007, outside the
acceptance manifest; the 31/50 and zero-C1 requirements are unchanged.

## Confirmed semantic coordinate bug

Native patch `0012-fix-semantic-guard-quaternion-order.patch` corrects two
Eigen quaternion constructors, in the sampling controller's unsafe-surface
distance calculation and the external semantic trajectory guard. Drake
state stores **wxyz**; Eigen's vector constructor reads **xyzw**. Both sites
now use explicit scalar w, x, y, z arguments. The sampler's existing scalar
constructors were already correct. C3 core solver sources are unchanged.

`scripts/validate_native_semantic_quaternion.py` extracts and compiles the
actual initializer from both native source files. Across 148 frame mappings
(yaw, tilt, quaternion sign and scale), maximum error was 0.197555 m before
the fix and 1.17872e-16 m afterward. The native online build passed. This
confirms a coordinate bug, not that it solely caused earlier C1 stops.

The old weight-10 long trial retained its previously mapped executable
during the rebuild. Its process executable hash remained
`5cf70d9e8f6908d8d2d73ebe08bce4a1a2abfe1bbbd7ac1ab3576350874995`,
matching its archived snapshot. The repaired binary hash is
`0546e46143684dc4d2d9ec55feeed46960c1493ec90f76acecb87c35db72683b`.
Trials `semantic_quat_fixed_q100_video180` and
`semantic_quat_fixed_q10_video180` use the repaired binary. The first stopped
at 80.5 s on C1 distance; the second completed 180 s with C1 pass and task
failure (12.579 mm XY, 0.31444 rad SO(3)). Its entire recorded trace matches
the old-binary weight-10 trial exactly. Neither established strict success.

## Completed additional trials

| Trial (before quaternion fix) | Executed time | Final XY | Final SO(3) | C1 | Strict task |
| --- | ---: | ---: | ---: | --- | --- |
| nonnegative_quat10_clearance5_35 | 35 s | 40.827 mm | 0.36040 rad | pass | fail |
| nonnegative_quat100_video180 | 57.53 s | 39.494 mm | 0.06140 rad | distance stop | fail |
| nonnegative_quat10_video180 | 180 s | 12.579 mm | 0.31444 rad | pass | fail |

The second trial did not complete 180 s. Its 57.5 s same-run video is
preserved; final measured contact forces were zero. The stop records the
conservative C1 proximity threshold and does not establish a forbidden
physical impact. The result field `forbidden_robot_contact_ever` includes
this oracle distance violation; do not reinterpret it as force evidence.

The repaired weight-100 trial subsequently stopped at 80.5 s: 32.323 mm XY,
0.03485 rad SO(3), C1 distance failure and zero measured contact force at the
last audit. Its complete 80.5 s / 1610-frame video is saved. The old weight-10
trial's complete 180 s / 3600-frame video is also saved. Both are 20 fps.

## Executed-plan prediction comparison

`scripts/analyze_c3_solver_consistency.py` pairs raw current C3 trajectories
with the same-timestamp current PD object rollout for executed C3 segments,
then compares both at knot 9 (0.675 s) with interpolated measured state.
There is no extrapolation beyond available measurements. For weight 10 and
100 short trials, raw median endpoint XY error was 125.997 and 130.343 mm;
PD median error was 3.743 and 9.854 mm. Initial raw/PD state disagreement was
below 1.7e-8 m. Thus the large difference is not simply an initial-position
offset or a mismatched extra PD tail knot.

Actual future motion includes subsequent MPC replans. These statistics are
not open-loop identification, and near-zero PD error on stationary samples
does not validate general contact prediction. No execution guard was changed
to use PD predictions in this follow-up.

## Actual finger clearance

`scripts/audit_finger_table_clearance.py` transforms the original cooked
PhysX body-frame convex vertices with **each finger's measured pose**,
including its actual joint opening. It measures surface clearance relative
to the environment's table plane. This supersedes the earlier approximation
that applied a closed-hand mesh to the measured hand pose.

| Trial | Recorded samples | Minimum surface clearance |
| --- | ---: | ---: |
| nonnegative_quat10_clearance5_35 | 424 | -0.00239 mm |
| nonnegative_quat100_video180 | 686 | -0.07651 mm |
| nonnegative_quat10_35 | 412 | -0.00416 mm |
| nonnegative_quat100_35 | 395 | -0.07651 mm |

At 29.5 s in the 5 mm-clearance trial, the commanded tip z was -11.501 mm,
the measured tip z was -16.556 mm, and tracking error was 5.055 mm. The
position command correctly used the height floor; it was not accidentally
replaced by the raw target. Downward feedforward remained -0.531 N while
reference vertical velocity was zero. A force/tracking interaction is a
plausible contributor, but its causal contribution has not been isolated.
Increasing reference clearance alone did not establish physical clearance.

These are sampled convex-surface measurements, excluding contact/rest offsets;
small negative clearance does not quantify contact force. Palm and arm
geometry are not included. No table-clearance guarantee is claimed.

The new clearance audit, solver pairing, and bridge tests passed together
(18 tests); the compiled native frame check passed all 148 mappings.
Detailed evidence is in
[the JSON record](evidence/contact_planner_m3_semantic_frame_fix_20260908.json).

## Missing native OSC guard in the Isaac execution chain

The online shell starts `franka_sampling_c3_controller` and the relay, which
subscribes to `TRACKING_TRAJECTORY_ACTOR`. It does not start
`franka_osc_controller`, where the native `SemanticC1TrajectoryGuard` is
instantiated. Thus the sampling controller's trajectory checks were present,
but the native high-rate measured-state shield was absent from Isaac OSC.
Correcting the external guard's quaternion constructor alone cannot affect
this execution chain.

At the repaired weight-100 failure, the measured tool reference was about
25.7 mm from the staged unsafe mesh, inside the configured 55 mm native stop
distance. Applying the native-equivalent guard to the measured and commanded
points returns hold/retreat (minimum over both points: 24.968 mm). This
offline check uses the final available object measurement, 50 ms older than
the tip audit. It is not proof that a counterfactual rollout would succeed.
See [the missing-guard evidence](evidence/contact_planner_m3_missing_osc_guard_20260908.json).

`semantic_trajectory_guard.py` now supplies the native hold/retreat calculation
in the Isaac runner at every servo step, using measured C3-frame object poses
and staged unsafe meshes. On hold, feedforward force and desired velocity
are zeroed. Existing robot reference speed and height limits still apply.
No goal-directed push is introduced, and the independent C1 audit is retained.
Malformed state/geometry fails before the next physics step. The evaluator
defaults to `--semantic-c1-guard-mode native-equivalent`; `disabled` preserves
the earlier integration as an explicit comparison. Mode, stop distance and
per-step activations are recorded and included in source/config snapshots.

Geometry tests cover safe pass-through, command-only hold, measured retreat,
translated/rotated/scaled-sign quaternions and missing/invalid state. Together
with acceptance, staging and bridge regressions, 29 tests passed. The
`native_osc_guard_q100_video180` completed 180 s with C1 pass and task failure
(20.3146 mm XY, 0.045665 rad). Its complete result, sampled guard outputs and
video have been audited. A separate native-tracking-gain 35 s trial completed with C1 pass
and all 13 recorded active guard samples showing zero force/velocity, but
task failure. See [the gain follow-up](OSC_GAINS_AND_PD_ROLLOUT.md).
