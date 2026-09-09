# Closed-gripper simulation gates

Latest follow-up: [PD inertia and contact calibration](PD_INERTIA_AND_CONTACT_CALIBRATION.md).
Refreshing contact geometry with matched scalar inertia improved same-plan
prediction errors in two overlapping diagnostic windows. Active closed-loop
cost and coarse-model comparisons are running; strict debug successes remain
zero and the frozen 50-scene acceptance set has not started.

Updated 2026-09-08. This record describes the current stock closed-Franka
gripper + safe sampling + canonical Push Anything/C3+ route. Historical RL,
attached-sphere, and contact-servo successes are separate experiments.

## Acceptance order

1. Target-only C1: establish correct contact execution and strict joint pose
   success on the fixed diagnostic scene, then reproduce with same-run video.
2. Frozen 50-scene target-only C1 evaluation: relative goal direction +/-90
   degrees, randomized initial/goal yaw, maximum 180 simulation seconds per
   scene. The edited user goal specifies >50% but retains at least 31/50;
   both conditions apply pending count clarification. Zero C1 violations
   remain required.
   See [the frozen acceptance protocol](SIMULATION_ACCEPTANCE_GOAL.md).
3. Randomized clutter, including objects close to or touching the target.
4. C1 hard constraint plus C2/C3 soft trajectory costs, with task and safety
   outcomes reported independently.
5. RGB-D replay and hardware preparation only after simulation acceptance.

Gate 1 remains open. Existing narrower-direction manifests and fixed-initial-yaw
results must not be relabeled as passing gate 2. The current executor's gate is
20 mm XY, 10 mm height, 0.105 rad full SO(3), and 0.5 s continuous dwell, plus
observed legal contact and no C1 violation. Thresholds remain unchanged in the
bridge comparisons below. Native Drake acceptance uses its own 0.1 rad gate.

## Current dynamics and execution follow-up

The [progress-window and fixed-goal follow-up](PROGRESS_WINDOW_AND_FIXED_GOAL.md)
records a complete guarded 180 s run with C1 pass but a strict position miss
(20.3146 mm). Shortening the native progress window also failed the task.
Staging now explicitly sets native fixed-goal mode 2; the old random mode
could advance goals before the required dwell. A new rotation-stiffness-100
trial uses the corrected fixed-goal contract. Gate 1 remains open.

The [semantic-frame and measured-clearance follow-up](SEMANTIC_FRAME_AND_MEASURED_CLEARANCE.md)
fixes a confirmed wxyz/xyzw constructor mismatch in two C1 guard sites.
The native build and 148 compiled frame checks pass; repaired-binary
weight-100 testing stopped at 80.5 s on C1 distance, and weight 10 completed
180 s with task failure and C1 pass. The audit then identified a missing high-rate native OSC guard
in the Isaac chain. Its equivalent is now wired into the executor and a
new weight-100 video trial has started. No configuration has passed gate 1.
Actual per-finger geometry reconstruction also confirms that commanded
height clearance does not guarantee measured clearance.

The [OSC gain and PD rollout comparison](OSC_GAINS_AND_PD_ROLLOUT.md)
completed a 35 s trial with native tracking gains. Wrist holding improved,
but the object overshot and the task failed (132.783 mm XY, C1 pass).
Recorded guard activations removed force and velocity commands. The separate
mass-scaled PD prediction-gain comparison also failed (51.683 mm XY,
0.40214 rad, C1 pass). Dynamic geometry replay did not show a substantial
Drake/Isaac contact-frame mismatch in the 175 inspected poses. No
configuration is being promoted to the frozen acceptance set yet.

The [workspace and reference-velocity follow-up](WORKSPACE_AND_REFERENCE_VELOCITY.md)
fixes a confirmed sign reversal caused by differentiating MPC restart jumps.
Both corrected 35 s trials completed without the previous workspace abort,
but remained task failures at about 49 mm XY error. Contact-height and native
QP/force variants are still diagnostic work; gate 1 remains open and the
frozen acceptance set has not been used for tuning.
Subsequent solver diagnostics exposed nonphysical finite-iteration contact
forces. Nonnegative force bounds are now an optional experiment; a longer
10-iteration run stopped at 40.84 s on C1 distance, so no configuration has
passed gate 1. See the linked follow-up for all rejected variants and evidence.

The [2026-09-08 alignment record](SIMULATION_DYNAMICS_ALIGNMENT.md) extends the
shared geometry with measured mass/COM/inertia, actual bearing support points,
effective friction, zero synchronous plan latency, and consistent task-point
pose/Jacobian/twist/wrench mapping. It also fixes brief-contact trace loss and
serializes concurrent staging. Development defaults select this aligned model.
The aligned 180 s attempt stopped at 89.51 s after a workspace-boundary
assertion: 49.599 mm XY and 21.843 deg SO(3), C1 pass and task fail. It is not a
completed 180 s trial. Gate 1 remains open; randomized and clutter acceptance
remain pending. The earlier runs below remain historical evidence.

## Current controlled comparison

All runs below use scene007 from
`data/manifests/contact_planner_m3/hammer_c1_canonical_front180_screen8_seed20260907.jsonl`:
60 mm translation, +10 degree goal yaw, sampler seed 293054. Planner runs at
20 Hz and Cartesian impedance control at 100 Hz. No contact-acquisition servo,
manual push direction, forced-C3 mode, or yaw-specific executor is enabled.

| Run under `outputs/contact_planner_m3/` | Change | Status |
| --- | --- | --- |
| `gate1_closed_gripper_clock0_20260908_v1` | 0 ms lookahead | Infrastructure failure before simulation; missing binary path, excluded from task rate |
| `gate1_closed_gripper_clock0_20260908_v2` | 0 ms lookahead, stock stationary-reference OSC | Completed 180 s: 53.86 mm XY, 20.63 deg SO(3); C1 pass, task fail |
| `gate1_closed_gripper_clock0_velocity_20260908_v3` | Same scene and 0 ms clock, velocity-aware OSC | Completed 180 s: 30.23 mm XY, 89.52 deg SO(3); C1 pass, task fail |
| `gate1_closed_gripper_clock0_velocity_fullpose_20260908_v4` | Velocity-aware OSC, upstream full-pose objective from start | Completed 35 s diagnostic: 46.04 mm XY, 23.53 deg SO(3); C1 pass, task fail |

The v3 `run.sh` and `evaluation_config.json` preserve the exact invocation.
It reads the same staged runtime as v2 without restaging or changing native
planner configuration. The two jobs use distinct GPU devices and TCP ports.
V4 uses an isolated runtime configuration; its sole native configuration
change is `cost_switching_threshold_distance: 1000.0`, matching the existing
full-pose staging mode. Its 35 s duration is a short diagnostic, not a full
180 s robustness trial. All three jobs completed and were cleaned up.

The checked-in [comparison evidence](evidence/contact_planner_m3_bridge_clock_velocity_20260908.json)
records final outcomes. Both 180 s runs returned fresh commands on 3599/3600
cycles, with one startup hold. Velocity tracking brought first contact from
44.35 s to 3.02 s and reduced the C3-phase servo-error p95 from 20.20 mm to
8.01 mm. It improves execution, but does not establish joint pose success.

## Bridge findings and fixes

- Relay source was stale in the canonical checkout. It is now synchronized
  with the repository source; the built relay logs `task_lookahead_ms=0.0`.
- The initial build selected the wrong Bazel output root, encountered a
  Drake/curl source-list mismatch, and repointed `bazel-bin` to that cache.
  Rebuilding against the existing `bazel-push-anything-v151` cache succeeded.
  The build script now infers the existing root from the checkout's
  `bazel-out` symlink, unless an explicit override is provided. Its `--check`
  invocation verified automatic selection of the correct root.
- Drake OSC tracks position and trajectory derivative. Isaac Lab 2.2's stock
  OSC assumes zero target velocity in its damping term. The optional
  `--osc-track-trajectory-velocity` action restores `Kd * (v_des - v)` while
  preserving the existing inertia, wrench, and null-space computation.
  Velocity follows the bounded reference, is capped, and is cleared on hold.
  The measured hand-to-tip frame conversion is differentiated as well.
  At that comparison stage it was off by default; the newer alignment entry
  now enables it, while direct legacy runner defaults remain available.
- Effect auditing previously compared relay timestamps (starting at 100 ms)
  directly with trace step labels (starting at zero). Trace measurements are
  also taken after the physics step. New traces carry `measurement_utime_us`;
  the analyzer converts legacy online-result traces explicitly. The old
  proxy-gap-calibration run's contact-overlap count changes from 21 to 19
  windows: 17 in C3 mode, 2 outside. Its task outcome is unchanged.
- The wrapper's previous 100 ms effect-audit default did not match the
  current native 75 ms knot duration. Future runs derive the duration from
  the staged position/pose timing and selected knot, failing on unequal mode
  durations unless an explicit audit override is provided. The completed
  comparisons were reanalyzed at 75 ms; use `effect_audit_corrected.json`.
- `scripts/analyze_c3_execution_tracking.py` separately reports reference
  limiting, servo error, and sampled contact overlap. Sparse trace fractions
  are not continuous contact duration. Missing historical fields remain null.

## Validation so far

- Native canonical online build: passed using the original v151 cache.
- Bridge protocol, frame mapping, and effect audit: 28 tests passed.
- Tracking diagnostic: 2 tests passed.
- Actual Isaac OSC integration, with AppLauncher active: 4 tests passed
  (perfect moving reference, stock-controller equivalence at zero reference
  velocity, clearing motion into a hold, and rejecting invalid references).
  The explicit status file records zero; an earlier one-line launcher lacked
  test execution markers and is not counted as validation.
- Three physical comparisons completed; none passed strict joint pose.
- Randomized success and clutter gates: pending.

## Contact geometry: initial proxy audit

`scripts/audit_closed_gripper_proxy_contacts.py` replays the observed poses
against the exported full target mesh. It removes the baked support rotation,
uses the common table/robot frame, welds duplicate mesh vertices for the
inside/outside test, and records mesh/result hashes.

| Run | Sampled legal physical contacts | Contacts with proxy clearance > 2 mm | Median / maximum proxy clearance |
| --- | ---: | ---: | ---: |
| v2 clock-only | 8 | 6 | 4.93 / 11.92 mm |
| v3 velocity | 7 | 7 | 5.25 / 9.60 mm |
| v4 full-pose short | 4 | 4 | 10.27 / 12.39 mm |

These are sparse sampled contacts, not complete contact counts or durations.
They establish a mismatch between the observed force events and the spherical
proxy at recorded poses. They do not alone separate finger shape, PhysX
collision approximation/margins, and contact reporting timing. Simply adding
two side spheres in an offline probe still left several millimetres of error;
no alternative proxy was installed in the planner.

## Live collision verification (2026-09-08)

The follow-up audit now extracts the composed USD colliders, actual PhysX
cooked convex vertices, and runtime contact/rest offsets from the same task
and manifest. See [the detailed audit](CLOSED_GRIPPER_GEOMETRY_AUDIT.md) and
[machine-readable evidence](evidence/contact_planner_m3_geometry_audit_20260908.json).

- The project overrides the remote stock USD with Isaac Sim's packaged
  Franka URDF converted locally. Both fingers use one convex hull each.
- Finger contact offset is 0.4205 mm and target offset is 0.5902 mm; rest
  offsets are zero. These margins cannot account for a 9.92 mm proxy gap.
- At all 19 reconstructed contact samples, fully closed finger convexes
  are within 0.034 mm of the actual target convexes. The 8 mm sphere against
  those same target convexes has median/max clearance 5.717/9.921 mm.
  Reconstructed proxy-coordinate error is at most 0.000074 mm.
- A second discrepancy is confirmed: Isaac cooks 16 convexes from the
  DOMINO **visual** mesh, while C3's full mesh comes from its **collision**
  mesh. Source-to-planner surface distances have median/p95/max
  1.372/3.159/11.995 mm across the USD source vertices; these are whole-object
  sampled surface distances, not distances specifically at contact points.
- Historical finger displacement was not recorded. The reconstruction
  explicitly assumes zero displacement; it does not recreate contact forces
  or establish contact-normal agreement. Sensors have zero history length,
  update every physics step, and the old runs use audit stride 1. A contact
  force summarizes the last 2.5 ms physics interval, while the pose is its
  endpoint, so exact within-step contact timing is not recoverable.

The executor now logs finger positions, hand/finger poses, robot base pose,
environment origin, and a separate contact-audit timestamp. A 1 s closed-loop
smoke run recorded all 10 trace samples with matching measurement/audit
timestamps. Finger displacement during this free-space smoke was
[-0.00205, 0.02605] mm. This checks instrumentation, not historical contact
deflection or task success. Twelve geometry/frame/effect/tracking tests passed.

That follow-up is implemented in the [shared PhysX contact model](SHARED_PHYSX_CONTACT_MODEL.md).
The C1 Isaac evaluator now defaults to the same 16 target convexes and two
finger convexes as PhysX, with hand orientation updated at every replan.
The sampler accounts for the oriented finger extent and excludes its own
finger geometry from object queries. Historical sphere runs require explicit
`--legacy-sphere-contact-model`; their results remain unchanged.

Nineteen one-step PhysX manifold probes versus native Drake queries pass the
geometry checks: maximum separation disagreement 0.01970 mm, maximum closest
manifold-normal disagreement 4.2778 degrees, maximum hand-orientation error
1.151e-6 rad. This is a zero-velocity geometry probe, not a historical force
reproduction or task acceptance. Thirty-eight related tests and the new
default evaluator's end-to-end 1 s smoke passed.

A corrected 35 s scene007 regression using stock OSC reached legal contact
at 18.87 s, ending at 59.944 mm XY and 10.162 degrees SO(3), with C1 pass and
task fail. Shared collision geometry is fixed; goal-directed pushing and the
strict pose gate remain unresolved. The target's original C3 three-sphere
table support approximation and dynamics parameters are retained, and hand
orientation is held within each prediction horizon. These remaining model
approximations must be distinguished from the corrected contact surfaces.

The full default-entry scene007 comparison with velocity-reference OSC then
completed 180 s: first legal contact 3.01 s, final XY 38.000 mm, height
0.988 mm, SO(3) 43.968 degrees, C1 pass, task fail, and zero strict pose dwell
steps. There were no infrastructure failures; all 1800 trace timestamps
matched their contact audits. The corresponding 35 s velocity comparison
ended at 38.121 mm and 28.989 degrees. These single-scene trials show pushing
without joint-pose convergence, and do not close Gate 1 or randomized gates.
See the [shared-model evidence](evidence/contact_planner_m3_shared_physx_contact_20260908.json).

A remaining dynamics check found a 14.039 mm body-frame COM disagreement
despite matching 0.05 kg target masses. This is recorded for the next dynamics
alignment step; no causal claim or target-dynamics change is made here.


Follow-up: [same-reference dynamics diagnostics](SAME_REFERENCE_DYNAMICS_DIAGNOSTICS.md)
separate the replanning confound and measure actual OSC inertia. Four diagnostic
runs completed with C1 pass; none are task-acceptance trials. Goal remains active.

The subsequent [PD/contact calibration work](PD_INERTIA_AND_CONTACT_CALIBRATION.md)
produced one online strict debug success at 172.66 s, XY 16.859 mm, height
0.988 mm and full SO(3) 0.096252 rad, with zero C1 violations. The original
trace cannot independently reconstruct all 50 terminal dwell steps, so an
archived-binary repeat now retains every strict-pose measurement. Its first
424 measured states (through planner time 21.25 s) exactly match the original.
Three other preexisting debug scenes are being evaluated with this same
configuration. Formal acceptance remains unstarted and the goal remains active.
