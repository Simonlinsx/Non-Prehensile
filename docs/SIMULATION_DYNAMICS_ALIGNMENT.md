# Dynamics and execution alignment (2026-09-08)

Gate 1 remains open. This follow-up starts from the shared-collision model's
180 s scene007 failure (38.000 mm XY, 43.968 degrees SO(3), C1 pass).
Acceptance remains 20 mm XY, 10 mm height, 0.105 rad SO(3), 0.5 s dwell,
physical legal contact, and no C1 violation. No manual pushing or forced-C3
contact mode is enabled.

## Changes under validation

- `closed_franka_hammer_physx_v2.json` retains the same 18 collision convexes
  and adds measured mass, COM, and central inertia. Staging rotates the COM
  and inertia into the support-baked C3 body frame. Both target SDF inertial
  poses now carry the measured COM. The executor rejects a changed physical
  mass/COM/inertia contract. Existing v1 data is retained for reproduction.
- The inertia exported by PhysX is already central inertia expressed in body
  axes. Its principal-axis quaternion must not be applied a second time;
  see the [PhysX tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidBodyView.get_inertias).
- Patch `0008-synchronous-simulation-plan-clock.patch` adds the optional
  `use_simulation_time_for_plans` setting. It leaves latency compensation at
  zero for synchronous simulation, which does not advance its measurement
  clock during a solve. The evaluator selects this mode by default. Native
  configurations without the flag preserve the upstream wall-clock behavior.
  Sampled old relay logs showed median trajectory-start leads of 86.33 and
  92.62 ms in the first two short trials, versus a 50 ms planning period.
- The relay logs every returned task reference's measurement, plan start,
  and query time. `analyze_c3_plan_clock.py` checks the synchronous zero-lookahead
  contract separately from task success.
- `--osc-control-point tip` uses one offset point for pose, Jacobian, twist,
  inertia, and force mapping. It rotates the hand-local offset into root axes
  before shifting the Jacobian, preserving the wrist moment of a distal force.
  Twist is J*qdot for the fixed-base arm, avoiding the old link-pose/COM-velocity
  mixture. The earlier analytic hand-position mapping remains available with
  `--osc-control-point hand`.

The new task-point action passed a live finite-difference check at three
joint configurations: maximum translation-Jacobian error 0.000105 m/rad,
angular-Jacobian error 0.000323, twist error 0.000027, and virtual-work torque
error 1.54e-8 Nm. This is kinematic validation, not task acceptance.

## Completed short regressions

All directories are under `outputs/contact_planner_m3/shared_dynamics_20260908/`.
Each trial lasts 35 simulation seconds and is an individual diagnostic.

| Directory | Difference | Final XY | Final SO(3) | Outcome |
| --- | --- | ---: | ---: | --- |
| `matched_short` | Measured inertia/COM, old hand mapping and wall clock, Kp 150 | 40.079 mm | 26.521 deg | C1 pass, task fail |
| `stiff600_short` | Same, Kp 600 | 42.059 mm | 23.487 deg | C1 pass, task fail |
| `simclock_short` | Simulation clock, hand mapping, Kp 150 | 49.960 mm | 21.443 deg | C1 pass, task fail |
| `simclock_fullpose_short` | Same, upstream full-pose objective from start | 52.580 mm | 14.730 deg | C1 pass, task fail |
| `tip_xy_short` | Simulation clock, unified task point, original objective | 27.792 mm | 59.521 deg | C1 pass, task fail |

| `tip_fullpose_short` | Simulation clock, unified task point, full-pose objective | 59.789 mm | 10.130 deg | C1 pass, task fail |

Gains alone did not resolve the task. Clock correctness
and kinematic consistency must not be described as successful pushing.

## Table support and friction alignment

The original three support spheres have radius 1 mm and center z=-12.9369 mm
in the support-baked body frame. Their bottom is therefore -13.9369 mm,
whereas the shared PhysX convexes' minimum z is -12.1179 mm: a 1.819 mm
geometry mismatch. At the settled Isaac pose, C3 repeatedly predicts a rise
of roughly 2 mm. The support spheres also use bounding-box corner locations.
The final diagnostic filters the actual static shape `/World/ground/geometry/mesh`
through the PhysX tensor contact API. Filtering its parent Xform returns no
contacts; enabling legacy contact events was unnecessary and has been removed.

The settled scene has 28 contact candidates but only three with nonzero normal
force (>1e-5 N). Their force sum is 0.490587 N, consistent with the 50 g target.
The support triangle contains the projected COM with barycentric coordinates
[0.253875, 0.456396, 0.289729]. Version 3 retains the original three-sphere
approximation/count and places each 1 mm sphere one radius above one of these
bearing contacts, transformed into the support-baked frame. This selection has
no dependency on the task goal. Non-bearing speculative contacts are excluded.
The initial C3 prediction's spurious rise changes from 1.988499 mm to
0.000064 mm. This validates resting support, not dynamic sliding or rotation.

Version 4 additionally aligns the effective pair friction: finger-target 0.4
and target-ground 0.3, replacing C3's 0.42 and 0.46. The actual target tensor
materials are 0.3, the robot 0.5, and the ground 1.0. The default average mode
therefore gives 0.4 for the finger pair; the ground's multiply mode takes
priority and gives 0.3. See the [PhysX combination rule](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html).
Both C3 option files receive the coefficients; unrelated pair types remain
unchanged. Physical materials are not modified.

`support_xy_short` stopped at 12.46 s: measured x=0.229165 crossed the original
0.23 m workspace boundary, triggering the native assertion and then the bridge
watchdog. It is an execution failure, not a completed 35 s trial. The original
workspace limit is retained. Final XY was 47.953 mm and SO(3) 22.672 deg; C1
remained satisfied. Other support/friction trials are recorded in the evidence
once complete.

## Reproducibility and checks

The evaluator serializes staging and snapshot creation with a checkout-local
file lock, so concurrent objective variants cannot copy a mixed configuration.
Four completed simulation-clock trials each have 699 reference observations;
all 2,796 plan starts and query times agree exactly with the measurement clock.
Thirty-nine focused tests passed, with the AppLauncher-dependent OSC test module
skipped in the ordinary pytest invocation. The separate live three-configuration
kinematic check supplies integration evidence for the new task-point action.

The revised contact trace stores every positive physical contact at the 100 Hz
control rate in addition to the normal 10 Hz trace. An 8 s replay preserved 26
contact rows, all in C3 mode, with the first row at step 404 exactly matching the
contact detector. The effect audit now sees 9 overlapping contact windows
(8 C3 and 1 straddling the mode transition). The older 35 s and full-180 runs
used uniform trace decimation, so their zero sampled-contact windows cannot
establish absence of contact. Their all-step task/safety flags are unchanged.
The tracking analyzer marks the new trace as nonuniform and does not infer
contact duration from row fractions.

Development defaults now select v4, simulation-time plans, task-point OSC and
trajectory-velocity tracking. These are model/execution defaults, not a claim
of task acceptance. To reproduce the historical sphere/hand baseline, specify
`--legacy-sphere-contact-model --planner-clock-mode wall --osc-control-point hand
--no-osc-track-trajectory-velocity` (plus the historical run's other settings).

## Final outcome of this follow-up

| Trial | Requested / executed time | Final XY | Final SO(3) | Outcome |
| --- | --- | ---: | ---: | --- |
| v3 support, full pose | 35 / 35 s | 59.721 mm | 10.116 deg | C1 pass, task fail |
| v4 support + friction, original schedule | 35 / 35 s | 49.599 mm | 21.843 deg | C1 pass, task fail |
| v4 aligned long attempt | 180 / 89.51 s | 49.599 mm | 21.843 deg | C1 pass, native workspace abort and watchdog |

The long attempt crossed x=0.23 m at measured x=0.229643 m. Its original
boundary was not relaxed. Five stale task references after each of the two
native workspace failures cause those runs' whole-stream clock audits to fail;
preceding references are aligned. These failures remain in the evidence.
The long run is **not** a completed 180 s trial and does not pass Gate 1.

The short contact-trace check and default-entry smoke both completed with the
expected time-limit task failure; their clock and measured-dynamics checks
passed. The twelve runs in the [machine-readable evidence](evidence/contact_planner_m3_dynamics_alignment_20260908.json)
include eight completed 35 s diagnostics, two early workspace failures, one
8 s contact trace check, and one 1 s startup smoke. All twelve kept C1; none
passed joint pose. They are not a randomized success-rate benchmark.

Next work should isolate why useful pushes cease after the first crossing
into the original 5 cm pose-cost region, and why reposition execution can
cross the original workspace boundary despite bounded reference tracking.
Use the contact-preserving trace for this investigation. The measured geometry,
inertia, support and friction are now explicit reusable contracts; do not
replace this remaining task failure with a calibration-success claim.

Reproduce the aligned fixed-scene attempt with the current defaults (use a
new output directory to preserve evidence):

```bash
/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python scripts/evaluate_push_anything_isaaclab_c1.py \
  --manifest data/manifests/contact_planner_m3/hammer_c1_canonical_front180_screen8_seed20260907.jsonl \
  --scene-id scene007 --output-root outputs/contact_planner_m3/aligned_scene007_repeat \
  --gpu 7 --base-port 8840 --max-sim-time-s 180
```

Current reruns include the contact-preserving trace; it changes logging only.
The development default remains the original Push Anything objective schedule.
