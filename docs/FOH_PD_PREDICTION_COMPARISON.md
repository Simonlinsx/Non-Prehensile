# FOH reference prediction comparison

The simulation goal remains active. There are no strict debug successes and
no runs on the frozen 50-scene acceptance manifest.

## Observed reference mismatch

The original `c3/core/traj_eval.cc` upsamples the planned state and force
using zero-order hold for the fine PD/LCS cost rollout. Its desired velocity
comes from the stored state. The native LCM position receiver instead uses
first-order hold of the position rows and ignores the stored velocity rows.
The Isaac relay likewise interpolates position and force and derives velocity
from successive position knots. This source-level difference is verified;
it is not yet established as the cause of task failures.

Patch `0013-optional-foh-pd-rollout.patch` adds an optional controller-side
prediction wrapper. It calls the existing fine `LCS::Simulate` transition
with the original regularization settings. The active C3+ solver, constraint
system, and core trajectory evaluator are unchanged. Existing patch 0002
remains the compilation compatibility change in the inactive MIQP stub.

`--c3-pd-rollout-interpolation zoh|foh` defaults to `zoh`. The evaluator records
it and checks the actual isolated runtime parameter before launch. The runner
records the runtime value in its result. FOH changes the reference used for
sample-cost prediction, and therefore can change sample ranking/mode behavior;
it does not change the executed reference interpolator.

The FOH rollout:

- Preserves the measured initial state, including its actual velocity.
- Interpolates published EE positions and forces at the fine LCS timestep.
- Uses the position derivative as desired EE velocity, including at time zero.
- Holds the last published knot with zero reference velocity during the last
  coarse interval, matching the Isaac relay's end behavior. The synthetic
  terminal state added for cost evaluation is not a published command knot.
- Samples every fourth fine state/control and preserves the final simulated
  state, matching the original downsampling convention.

This aligns interpolation only. It does not establish matching contact
physics, impedance dynamics, feedforward conventions, or geometric safety
shield behavior in the prediction model.

## Verification and controlled trials

The actual helper header is extracted from the patch and compiled in a test
against Eigen. Checks cover nonzero initial velocity, right derivative, force
interpolation, feedforward disabled, a single published knot, terminal hold,
stateful fine transitions, coarse sampling, and invalid inputs. Together with
shared staging and fixed-goal tests: 10 passed. Motion-stratified prediction
analysis tests: 3 passed. Python compilation passed.

Native build succeeded; binary SHA256:
`350258ee67c16463d64ccb7da4dd389c0de8e184478c0b68b42464b7f9dce3cf`.
The existing long trial's process retained its original binary hash after
linking. Build and process evidence are stored under
`outputs/contact_planner_m3/workspace_height_20260908/foh_native_build*`.

The controlled pair uses scene007, fixed native goal mode 2, seed 293054,
35 simulated seconds, translation/rotation stiffness 200/800, damping ratio
sqrt(0.5), nominal PD rollout gains, quaternion cost 100, original progress
window, nonnegative contact force constraint, geometry floor, and high-rate
C1 guard. Configuration comparison showed only GPU and interpolation differ:

- `foh_native_gains_fixed_goal35`: FOH, GPU 3, ports 9200/9201.
- `zoh_native_gains_fixed_goal35`: ZOH, GPU 5, ports 9220/9221.

Both retain source snapshots, native binaries, actual runtime files, traces,
predictions, and video. Both completed; neither is an acceptance run.

## Motion-stratified error

`analyze_c3_solver_consistency.py` now preserves per-plan pairs and separates
actual horizon displacement >=5 mm from smaller motion. The 5 mm threshold is
a diagnostic grouping, not a success threshold. Actual future motion includes
subsequent replans and therefore is not an open-loop model calibration.

For the earlier nominal-PD/native-gain trial, 49 of 155 paired horizons moved
>=5 mm. Their median actual motion was 55.391 mm, PD predicted motion 3.675 mm,
and PD endpoint error 51.998 mm. For mass-scaled PD, only 12 of 124 horizons
moved >=5 mm; their median PD endpoint error was 13.225 mm. Its extremely small
overall median was dominated by near-stationary horizons. Neither result
establishes successful manipulation.

## Completed comparison

| Prediction | XY error | SO(3) error | C1 | Strict dwell | Task |
| --- | ---: | ---: | --- | ---: | --- |
| ZOH | 132.783 mm | 0.016717 rad | pass | 0 | fail |
| FOH | 58.074 mm | 0.444940 rad | pass | 0 | fail |

Both videos were independently checked: 35 seconds, 700 frames, 20 fps.
FOH had 193 paired plans, but only 10 horizons with >=5 mm actual motion;
ZOH had 151 pairs with 49 moving horizons. Moving-horizon median PD endpoint
errors were 20.422 mm / 0.194815 rad for FOH and 51.998 mm / 0.088820 rad for
ZOH. These are different state/action distributions; lower XY error does not
establish better dynamics prediction. Initial quaternion agreement is within
about 6e-6 rad; large future errors are not a fixed quaternion-frame offset.

The optional FOH setting is not promoted. A useful next diagnostic is to
execute one captured native planner trajectory over its own short horizon,
with C1 and workspace protection retained, and compare against that exact
plan's PD rollout. Replanning otherwise confounds future endpoint comparisons.
Such a calibration must be explicitly excluded from task acceptance, must
use planner-generated references, and must record any downstream guard,
height/speed/force limiting that changes the planned inputs. It cannot count
as a successful manipulation trial or replace receding-horizon acceptance.

## Acceptance record schema correction

Reviewing the actual output found two reader mismatches: the runner writes
`physical_end_effector: stock_franka_gripper_closed`, and native goal/PD
settings are under `controller_parameters`. The auditor previously expected
a different gripper string and top-level native fields. That would classify
all actual runs as invalid. The reader now follows the actual schema.

A fixture extracted from this real failed FOH result checks that it is a
valid failed record, never a success. Synthetic success-count tests also
follow the actual schema; malformed or wrong-mode nested settings are
rejected, and top-level lookalikes cannot override them. Both completed
runtime goal artifacts were checked against their actual files and hashes.
The latest acceptance/fixed-goal checks passed 8 tests. No result was edited.

See [machine-readable evidence](evidence/contact_planner_m3_foh_prediction_20260908.json).


The concurrent rotation-stiffness-100 fixed-goal long trial also completed
180 s: 16.102 mm XY / 0.577069 rad SO(3), C1 pass, zero strict dwell. The
180 s video was verified at 3600 frames / 20 fps. All three evaluations are
finished; no strict diagnostic success and no acceptance batch yet.


Follow-up: [same-reference dynamics diagnostics](SAME_REFERENCE_DYNAMICS_DIAGNOSTICS.md)
separate the replanning confound and measure actual OSC inertia. Four diagnostic
runs completed with C1 pass; none are task-acceptance trials. Goal remains active.
