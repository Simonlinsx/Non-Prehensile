# Progress window and fixed native goal

Gate 1 remains open. No debug trial has passed strict joint pose plus dwell,
and the frozen 50-scene acceptance set remains unused.

## Completed runs

| Trial | Sim time | XY error | SO(3) error | C1 | Task |
| --- | ---: | ---: | ---: | --- | --- |
| native_osc_guard_q100_video180 | 180 s | 20.3146 mm | 0.045665 rad | pass | fail |
| progress5_native_gains_q100_35 | 35 s | 67.7326 mm | 0.471215 rad | pass | fail |

The first trial's height error was 0.98684 mm and its maximum strict dwell
was zero steps. The position criterion is **strictly below 20 mm**; a small
miss is still failure. It completed a valid 180 s / 3600-frame video. The
high-rate semantic guard activated on 1599 servo steps; all 225 recorded
active samples had zero commanded feedforward force and reference velocity.
This supports the observed run's C1 outcome, not universal safety or task
success. The maximum recorded hand orientation error was 0.71163 rad, so
wrist holding remains an execution concern with rotation stiffness 25.

The short-window comparison retains the previous native tracking gains
(translation 200, rotation 800, damping ratio sqrt(0.5)) and nominal PD
rollout gains. It changes the native `kConfigCostDrop` window from 35 to 5
planner updates and the drop fraction from 0.5 to
`1 - 0.5**(4/34) = 0.07831035905913464`. At 20 Hz, the first-to-last history
span changes from 1.7 s to 0.2 s while retaining the same exponential
improvement rate. The controller started repositioning earlier, but strict
task performance did not improve sufficiently. The configuration is not
promoted.

The evaluator and isolated staging script now expose
`--c3-progress-window-loops` and `--c3-progress-cost-drop`. Defaults leave
native parameters unchanged. Overrides are restricted to the copied demo
runtime and native `kConfigCostDrop` mode. Materialization tests verify the
written values and unchanged source checkout.

## Fixed-goal configuration defect

The staged native `goal_params.yaml` supplied the requested fixed target
position and orientation but retained `goal_mode: 0` from the random-goal
template. The native enum defines random=0 and fixed=2. On an instantaneous
pose hit, random mode generates another target before the Isaac 0.5 s dwell
is guaranteed. This is a latent acceptance blocker.

The inspected completed trials had zero strict dwell and only one initial
`Detected goal change!` message. There is no evidence that random goal
switching caused their failures. Their configured target coordinates match
the Isaac goal (apart from the documented 29 mm frame-height offset and
support-quaternion transform).

`configure_fixed_task_goal` now explicitly writes mode 2 together with the
target pose, retaining the original thresholds. The evaluator checks the
staged mode before launching. The runner records the actual native goal mode
and the resolved goal-file path/hash. The acceptance auditor requires fixed
mode in both config and result, and verifies that the actual hashed goal
file lies within the scene's runtime and still declares mode 2. Historical
runs without this evidence cannot become formal acceptance results.

Regression tests cover random-template conversion, idempotent staging,
unchanged pose thresholds, changed goal files, wrong mode despite updated
hashes, and artifacts outside the scene directory. The latest fixed-goal,
acceptance and shared-staging test run passed 14 tests. Runner/evaluator
syntax and CLI checks also passed.

`fixed_goal_rot100_q100_video180` has started with verified native mode 2,
rotation stiffness 100, and the remaining settings from the low-gain guarded
trial (translation 150, damping ratio 1, quaternion cost 100, original
progress window and PD rollout gains). Runtime outcome remains pending.

## Solver provenance clarification

The C3 checkout is still pinned to
`5c08cb2e14b1ab10e024cb46e8504970cffcd5ea`. Its only tracked source diff is the
existing patch 0002: `options.M.value_or(1000)` in the no-Gurobi MIQP stub's
constructor. That constructor throws when invoked; the active configuration
uses the C3+ projection branch. Thus the active C3+ algorithm is unchanged,
while claiming that every core source byte is pristine would be inaccurate.
The compatibility patch is already archived with execution snapshots.

See [the evidence record](evidence/contact_planner_m3_progress_fixed_goal_20260908.json).


The fixed-goal rotation-stiffness-100 trial has now completed all 180 s:
16.102 mm XY, 0.577069 rad SO(3), 0.98553 mm height error, zero strict dwell,
and C1 pass. It is a task failure. The video has 3600 frames at 20 fps.
Maximum sampled hand orientation deviation was 0.177544 rad; minimum sampled
finger clearance was -0.04038 mm. The runtime goal artifact verified as mode 2.
All results retain their original bytes. The subsequent
[FOH comparison](FOH_PD_PREDICTION_COMPARISON.md) also failed task acceptance.
