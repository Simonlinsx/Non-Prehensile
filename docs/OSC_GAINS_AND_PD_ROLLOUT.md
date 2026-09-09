# OSC gains and contact prediction follow-up

The fixed diagnostic scene and all strict acceptance thresholds remain
unchanged. Gate 1 is open; the frozen 50-scene evaluation has not started.

Native `examples/sampling_c3/shared_parameters/osc_params.yaml` defines
translation Kp=200, Kd=20 and rotation Kp=800, Kd=40. The Isaac integration
was using translation stiffness 150, rotation stiffness 25 and damping ratio
1. With Isaac's gain convention, damping ratio sqrt(0.5) gives Kd=20 for
Kp=200 and Kd=40 for Kp=800. `--osc-damping-ratio` is now exposed by the
evaluator, passed to the existing runner option, validated and recorded.
These match the native tracking gains, not the entire native inverse-dynamics
QP or its null-space objective. Wrist orientation remains held for the closed
finger geometry, although the upstream spherical-actor demo disables it.

## Completed 35-second comparison

`native_tracking_gains_guard_q100_35` uses those tracking gains, quaternion
cost weight 100, the repaired quaternion constructors and native-equivalent
high-rate semantic guard. It completed 35 s with C1 pass and task failure:
132.783 mm XY and 0.016717 rad SO(3). Maximum applied joint effort was
15.965 Nm. The 35 s / 700-frame / 20 fps video is preserved.

Across 536 trace samples, maximum hand orientation deviation from its first
recorded pose was 0.027233 rad (1.56 degrees); median was 0.009065 rad.
The earlier unshielded weight-100 run reached 0.520259 rad (29.81 degrees).
These full-orientation deviations include yaw and should not be confused
with the earlier downward-tilt metric. A same-duration comparison against
the low-gain shielded long run remains pending.

Minimum sampled finger surface clearance was +0.005729 mm. This is almost
table contact, not a verified 2 mm clearance guarantee. The semantic shield
activated on 128 servo steps. All 13 recorded active samples had zero
feedforward force and zero reference velocity. The independently measured
C1 oracle did not report a violation. These trace samples do not prove the
force/velocity values on unrecorded steps.

## Why prediction is the next test

During a sustained push around 8--9 s, the object passed the goal. At 8 s,
the PD rollout predicted an endpoint only about 0.35 mm from its initial
object position after 0.675 s, while measured motion continued by several
centimetres. Across 155 matched executed C3 plans, PD endpoint XY error had
median 15.344 mm, p90 72.135 mm and maximum 90.158 mm. The raw final-QP
prediction's median error was 91.381 mm. Subsequent replans are included in
actual future motion, so this is not an open-loop calibration.

The native cost rollout uses a 0.057 kg actor and force-domain PD gains
`Kp=[100,100,50]`, `Kd=[0.5,0.5,0.5]`. The controller integration's desired
acceleration gains are a different convention. An isolated hypothesis is to
use actor-mass-scaled gains: Kp=0.057*200=11.4 N/m and
Kd=0.057*20=1.14 N s/m on all three translation axes.

`--c3-pd-rollout-kp` and `--c3-pd-rollout-kd` now override only the existing
native rollout parameters in each isolated runtime. They leave C3 core
sources and the planner's Q/R costs unchanged. The materialization tests
check both option files and that the source runtime remains unchanged.
Defaults leave the original rollout gains intact. These options and the
OSC damping ratio are included in evaluation provenance and snapshots.

The `mass_scaled_pd_native_gains_q100_35` experiment completed with the
same actual controller and C1 guard as the comparison, changing only these
PD rollout gains. It failed at 51.683 mm XY / 0.40214 rad, with C1 pass and
zero guard activations. The hand remained well held (maximum 0.02896 rad
deviation), but the object mostly stopped moving after initial contact.
PD median endpoint error was 0.000706 mm across 124 matched plans; this tiny
median is dominated by stationary behavior and does not validate useful
contact prediction or task performance. The gain hypothesis is not promoted.
This does not by itself match feedforward,
operational inertia, contact dynamics, or continuous interpolation: the
native cost evaluator uses a zero-order-held fine rollout. The hypothesis
must be assessed against measured predictions, task success and C1 safety.

## Dynamic geometry cross-check

The native geometry-audit mode replayed 175 measured arm/object/tool poses
from 7.5--9.5 s of the native tracking-gain trial. It used the trial's isolated
runtime and existing binary with `--lcm_url=memq://` (the default multicast
self-test failed on the first attempt; its failure log is preserved).
Maximum Drake/Isaac hand rotation disagreement was 1.42e-6 rad. In the
172 samples with measured hand force above 0.01 N, the Drake finger/target
surface distance ranged from -0.10759 mm to +0.005852 mm (median -0.01524 mm).
Thus these actual contact states do not show a large missing-contact geometry
gap. This does not validate future contact dynamics or all sampled poses.

The original progress monitor is another concrete next diagnostic: native
`kConfigCostDrop` compares configuration costs over 35 planner updates and
requires a 50% reduction. At 20 Hz this can retain a contact for roughly
1.7 s, consistent with the observed overshoot interval. A shorter window
using the existing native progress parameters has not yet been tested;
no progress-window or cost-drop override has been implemented in this turn.

Detailed results and hashes are in
[the evidence record](evidence/contact_planner_m3_osc_pd_gains_20260908.json).
