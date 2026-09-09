# PD inertia and contact-model calibration

Current follow-up: the historical matched-coarse failure below used binary
`1b68b374...`, before the finite-cost hysteresis correction in patch 0020.
An independent audit matches all 3,599 recorded cost/command timestamps and
finds 111 invalid-current/finite-alternative boundaries, including four with
C3 still selected; 21 invalid-previous/finite-alternative boundaries still
selected repositioning. These observations do not prove the cause of failure.
They motivate the new `coarse_matched_finite_hysteresis_scene007_180` trial,
which combines matched coarse dynamics with frozen patch 0020, original
resolution 4 and partial OSC. The trial is running with the same strict task
criteria. See `evidence/contact_planner_m3_matched_coarse_revisit_20260909.json`.

The new run now reproduces the first 62 measured object states exactly through
3.15 s. At that timestamp both versions have identical candidate costs:
current infinity, previous 358.3129, best finite alternative 268.6131. The old
version keeps C3 mode; patch 0020 selects repositioning. The first physical
state difference is at 3.20 s. A frozen prefix artifact preserves this causal
mode-boundary comparison while the full trial continues; it is not a final
task-success result.

Scalar apparent-inertia correction alone did not explain the contact prediction
error. The subsequent 1 kg, contact-relinearized PD cost produced the first
online strict debug success at **172.66 s**, with **16.859 mm XY / 0.988 mm
height / 0.096252 rad full SO(3)** and zero C1 violations. The online counter
records 50 consecutive 100 Hz steps. Its older noncontact trace was decimated,
so those 50 terminal measurements cannot all be reconstructed independently.
The archived-binary repeat also succeeded at 172.66 s with identical final
errors. It retains every strict-pose measurement, and independent recomputation
certifies all 50 terminal steps. All 3454 relay-measured states and all 1890
previously retained trace rows match exactly; 41 additional trace rows complete
the evidence. There is one independently certified, unique tuned debug scene,
not two independent successes. The frozen 50-scene batch has not started.
The passive replay diagnostics below remain ineligible for task acceptance.

In `reference_replay_calibrated_pd_t8p3_diag10`, the original native reference
was captured at **8.350 s** and replayed through **8.650 s**. This is the
actual capture time, not the requested earliest capture of 8.300 s. The
197 recorded execution samples are exactly equal to the preceding
`reference_replay_inertia_t8p3_diag10` baseline, including OSC dynamics caches.
The observer did not change this physical trajectory.

The target moved **26.033 mm** in XY during the 0.3 s window. There were zero
C1 shield interventions, zero C1 violations, and 14 sampled positive legal
hand-contact forces. Both new observers use the same raw plan and initial
state, FOH reference interpolation, acceleration-domain gains 200/20,
point-mass gravity compensation, and the actual negative feedforward-force
sign. They differ in scalar actor mass. Full directional robot inertia is
not reproduced by either scalar model.

| Prediction | Predicted XY motion | Target XY endpoint error | Target SO(3) endpoint error |
| --- | ---: | ---: | ---: |
| Original ZOH | 1.891 mm | 24.421 mm | 0.043149 rad |
| Original FOH | 2.175 mm | 24.164 mm | 0.043149 rad |
| Matched PD, 0.057 kg | 0.792 mm | 25.336 mm | 0.040126 rad |
| Matched PD, 1 kg | 2.844 mm | 23.585 mm | 0.041328 rad |

These are prediction errors against the same measured endpoint, not final
task errors. At the end of the full 10 s run, task XY was 17.385 mm and full
SO(3) was 0.168737 rad: task failure, zero strict dwell. C1 passed. The video
has 200 frames at 20 fps; the actual isolated fixed-goal artifact hash was
verified. See the [machine-readable evidence](evidence/contact_planner_m3_pd_calibration_20260908.json).

Patch 0015 adds a default-disabled, read-only diagnostic within planner
clock 8.2–8.8 s. Both double and AutoDiff actor masses are restored before
ordinary planning continues, including on exceptions. Each evaluation
archives the exact native executable and patch bytes. The first revision
produced this scalar-only comparison; the next revision additionally offers
per-fine-step contact relinearization and quaternion normalization, and
restores plant state as well as mass. That comparison is running. It will
use the existing Anitescu LCP simulation and will remain outside active
C3 planning costs. Updating local contact geometry tests a separate model
approximation; it does not establish task success.

The first relinearized-model run stopped at **8.41 s** with watchdog timeout:
a 0.057 kg counterfactual produced an undefined geometry normal during a
future predicted contact. This is an incomplete diagnostic run. The next
revision records each failed counterfactual separately and restores model
state/mass before continuing the original controller.

`reference_replay_relinearized_isolated_t8p3_diag10` completed 10 s. All 192
execution records match the earlier dual-prediction baseline after excluding
only the OSC cache field, which the baseline did not enable. It captured at
**8.300 s**, replayed through **8.600 s**, and observed 18.440 mm of target XY
motion with no C1 shielding in the comparison window. The 1 kg model with
per-fine-step relinearization predicted **13.773 mm**; target endpoint errors
were **6.708 mm / 0.045254 rad**, versus original ZOH **16.281 mm / 0.063639 rad**.

A second comparison uses the previously executed **8.350–8.650 s** reference.
The two runs' solver x/u/lambda/z/delta messages agree exactly; full-precision
native raw/ZOH/FOH states and both frozen calibrated trajectories also agree
exactly at that plan. The relinearized 1 kg prediction reduces endpoint errors
from **24.421 mm / 0.043149 rad** to **9.531 mm / 0.003536 rad**. These neighboring
windows overlap and do not constitute independent held-out task validation.
The evidence file includes the artifact hashes and cross-run equality checks.

Across all 13 observed planner times, all 13 relinearized 1 kg trajectories
were valid. Five 0.057 kg trajectories failed at fine steps 23–37; these are
explicit invalid predictions, without endpoint-error claims. The complete
isolated run ended with task XY 8.889 mm and SO(3) 0.163402 rad, zero strict
dwell, and C1 pass. It remains a task failure.

Patch 0016 now offers `--c3-relinearized-pd-cost`, default disabled. It evaluates
the same raw C3 solution using the tested 1 kg, 200/20 acceleration-gain FOH
model, gravity compensation, negative feedforward-force sign, and refreshed
contact geometry. It retains the existing object-only quadratic cost and
original C3+ solve. Shared model updates are serialized within the parallel
candidate evaluation, with state and mass restoration. Invalid predictions
receive infinite candidate cost and are logged. Activation requires matching
FOH, tip OSC, trajectory velocity, force sign and 200/800 OSC gains. The
`relinearized_pd_cost_native_gains12` closed-loop regression is running;
model-prediction improvement has not yet demonstrated task success.

The 12 s active-cost run completed with **47.897 mm / 0.372850 rad**, C1 pass,
zero strict dwell. Twelve candidate predictions were invalid and received
infinite cost. The same N=10 configuration is running for 180 s. A separate
35 s N=5 comparison ended at **54.430 mm / 0.465685 rad**, C1 pass, zero dwell;
there is no evidence to adopt the shorter horizon. `--planning-horizon`
now exposes the existing native staging option, defaults to 10, and is
recorded and checked against the actual isolated runtime.

Patch 0017 adds a separate, default-disabled coarse-model alignment option:
`--c3-osc-matched-coarse-model`. It temporarily generates the original coarse
LCS using the same 1 kg scalar actor as the calibrated cost model, then
expresses its input as the desired external force used by the receiver:
`u_actuation = -u_external - m*g`. Consequently B and H change sign, while d
and c receive the original B/H times the gravity-compensation force. Target
gravity and all contact terms remain. Both context masses are restored before
creating the ordinary fine model. This is a model and input-coordinate change,
not a change to the C3+ solver or the robot's force-action sign.

Runtime checks exercise the generated actor acceleration response, zero
actor gravity drift after compensation, and equality of the two contact-LCP
simulations under this affine input mapping. The option requires the calibrated
PD cost. A new 12 s trial is running; no outcome is claimed yet. The latest
0016 revision directly evaluates calibrated object-only cost instead of first
running an unused native PD model. Each run archives its exact patch and
executable version; the ongoing 180 s run was verified to retain its earlier
loaded executable after the new build.

The matched-coarse 12 s run completed: **55.090 mm / 0.153045 rad**, C1 pass,
zero strict dwell. All 239 logged current-location affine-contact checks
passed (maximum state difference **1.43863e-14**); each additional candidate
was checked as well. Seventy-six fine-model candidate predictions were
invalid and excluded. The exact goal hash and 240-frame/12 s video were
verified. A same-configuration 180 s run is now in progress alongside the
cost-only 180 s comparison. The cost-only comparison subsequently produced
the online success described above; the matched-coarse run subsequently failed
at 180 s as recorded below.

## First online success and independent repeat

The successful native executable SHA256 is
`bde00e4dacf064037e14f2c5ab0e0d1d8dcdc48ac7a8d71c3446c94805e10140`.
It uses patch 0016's initial cost revision,
N=10, FOH, matched 200/20 acceleration PD in candidate cost, physical OSC
200/800 with damping ratio sqrt(0.5), and the existing negative force sign.
The coarse alignment and new planner floor are disabled. Native fixed goal
mode 2 and its actual artifact hash were verified. The 172.65 s video has
3453 frames at 20 fps, consistent with the 172.66 s simulation endpoint.
Positive legal hand forces were recorded, and the terminal video frame was
inspected. Neither that video nor sparse trace samples substitute for a full
50-step dwell audit.

`frozen_relinearized_pd_bde` contains a copied controller executable, its
original patch snapshot, and `binary_provenance.json`. Existing pinned Drake
runfiles supply its libraries. A first repeat launch exposed Drake's strict
YAML schema: it rejects unknown optional keys even when false or null. Those
failed launch directories are retained. The stager now omits disabled optional
keys, including stale enabled values inherited from another trial. The repeat
and three other preexisting debug scenes are running in new directories.

`audit_strict_pose_dwell.py` recomputes XY, height and full SO(3) from normalized
measured quaternions and the fixed goal. It requires all 50 consecutive terminal
step indices and matching post-step timestamps. Missing samples are never
interpolated. The runner now records every strict-pose step even after contact
ends; this is a logging change. The formal protocol requires that independent
dwell certificate in addition to its existing physical contact and C1 checks.
The staging, acceptance and dwell audits have 18 passing tests.

## Planner finger floor: geometry correct, constraint verification incomplete

Patch 0018 optionally computes the minimum actor height from the actual two
finger convex hulls, current measured hand orientation, table top and 2 mm
clearance. The 12 s trial has 122 exactly synchronized geometry comparisons:
native versus executor floor differs by at most **1.79e-8 m**. However, 19 of
238 current C3 plans violate the requested lower bound by more than 10 micrometres;
the maximum is **0.895 mm**. This exceeds float32 reporting error. The option
is not part of the frozen successful configuration and is not considered fully
verified. The physical 12 s trial ended at **50.482 mm / 0.431825 rad**, zero
strict dwell and zero C1 violations. This is a task failure.

The matched-coarse 180 s run subsequently completed with XY approximately
**2.6 mm**, but full SO(3) **0.8714 rad**, zero strict dwell and zero C1
violations. It is a strict task failure and does not replace the earlier
successful cost-only configuration. The archived cost-only controller is
now also running the separate eight-scene random-initial-yaw debug manifest
with seed 20260910. That manifest does not overlap the 50 acceptance scenes.

An additional scene-contract audit checks actual initial and goal orientations
against the pinned semantic support pose, absolute manifest yaw and frame
offset, and checks both Isaac and native goal translations. The first online
success passes this audit. The old initial-45/goal-55-degree smoke also matches
the expected Isaac poses, but predates native-goal artifact recording and is
not treated as a complete goal-contract pass. Across staging, scene-contract,
dwell and acceptance checks, 25 distinct tests passed in two overlapping groups.
