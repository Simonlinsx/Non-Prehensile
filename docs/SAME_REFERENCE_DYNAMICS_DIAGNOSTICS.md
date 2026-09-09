# Same-reference dynamics diagnostics

The goal remains active: zero strict debug successes, and the frozen 50-scene
acceptance manifest has not been used. All runs below are **diagnostic** and
are explicitly rejected by the acceptance auditor, even if a pose happened
to pass during replay. They do not replace receding-horizon task validation.

## What changed

The relay can capture one freshly planned native C3 actor position/force
trajectory together with its matching PD object prediction, execute that
reference for a bounded interval, then resume normal replanning. Native C3
continues receiving measurements and solving. No force-C3 radio command is
introduced. C1, workspace, height, speed and force protections remain active.

`--diagnostic-reference-replay-start-s` is the **earliest planner-clock time**
for capture. The Isaac protocol clock has a 0.1 s origin offset from simulation
time. Capture waits for a current C3-mode plan and a matching object prediction;
it can therefore start later. The actual capture and resume times are logged.
`--diagnostic-reference-replay-duration-s` must fit within the published knots.
The default is disabled. Config/result markers prevent acceptance use.

Patch 0014 adds optional, read-only native output: `--diagnostic-compare-pd-references`
asks the same solved current-location plan, fine LCS, gains and initial state
to produce both ZOH and FOH PD predictions. It does not replace the executed
plan, candidate costs or chosen mode. The selected prediction is cross-checked
against the captured LCM prediction. C3 core remains unchanged except for the
preexisting inactive-MIQP compatibility shim in patch 0002.

The new replay auditor checks actual capture completion, one unchanged plan,
reference coverage, matching knots, measured-state coverage and actual positive
legal hand forces. It reports downstream reference/force modifications separately.
Trace references precede their recorded measured poses by one servo interval;
the audit accounts for that timing. Observed states between packets use
interpolation without extrapolation.

## Completed windows

All four runs completed 10 s with C1 pass and verified 200-frame, 20-fps videos.
None had a strict pose dwell. Paths are under
`outputs/contact_planner_m3/workspace_height_20260908/`.

| Run | Actual planner-clock window | Actual XY motion | Selected PD motion | Endpoint XY error |
| --- | --- | ---: | ---: | ---: |
| reference_replay_zoh_t8_diag10 | 8.000–8.675 s | 4.351 mm | 1.176 mm | 3.193 mm |
| reference_replay_zoh_t8p3_short_diag10 | 8.300–8.600 s | 18.440 mm | 2.374 mm | 16.281 mm |
| reference_replay_dual_pd_t8p3_diag10 | 8.300–8.600 s | 18.440 mm | 2.374 mm | 16.281 mm |
| reference_replay_inertia_t8p3_diag10 | 8.350–8.650 s | 26.033 mm | 1.891 mm | 24.421 mm |

The first window had three recorded shield-active samples and up to 30.16 mm
reference correction. Its complete horizon cannot establish an unmodified
contact-model error. Its initial physical state, native reference and PD
prediction exactly matched the prior normal-replanning run. Over the first
0.3 s, before recorded shield activation, normal replanning moved the object
26.411 mm, whereas the replay moved it 4.351 mm. Replanning is a material
confound in the earlier future-endpoint statistics.

The short 8.3–8.6 s window had no shield activation, 12 positive legal hand
force samples, maximum sampled raw FOH-reference mismatch 0.9745 mm, maximum
governor correction 0.7426 mm, and median tip tracking error 6.451 mm. Force
holding between 20 Hz packets differs from ideal FOH by up to 0.25293 N.
Thus a material contact-response discrepancy remains even with one reference.

The read-only dual-prediction run reproduced the entire 192-sample trace of
the short run exactly. Its logged ZOH object trajectory exactly matched the
selected LCM prediction. For the same plan/window, FOH endpoint error was
15.750 mm versus ZOH's 16.281 mm; both orientation errors were 0.063639 rad.
Interpolation accounts for little of this window's error. Raw final-QP target
prediction error was 102.361 mm / 1.15994 rad. FOH is not promoted to a task
configuration based on these results.

## Actual OSC inertia

`--osc-dynamics-audit` reads the existing action/controller caches at recorded
trace samples. It logs joint state, task-point Jacobian, joint mass matrix and
the operational inertia actually used by OSC. It does not change torque
computation. The snapshot comes from the last OSC call before the final physics
substep of the servo interval, not the post-step contact measurement.

The instrumentation run captured at 8.35 s, after waiting an extra 50 ms for
a matching prediction, so it is a different replay window from the 8.3 s pair.
The inertia audit uses its actual 8.35–8.65 s window, with 15 samples.
Reconstruction of `(J_translation M^-1 J_translation^T)^-1` agrees with the
cached partial-decoupling matrix to a maximum absolute difference of 2.16e-6.
Its principal inertia values span 0.7598–9.1578 kg; median principal values
are approximately 0.7609, 0.9851 and 9.1113 kg. The staged C3 point mass is
0.057 kg, verified from its actual URDF: a principal-value ratio of 13.33–160.66.

These matrices describe different aspects of the controlled mechanical system;
the ratio alone does not prescribe a scalar replacement mass or prove task
improvement. The next model calibration should account jointly for directional
inertia, force-domain PD gains, gravity compensation and the published-force
convention, then repeat a same-reference check before returning to full
receding-horizon task trials. Do not blindly increase the point mass, reverse
forces, or count replay as task success.

## Verification

Seventeen replay/real-window/acceptance/fixed-goal tests passed. Fixtures use
actual recorded windows, including mismatched-plan and missing-reference
rejection. Native patch 0014 built successfully and its reverse check passed.
All four goal-file hashes and video streams were checked. Every diagnostic
result is rejected by formal acceptance as intended; no results were edited.

See [machine-readable evidence](evidence/contact_planner_m3_same_reference_20260908.json).
