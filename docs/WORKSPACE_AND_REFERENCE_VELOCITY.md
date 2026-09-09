# Workspace and reference-velocity diagnosis

Continuation of the 2026-09-08 dynamics alignment. Fixed scene007 remains a
debug scene, separate from the frozen 50-scene >60% acceptance goal.

## Confirmed issues and changes

1. Native C3 omits actor workspace linear constraints when `include_walls` is
   true. Object-wall contacts do not bound the actor. Patch
   `0009-actor-workspace-constraints.patch` adds optional actor constraints
   regardless of virtual walls. `--enforce-actor-workspace` enables them using
   the same 30 mm horizontal inward margin as native safe-sample generation.
   Vertical planes retain the native unshrunk interval. The first four
   comparisons below used an initial patch with a 30 mm vertical margin too;
   that snapshot is saved as `initial_actor_workspace_patch.patch` in the
   output root. Height-floor trials use the corrected horizontal-only margin.
   The hard
   measured-state workspace limits and stop assertions are unchanged.
   This currently constrains the linear workspace planes, not the annular
   radius limits; full execution-boundary validation remains necessary.
2. Planar execution overwrites C3's optimized EE z with `sampling.z_height`
   (and the upstream near-wall offset). `--execution-height-mode optimized`
   uses the original `planar_demo: false` option to preserve optimized z;
   the default remains `planar`. This is an ablation, not task acceptance.
3. The position governor previously differentiated the reconciliation jump at
   every MPC restart as if it were a physical trajectory. At 24 s in the
   optimized-height trial, C3 requested +0.14 m/s along x while the OSC received
   about -0.234 m/s. This created outward drift even when the future C3 knots
   moved inward. The default `--osc-reference-velocity-mode trajectory` now
   retains the local trajectory derivative, while keeping position and speed
   limits. The historical behavior is `--osc-reference-velocity-mode governed`.
   Holds still clear velocity. No goal-dependent pushing direction is added.
4. Task-reference audit rows now retain execution trajectory knots, knot times
   and command flags. Evaluator configurations record the input-manifest,
   native-binary and execution-source hashes and reject changed/unversioned
   resume attempts.

## Completed comparisons

All are under `outputs/contact_planner_m3/workspace_height_20260908/`, with
v4 measured physics, simulation clock, task-point OSC and C1 enabled.

| Trial | Executed | XY | SO(3) | C3 planar velocity reversals | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| bounded_planar35 | 35 s | 49.598 mm | 21.844 deg | 63/116 | C1 pass, task fail |
| bounded_optimized35 | 25.36 s | 49.613 mm | 21.630 deg | 57/90 | Workspace abort, C1 pass |
| trajectory_velocity_planar35 | 35 s | 48.410 mm | 18.488 deg | 0/184 | C1 pass, task fail |
| trajectory_velocity_optimized35 | 35 s | 48.705 mm | 19.110 deg | 0/186 | C1 pass, task fail |
| positive_force35 | 35 s | 47.864 mm | 18.674 deg | — | C1 pass, task fail |
| final_qp35 | 35 s | 48.882 mm | 18.390 deg | — | C1 pass, task fail |
| geometry_floor35 | 35 s | 48.521 mm | 19.572 deg | — | C1 pass, task fail |
| spatial_geometry_floor35 | 35 s | 46.985 mm | 21.375 deg | — | C1 pass, task fail |

Reversal counts use recorded C3-mode rows with planned horizontal speed
>0.01 m/s and a negative dot product between planned and applied horizontal
velocity. They are sampled diagnostics, not a continuous-time fraction.
Minimum measured x after the velocity fix was 0.2935 m (planar) and 0.2808 m
(optimized), versus 0.2285 m in the failed old-velocity trial. Longer boundary
and task validation is still required.

## Further controlled checks

`--c3-force-action-sign 1` compares the C3 input sign used by its positive-input
PD rollout against the historical native-OSC external-force mapping (-1).
`--c3-end-on-qp-step` selects the upstream final-QP option, so cost computation
and execution use the same returned state/input solution. The upstream source
explicitly documents differing full-solution and state/input getters when this
option is false. These are diagnostic alternatives; defaults remain -1 and
false pending evidence. Neither alters success or C1 thresholds.

`--task-height-floor-mode finger-geometry` computes the lowest allowed task
reference from the measured hand orientation and both exported PhysX finger
convexes, with 2 mm table clearance. This replaces the old sphere-calibrated
-5 mm reference floor in this ablation only. `--spatial-safe-sampling` selects
the native `gen_planar_samples: false` option while preserving semantic-safe
meshes. Geometry floor and spatial sampling remain off by default until live
contact and clearance validation is complete.

## Solver convergence follow-up

The relay now records raw optimizer object plans, PD rollout plans, sample
locations/costs, and the native C3+ solution/intermediates separately. The
solver message is `c3.lcmt_output`, not the obsolete dairlib message advertised
in the channel YAML comment. Patch 0010 adds its generated Python dependency.
Two initial diagnostic runs stopped at startup on that decode mismatch; their
error artifacts are retained. The corrected relay built and ran successfully.

`analyze_c3_solver_consistency.py` compares both predictions at knot 9 (0.675 s
for this configuration). It excludes frozen zero-input fallback solutions
from contact-product summaries because native fallback retains stale eta.
For `qp_solver_diagnostic12_v2`, raw current-location XY prediction has median
159.685 mm while the best candidate's PD rollout has median <0.001 mm.
These are different channel distributions across all modes, not a paired
prediction-error statistic. The raw solver also returns negative lambda
components (median maximum magnitude 0.1583 in native coordinates). A
successful QP solve does not imply convergence of C3's contact complementarity.

The original final-QP contact penalty covers only EE coordinates 0–3.
`--c3-final-contact-scaling-mode all` applies that same existing 1000 multiplier
to all 18 Anitescu coordinates. At the original 200-QP-iteration budget,
`all_contacts_qp12_v2` logged 261 failed QP solves and ended at 47.748 mm XY,
0.14349 rad SO(3), C1 pass and task fail. The count spans candidate solves and
ADMM substeps, not 261 simulation cycles.

`--c3-qp-max-iterations 2000` materializes an isolated solver YAML; upstream
and other runtimes stay unchanged. With the original EE-only penalty, the
12 s result reproduced exactly. With all-coordinate scaling, QP failures
disappeared but `qp2000_all12` stopped at 4.30 s on the C1 distance gate
(11.314 mm minimum forbidden distance). Contact sensors reported zero force
on that final row; this is a conservative C1-distance rejection, not evidence
of a measured forbidden impact. It remains a rejected trial and cannot be
promoted. Follow-up retained EE-only scaling and compared 6/10 ADMM
iterations with a 2000-iteration QP budget. Both completed 35 s without C1
violations: 6 iterations ended at 32.185 mm / 0.65656 rad, and 10 iterations at
39.813 mm / 0.26303 rad. Neither passed the strict task gate. The median
maximum negative-lambda magnitude fell from 0.1330 (6 iterations) to 0.0704
(10 iterations), but was still materially nonzero. The latter trial showed
late XY and orientation improvement. Its 180 s attempt subsequently stopped
at 40.84 s on the C1 distance gate, ending at 115.958 mm / 0.18870 rad. It is
not a completed 180 s run and this configuration cannot be promoted.

Patch 0011 optionally adds lambda >= 0 through the existing unmodified C3
`AddLinearConstraint(..., FORCE)` interface. Anitescu lambda represents
nonnegative friction-cone rays. This bound is necessary in the original
complementarity feasible set; enforcing it in each finite-iteration QP
tightens the relaxation. It does not enforce eta >= 0 or lambda*eta = 0 by
itself. `--c3-nonnegative-contact-forces` is an experimental controller-side
constraint, off by default. C3's core solver sources remain unchanged.
The 35 s, 3-ADMM-iteration trial completed without C1 violations but failed
the task at 49.801 mm / 0.31250 rad. Across messages after 4 s, its maximum
negative-lambda magnitude was 1.998e-5 and median 1.054e-14. This validates
the nonnegative bound to solver tolerance, not contact complementarity or
task performance. Native quaternion-Hessian weights 100 and 10 are now being
compared against the unchanged default 1000, with all strict gates preserved.
Both 35 s trials completed without C1 violations. Weight 10 ended at
15.120 mm / 0.32231 rad, and weight 100 at 18.600 mm / 0.18469 rad (see JSON
for exact values). Position passed the strict threshold but orientation did
not. No strict task success has been established.

Offline reconstruction from recorded hand poses and shared finger convexes
also found near-table contacts despite the geometry floor's 2 mm reference
clearance: sampled minimum finger z was -0.081 mm in the 10-iteration trial.
Reference clearance alone does not certify measured clearance under tracking
error. The geometry floor therefore remains an experiment rather than a
validated table-clearance guarantee.
`--task-height-clearance-m` exposes this reference clearance (default 2 mm).
A 5 mm, 35 s comparison and a 2 mm, 180 s same-run video trial both completed
with task failure and C1 pass. The latter ended at 12.579 mm / 0.31444 rad.
Their evaluation roots preserve exact source/input/binary
bytes in `source_snapshot`. The initial video launch had a duplicate CLI
argument introduced during wiring; it was fixed before simulation startup,
and the CLI preflight passed. The final focused regression run passed 48
tests with one skipped Isaac-dependent test.

The [subsequent semantic-frame audit](SEMANTIC_FRAME_AND_MEASURED_CLEARANCE.md)
corrected two native C1 quaternion constructors and started repaired-binary
weight-10/100 video comparisons. It also replaces approximate hand-based
clearance estimates with each finger's actual pose and cooked convexes.
The old weight-100 long trial stopped at 57.53 s on C1 distance; it is a task
failure, not a completed 180 s trial. Gate 1 remains open.

The [machine-readable evidence](evidence/contact_planner_m3_workspace_solver_20260908.json)
preserves task/safety outcomes, hashes, solver failures and prediction summaries.
None of these debug runs counts toward the frozen 50-scene acceptance set.
