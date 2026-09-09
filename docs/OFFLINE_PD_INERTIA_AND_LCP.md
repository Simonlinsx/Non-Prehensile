# Offline measured inertia and contact solver checks

The measured translational inertia tensor does not consistently improve the
captured physical predictions. It remains an offline diagnostic; both running
debug batches use the archived successful `bde00e4...` controller unchanged.
Formal 50-scene acceptance has not started.

`prepare_offline_pd_inertia.py` reconstructs `(Jv M^-1 Jv^T)^-1` from recorded
Franka joint dynamics, verifies the task-point frame and partial OSC setting,
and compares against the cached controller matrix. Differences are below
1.2e-6 kg. Principal inertias are approximately 0.76, 0.98 and 9.2 kg. The
cached state precedes the final 2.5 ms physics substep; the frozen tensor omits
arm rotation, changing inertia, nullspace, joint limits and the position governor.

Patch 0019's offline sampled-reference function accepts a row-major 3x3 matrix
in `PUSH_ANYTHING_OFFLINE_PD_INERTIA`. It transforms the unit-mass actor's
Anitescu B/D/F/H blocks together and scales PD gains and gravity compensation
by the tensor. A/E/d/c retain target dynamics. At ten coarse states it compares
all eight coefficients against direct multibody regeneration at masses
0.25, 2 and 10 kg, and compares two contact LCP outputs for each mass. The
original C3 core and active calibrated-cost function are unchanged.

Default Moby tolerances proved sensitive to roundoff: coefficients agreeing
within approximately 1e-15 could yield visibly different simulated states.
Additional logs showed both solves returning true, but residuals against the
original unregularized contact problem differed. Drake's regularized solver
can succeed on a perturbed matrix. This does not establish a failed solver
return or a causal explanation for the physical cross-scene failures.

`PUSH_ANYTHING_OFFLINE_PD_LCP_ZERO_TOL` permits an explicit offline tolerance.
At 1e-8 and 1e-6, both windows pass all scalar-factory comparisons (maximum
state difference below 7.3e-13). Predictions across those two tolerances differ
by at most 1.1e-10 over all 11x19 states. Tolerance 1e-10 still fails a check in
the second window. This validation does not certify every actual fine-step LCP.

All comparisons below use the same 2.5 ms integration step, sampled reference
clocks and 1e-8 tolerance:

| Model | 8.300–8.600 s XY / SO(3) error | 8.350–8.650 s XY / SO(3) error |
| --- | ---: | ---: |
| Scalar 1 kg | 8.998 mm / 0.061486 rad | 8.488 mm / 0.017393 rad |
| Measured tensor | 20.376 mm / 0.016151 rad | 16.588 mm / 0.065811 rad |

Tensor XY error is worse in both windows; rotation improves only in the first.
The windows overlap and are from one tuned debugging scene, so they provide
no independent task success evidence. Seventeen input, reference-clock and
inertia extraction tests pass. Native build and all patch reverse checks pass.

Exact commands, environment overrides, inputs, logs, comparisons, binary and
source snapshot are under
`outputs/contact_planner_m3/workspace_height_20260908/offline_pd_tensor_tolerance`.
Earlier failed checks are retained in `offline_pd_tensor_t8p3`,
`offline_pd_tensor_t8p35` and `offline_pd_lcp_status`.
Evidence: `docs/evidence/contact_planner_m3_offline_pd_tensor_20260909.json`.

Two further physical debug failures are retained: the front-direction batch
scene001 ended at 180 s with XY 21.178 mm / SO(3) 0.652748 rad; random-yaw
scene000 ended at 180 s with XY 53.459 mm / SO(3) 0.400190 rad. Both had zero
strict dwell, positive legal contact and no C1 violation. Their actual initial,
Isaac goal and native goal poses match the manifest, with native goal hashes
verified. They remain failed tasks in their respective debug batches.
