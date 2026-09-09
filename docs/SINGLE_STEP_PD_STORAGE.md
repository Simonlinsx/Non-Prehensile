# Single-step PD model storage

The relinearized PD cost regenerated an LCS at every fine step with storage
for `N * resolution` copies of each coefficient. The existing C3 `LCS::Simulate`
reads only coefficient index zero. The factory computes one set of matrices
and its constructor replicates them for the requested storage length.

Patches 0016 and 0019 now request `N=1` for these single-step instances. The
outer rollout still advances all `N * resolution` steps and returns the same
coarse states. The C3 optimizer horizon, integration step, contact resolution,
cost formula, physics controller and original C3 core are unchanged.

The native build and patch-chain checks pass. Frozen binary
`frozen_pd_single_step_storage` has SHA256
`f06501f9e01db95e980cbc80f08a19a15d1f2b3035a43c171f622fdb5f0593a2`.
It includes canonical patches through 0022, including 0021. Existing physical
trials continue to use their original archived patch-0020 binary.

Five captured-input configurations, each compared twice against frozen
`8deb4cb...`, produce exactly equal native prediction and reference records:
active FOH resolutions 4 and 15, sampled resolutions 30 and 60 at 106.35 s,
and the earlier 8.3 s sampled calibration input. Sampled comparisons include
every logged fine state in the first 50 ms. Active comparisons cover all
11x19 coarse states.

Paired process times under concurrent simulation load improve from about
0.91 to 0.41 s for sampled resolution 30, and 4.35 to 0.62 s for resolution 60.
Active FOH resolution 15 improves more modestly, about 0.32 to 0.28 s. These
are illustrative offline timings, not a guarantee of real-time execution or
evidence of a new task success. A separate physical regression completed in
`single_step_storage_full_aligned12`, using full OSC and patch-0021 clearance
to match the existing `osc_full_aligned_clearance12` baseline. All 240 measured
object states and all 140 retained servo trace rows match exactly. The fixed
scene poses, native goal hash and execution source hashes pass independent
checks. The result retains zero C1 violations and zero strict dwell: it is
physical regression evidence, not a task success. Its only additional
evaluation setting is a 10 s synchronous simulation wait; timestamp checks and
physical timing remain unchanged.

Reproduction script and logs:
`outputs/contact_planner_m3/workspace_height_20260908/single_step_storage_regression`.
Evidence: `evidence/contact_planner_m3_single_step_storage_20260909.json`.
