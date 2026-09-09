# Precise native PD replay input capture

Patch 0022 adds a default-off `PUSH_ANYTHING_PD_CAPTURE_TIME_S` diagnostic.
At the selected planning timestamp it prints the current solved raw states and
forces with 17-digit precision. It does not run another model, change costs,
select a contact or replace the command. The isolated `osc_capture_variant`
evaluator records the requested capture time and the execution source hashes.

The diagnostic archive `frozen_precise_capture_0020_0022` deliberately excludes
patch 0021 so it can reproduce the failed patch-0020 full-inertia trajectory.
Its binary SHA256 is
`86461625f4560da86a1bead3fae5d1fa970e317283f1bedb6d87260d23dfbe2a`.
The canonical source and default binary have been restored and rebuilt with
patches through 0022, including 0021. Active physical trials use frozen copies.
The first insertion location broke the older patch-0014 reverse check; that
build is retained as `rejected_precise_capture_patch_context`. The revised
location passes the patch checks and builds successfully.

The 12 s physical regression reproduces all 240 measured object states of
`osc_full_variant12` exactly. At 2.35 s, all captured 11x19 state entries and
10x3 force entries are finite, and the raw actor positions and forces match
the published executed reference exactly. This is a logging regression, with
zero strict dwell and no additional task success.

That short capture has no retained joint pose at exactly 2.35 s, so it is not
used to construct a physics replay input. The selected long-run capture time,
106.35 s, does have an exact joint-pose record in the original trajectory.
The 107 s capture run has finished. All 2,140 measured object states match
the original trajectory prefix exactly, including and after the capture event.
The single 106.35 s capture has an exact measured joint pose, and its raw actor
positions and forces match the published reference exactly.

`scripts/prepare_offline_pd_capture.py` requires a unique fresh command and
exact measured arm pose, checks published positions and forces against the
native double capture, and emits the existing offline replay input format.
Three tests cover exact extraction, force mismatch, missing joint state and
stale held plans. Only the first 50 ms precedes the next replanning command;
longer native predictions are not an observed open-loop experiment.

Evidence is in `docs/evidence/contact_planner_m3_precise_pd_capture_20260909.json`.

The offline replay diagnostic can now log every fine state in the first 50 ms
with `PUSH_ANYTHING_OFFLINE_PD_FIRST_PERIOD=1` (default off, offline only).
At resolution 30 this provides 20 states at 2.5 ms intervals, so the endpoint
does not rely on interpolation between 75 ms model knots. The archived binary
`frozen_offline_first_period` has SHA256
`8deb4cb66c98d48abf3a5f093720384b33543946a28c6b5f3ec793a6105f5837`.
It builds successfully and all 11x19 coarse output values remain exactly equal
with logging enabled/disabled and to the historical 8.3 s replay baseline.

`scripts/analyze_offline_pd_first_period.py` checks physical source hashes,
exact start/end timestamps, all five servo references, legal contact and C1
flags. Two tests cover known prediction error and rejection of missing fine
states or mistimed endpoints; missing servo samples make a comparison ineligible.
The old 8.3 s calibration window gives a 50 ms prediction error of 0.458 mm XY
and 0.020371 rad SO(3). Its first governed position differs from the raw
reference by 0.743 mm; subsequent positions and all feedforward forces match.
This validates the comparison pipeline, not task generalization. Artifacts
are under `offline_first_period_regression`.

The exact 106.35--106.40 s comparison is now available in
`offline_precise_t106p35`. All five measured servo position/velocity references
and applied feedforward forces match the offline references to roundoff; no
C1 guard intervenes and each sample has positive legal hand contact. The
2.5 ms sampled model still has XY error 3.614 mm and SO(3) error 0.027967 rad.
Thus reference governing does not explain this particular prediction error.

The original resolution-4 offline replay reproduces all 11 published object
positions and quaternions exactly. Its interpolated first-period task-cost
drop is +2.367; the exact sampled resolution-30 drop is -0.681, while the
physical drop is -4.890. The diagnostic metric is normalized XY/SO(3), not
the native optimization objective. This changes the earlier interpretation:
the precise model also predicts regression for this window.

Smaller integration steps do not establish convergence: sampled resolutions
15 / 30 / 60 yield XY errors 1.378 / 3.614 / 7.718 mm and SO(3) errors
0.001121 / 0.027967 / 0.074357 rad respectively. LCP zero tolerances 1e-8 and
1e-6 leave these first-period results unchanged to roundoff. Resolution 15
is an experimental candidate, not a validated model correction. Two physical
180 s debug trials (scene007 and random-yaw scene000) compare it with
the existing resolution-4 partial-OSC/patch-0020 baselines. Their first
attempts stopped at 2 s with zero fresh commands: the 1 s wall-clock wait
was insufficient. These are startup failures, not completed task comparisons.
New `_wait10` runs allow 10 s of wall-clock computation while keeping the
100 us timestamp check, 20 Hz simulation planning clock and task thresholds.
The physical simulator waits for each reply; the socket timeout is 15 s.
The isolated
`pd_resolution_variant` changes only optional model-resolution staging and
evaluation checks; all other sampling parameters compare equal in staging.
