# OSC inertia comparison, 2026-09-09

The current closed-gripper executor uses IsaacLab's partial inertia decoupling
with all six task axes controlled. In the installed IsaacLab 2.2.0 controller,
partial mode separates translational and rotational task inertia and uses a
kinematic nullspace projector. Full mode uses the complete task inertia and a
dynamically consistent nullspace projector. This is an existing framework
option; its effect on this task requires physical simulation.

`scripts/audit_osc_nullspace_coupling.py` reconstructs the posture torque from
recorded joint dynamics and the preceding servo target. Missing preceding
records are excluded. In the earlier 8.3 s reference diagnostic, 90 consecutive
pairs show median task acceleration contributions of 0.133 m/s² translation
and 1.028 rad/s² rotation from the partial-mode nullspace term. The full-mode
float64 projector reduces that reconstructed component below 2.1e-14. This
does not measure total physical acceleration or prove a cause of task failure.
The observed full task inverse-inertia condition number ranges from 3716 to
5325; this one trajectory does not establish conditioning across random poses.

An isolated source copy at
`outputs/contact_planner_m3/workspace_height_20260908/osc_inertia_variant`
adds an explicit `--osc-inertia-mode partial|full` option. Its provenance and
patch record the two changed files. The original execution sources remain
unchanged while the four existing long batches run. Both modes use the same
archived patch-0020 native planner, gains, closed gripper, contact model and
safety constraints.

The isolated partial-mode 12 s run reproduces all 240 measured object states
of `finite_hysteresis_pd12` exactly. Final XY, height, SO(3), maximum torque
and C1 observations also match. Its result is a short task failure, not an
acceptance success. The comparison is saved in
`osc_partial_variant12/scene007/partial_baseline_comparison.json` beneath the
experiment root. The full-mode 12 s physical comparison also finishes without
strict success or a C1 violation. Its final XY error is 35.693 mm versus
49.030 mm, and SO(3) error is 0.185641 rad versus 0.350570 rad. Maximum joint
effort is 13.086 Nm versus 12.806 Nm. Semantic protection is active for 97
control steps (0.97 s), versus zero. The counter counts guarded servo steps,
not separate episodes. The smaller endpoint errors do not establish better sustained
execution.

All 138 full-mode dynamics snapshots confirm that partial decoupling is off.
The cached task inertia is finite and differs from float64 reconstruction by
at most 4.5e-5 in absolute matrix entries. The measured trajectory's full task
inverse-inertia condition number stays between 3745 and 5066. Initial and goal
poses, native fixed-goal hashes, positive legal hand contact and execution
source hashes pass independent checks in both runs. Evidence is saved in
`docs/evidence/contact_planner_m3_osc_inertia_comparison_20260909.json`.

The full-mode short run selects 107 current-location C3 plans. Twenty enter
the gap between the 25 mm planning clearance and 55 mm execution stop distance;
nine predict entering the stop region from a clear current state within the
first 50 ms. Eight start inside the stop region. None violate the original
25 mm planning limit. The partial-mode baseline selects 58 plans with none of
these conditions. This is evidence of a remaining planning/execution mismatch,
not a claim that all 97 guarded steps have the same cause.

Counter increments place all 97 guarded steps between retained measurement
times 5.41 and 6.61 s. The per-plan 50 ms join is too sparse to identify the
guarded steps for the nine entry predictions: five have no retained servo
sample, and the other four retain only a clear first step. No missing samples
are interpolated. A short combination trial with the archived patch-0021
planner is running as `osc_full_aligned_clearance12`; the full-mode patch-0020
180 s trial continues separately.

The same isolated full-mode configuration was extended to a 180 s limit in
`osc_full_scene007_180`; its outcome is recorded below.

## Long-run outcome

The long runs do not support promoting full inertia decoupling. Scene007 with
patch 0020 stops at 129.91 s after a workspace-radius assertion, with XY
280.626 mm and SO(3) 0.246349 rad. The patch-0021 combination runs to 180 s but
also fails, with XY 24.344 mm and SO(3) 0.427776 rad. Its protection counter is
only 21 servo steps, so removing most guarding does not produce task success.
The first independent random-yaw full-mode scene also fails at 180 s, with
XY 55.006 mm and SO(3) 0.438920 rad. All three retain zero strict dwell and zero
C1 violations. Their initial and fixed-goal contracts and positive legal hand
contact have been independently checked.

Full-mode dynamics snapshots remain finite. The maximum full inverse-inertia
condition numbers are 7146, 5066 and 4375, respectively; no task-success claim
follows from these numerical checks. Detailed metrics are in
`docs/evidence/contact_planner_m3_osc_inertia_long_followup_20260909.json`.

The old interactive process handles disappeared, and a host-level process
check confirms that those simulations are no longer running. Their completed
per-scene results are preserved, including results not yet incorporated into
batch summaries. The remaining random-yaw scene001 is resumed with identical
configuration in tmux `c1_full_randomyaw_resume_20260909`, with a separate
`persistent_resume_state.json` and log. The first completed scene is reused.

The next diagnostic uses exact 50 ms measured endpoints and fresh plans.
`audit_c3_first_period_effects.py` finds three eligible disagreement windows in
the failed full-mode trajectory, at 106.35, 116.35 and 119.50 s. These have all
five servo observations, positive legal contact and no safety protection. The
predicted 50 ms poses are interpolated from the published 75 ms knots, so the
windows identify candidates for nonlinear replay; they do not isolate a cause.
Tests cover stale plans, exact endpoint timing, missing samples and protection.

Full mode changes both task inertia and nullspace projection. Any improvement
must therefore be attributed to the combined mode change unless a further
controlled experiment separates them. Neither mode comparison counts toward
the frozen 50-scene acceptance set.
