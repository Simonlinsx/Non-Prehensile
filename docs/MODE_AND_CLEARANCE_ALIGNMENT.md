# Candidate mode and execution-clearance alignment

Two execution-related boundary corrections are now under physical validation.
They preserve the closed stock fingers, safe sampler, C3+ solve and Cartesian
impedance controller. Patch 0020 now completes the previously successful tuned
scene in 45.91 s, compared with 172.66 s. Independent audit certifies all 50
terminal pose measurements, positive legal hand contact, zero C1 violations,
the fixed scene/goal contract and unchanged execution sources. This remains
one unique tuned scene; cross-scene success is not yet established.
The frozen 50-scene acceptance batch remains untouched.

## Nonfinite costs

Patch 0020 fixes relative hysteresis when the incumbent prediction is invalid.
Previously `infinity > finite + fraction * infinity` was false, preventing a
valid alternative from replacing an invalid current candidate. An invalid
previous reposition target could also add an infinite switching penalty to a
valid new target. Finite costs retain the original arithmetic and tie behavior;
invalid candidates cannot win a comparison against finite candidates.

The actual C++ helper passes finite-input and nonfinite-boundary assertions.
The 12 s scene007 comparison provides a direct physical example: the first
213 measured records are identical, and at 10.65 s both versions have identical
costs (`current=inf`, best alternative `559.1521345855648`). The old controller
stays in C3; patch 0020 switches to repositioning. The first measured physical
difference is at 10.75 s. Thus the full 12 s trajectories are **not identical**.
Patch 0020 ends with XY 49.030 mm / SO(3) 0.350570 rad, zero strict dwell and
zero C1 violations. This is a failed short task, with the intended boundary
behavior confirmed. Full 180 s paired trials are running.

The subsequent full scene007 run succeeds at 45.91 s, with XY 18.364 mm,
height 0.985 mm and SO(3) 0.071673 rad. Its video contains 918 frames at 20 fps.
Evidence: `contact_planner_m3_finite_hysteresis_success_20260909.json` under
`docs/evidence`. An eight-scene random-yaw batch now tests this archived binary.

`audit_nonfinite_cost_modes.py` joins candidate costs to the command's
`relay_state_utime_us`, allowing the native timestamp's 1 us truncation. It
does not use an older held plan's ID as the command timestamp. Two tests cover
held plans with changing watchdog mode and contradictory same-time commands.
Historical boundary counts are observations, not counterfactual task outcomes.

## Planning and execution distances

The stock closed-gripper staging used a 25 mm planning clearance and a 55 mm
execution stop distance. In the failed random-yaw debug scene001, 112 of 783
selected C3 predictions enter that gap; 75 already start below 55 mm. None are
below the original 25 mm planning clearance. This was computed from published
float32 plans, normalized object quaternions and the staged unsafe mesh, using
the native four-substep interpolation. Indexed and direct mesh queries agree.
No sampled prediction moves from a clear initial state into the stop region
within the first 50 ms. These distinctions limit the causal claim: the audit
does not explain every physical hold or certify all candidate trajectories.

Patch 0021 uses `max(requested clearance, execution stop distance)` for closed
finger planning, and logs the effective value. The execution stop distance
remains 55 mm. Legacy point models retain their requested planning clearance.
The original offline model prediction still matches all 11x19 baseline states
exactly; the native log confirms an effective 55 mm planning clearance. This
checks model preservation, not the planning filter's full physical behavior.
The separate 12 s physical regression passes: all 240 measured states match
patch 0020 exactly, and all 58 selected C3 plans meet the effective 55 mm
clearance within the audit's 0.1 mm classification margin. A full 180 s
comparison on the previously failed random-yaw scene001 is now running.

That 180 s comparison has now finished with strict failure: XY 225.032 mm,
SO(3) 0.765174 rad, zero dwell and zero C1 violations. All 732 selected C3
plans satisfy the effective 55 mm clearance within the 0.1 mm audit margin;
none start inside or predict entering the execution stop region within 50 ms.
The executor still guards 318 servo steps (3.18 s). Initial/goal poses, fixed
native goal hash and positive legal hand contact pass independent checks.
The planning constraint works on the retained predictions, but this run does
not establish a task-performance improvement. Its partial-OSC baseline had
failed with XY 177.160 mm / SO(3) 0.667050 rad.

Both native builds and all patch reverse checks pass. Binaries and exact patch
snapshots are archived as `frozen_finite_hysteresis_0020` (SHA256 `88b68b7...`)
and `frozen_aligned_clearance_0021` (`f528c59...`) beneath the experiment root.
Detailed evidence, full hashes and comparisons are in
`docs/evidence/contact_planner_m3_mode_clearance_alignment_20260909.json`.

The previous front-direction three-scene batch finished with 0/3 successes
and zero C1 violations; scene000 ended early with a workspace watchdog failure.
The first two random-yaw scenes also failed at 180 s with zero C1 violations.
The improved scene007 run remains the same unique strict debug success. These
debugging results do not establish the requested acceptance rate.

The patch-0020 front-direction scene000 also fails, at 104.21 s after a native
workspace-radius assertion and executor watchdog timeout. Its final XY error
is 336.303 mm, SO(3) error 0.575417 rad, with zero strict dwell and no recorded
C1 violation. Delaying the older 74.31 s failure does not resolve this scene.

Its retained raw references remain below radius 0.749333 m, and governed
references below 0.746206 m, while the measured task point reaches 0.750478 m.
The final retained tracking errors are roughly 5–7 mm while semantic protection
is active. Thus the retained evidence does not show an explicitly out-of-bounds
reference as the cause of this failure. Missing servo samples are not filled
in, and the force/inertia cause of the tracking error remains unproven. The
per-scene `workspace_radius_reference_audit.json` records these bounds and the
source result hash.
