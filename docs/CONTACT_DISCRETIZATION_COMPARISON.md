# Captured contact discretization comparisons

Neither increasing the number of retained gripper contact pairs nor increasing
friction-cone directions shows consistent improvement across the two exact
captured windows. These changes remain offline diagnostics.

Both windows use sampled 20 Hz plan / 100 Hz reference clocks, 2.5 ms model
steps, the same measured initial state and arm pose, and exact 50 ms physical
endpoints. All five servo samples have legal hand contact and no C1 guard.
The frozen storage-optimized binary `f06501f...` is used throughout.

| Model | 8.30 s XY / SO(3) error | 106.35 s XY / SO(3) error |
| --- | ---: | ---: |
| Baseline: one gripper pair, four friction rays | 0.458 mm / 0.020371 rad | 3.614 mm / 0.027967 rad |
| Two gripper pairs | 0.458 mm / 0.020371 rad | 4.152 mm / 0.032116 rad |
| Eight friction rays | 0.143 mm / 0.016170 rad | 3.798 mm / 0.052064 rad |

For the two-pair case only the cost-model contact-list index changes. The
original C3 optimization contact index is retained. For the eight-ray case,
`num_friction_directions` changes from 2 to 4; native nonplanar contacts use
twice that many rays. Sixteen fixed tangential/conic weight vectors must also
expand from 20 to 40 entries, retaining each contact's constant weight. The
initial attempts omitted this dependent expansion and aborted during native
construction; their logs are preserved and excluded from numerical comparisons.

All replays evaluate supplied captured plans without running a new C3 solve
or publishing a physical command. The two windows are debugging evidence,
not independent task-success trials. The active physical trials retain the
baseline contact count and friction discretization.

Artifacts: `offline_two_contact_pairs` and `offline_eight_friction_rays` under
`outputs/contact_planner_m3/workspace_height_20260908`.
Evidence: `evidence/contact_planner_m3_two_contact_pairs_20260909.json` and
`evidence/contact_planner_m3_eight_friction_rays_20260909.json`.
