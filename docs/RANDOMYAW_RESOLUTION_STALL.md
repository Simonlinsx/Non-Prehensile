# Random-yaw resolution comparison: frozen 70-second prefix

The resolution-15 trial remains in progress. This comparison uses only a
frozen prefix through native timestamp 70,000,000 us of debug scene000, against
the completed resolution-4 finite-hysteresis baseline. Neither prefix is an
acceptance result.

Artifacts are under
`outputs/contact_planner_m3/workspace_height_20260908/randomyaw_resolution_candidate_prefix70/`.
Each trial retains selected original measured-state, candidate-cost,
task-reference and object-plan JSONL rows. `comparison.json` records SHA256
hashes of these immutable subsets, not hashes of the growing source log.

All 1,398 candidate-cost timestamps per trial match a recorded command mode.
Neither trial has an all-invalid candidate vector in this prefix, and neither
selects C3 at an invalid-current/finite-alternative boundary.

The interval (40, 70] native seconds contains 600 measured states and 600
candidate-cost/command pairs per trial:

| Measurement | Resolution 4 | Resolution 15 |
| --- | ---: | ---: |
| C3-mode commands | 168 | 184 |
| All-invalid candidate vectors | 0 | 0 |
| Invalid candidate entries / all entries | 1054 / 4198 | 870 / 4197 |
| Maximum measured XY displacement from interval's first state | 0.000433 mm | 0.000100 mm |
| Published object plans predicting >1 mm XY movement anywhere in horizon | 21 / 587 | 2 / 568 |

Both objects are effectively stationary during this interval despite ongoing
planning and mode transitions. The finer rollout has not resolved this stall.
The records rule out all-candidate invalidity as its immediate explanation;
they do not rule out effects from individual invalid candidates. Published
object plans can be held, so their counts do not independently establish fresh
command effectiveness. These records also do not establish whether a legal
contact occurred at every C3-mode command.

Continue both live resolution trials to their terminal result. If the random
scene remains unsuccessful, inspect the stalled interval's raw C3 trajectory,
calibrated predicted motion and executed contact/reference together before
another parameter change. The matched-coarse trial is a separate diagnostic
and cannot be counted as random-scene generalization.
