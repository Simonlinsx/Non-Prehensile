# Contact Planner M2: Restored Physics Rollouts

## Status

M2 now has an **accepted single-scene native Isaac Lab execution**, but not an
accepted randomized result.  The accepted run satisfies XY, height, full
SO(3), dwell, and C1 simultaneously; an eight-scene +/-90 degree batch remains
0/8.  These two facts must be reported together rather than turning one
deterministic success into a robustness claim.

The same executor can audit C2 and C3 during restored rollouts through
`--safety-scope`, but clutter currently has only one-push smoke coverage.
RGB-D perception remains outside M2.

## What changed from M1

- Snapshot and restore the complete physical scene before every candidate.
- Evaluate several semantic/IK-valid macro-actions in Isaac physics.
- Require a real legal-safe contact and reject every C1-violating rollout.
- Rank the strict joint predicate (XY, height, full SO(3)) with normalized
  exact penalties instead of comparing separate minima.
- Preserve positive, neutral, and negative contact-moment hypotheses.
- Search the full push-direction circle and 3/7/11/15 mm push distances.
- Synchronize the latched action controller after every scene restore.
- Optionally use a two-step local-effect shooting horizon, while executing
  only its first action and replanning from the observed state.
- Latch the first physical legal-safe contact and reanchor the Cartesian push
  delta at that measured configuration.
- Retreat vertically from the measured post-push pose.  Reversing the old
  joint path after target motion was found to cause a second uncontrolled
  contact and is no longer used.
- When recording with physics rollouts, stream only formal execution frames;
  shadow candidates never appear in the video.

This remains sampling plus physics-based MPC.  It is not SCSP, contact-implicit
trajectory optimization, or a learned world model.

## Safety and restoration evidence

The targeted planner/bridge suite contains 26 passing tests.  In accepted run
`scene000_vertical_liftoff_selective_video_v30`:

- 160 candidate rollouts were evaluated and 112 were legal;
- all 10 formally executed pushes passed the physical contact gate;
- C1 violations, forbidden-hand contacts, and proximal-arm contacts were 0;
- mean shadow-to-formal disagreement was 0.617 mm translation and 0.00799 rad
  rotation;
- vertical-lift retreat removed the large target drift previously caused by
  reverse-path retreat.

The restore check itself is exact at the exposed articulation/rigid-body state.
Residual replay disagreement is caused by contact solver state/warm-start and
grows in contact-rich multi-candidate runs, so it must be included in the
planner margin rather than ignored.

## Strict result and bottleneck

The accepted same-run result and video are:

- `outputs/contact_planner_m3/isaaclab_closed_loop/scene000_vertical_liftoff_selective_video_v30.json`
- `outputs/contact_planner_m3/videos/native_c1_liftoff_v30/scene000_c1_audited_run-actual-only.mp4`

Their compact hashes and acceptance statistics are checked in as
[`contact_planner_native_c1_scene000_v30.json`](evidence/contact_planner_native_c1_scene000_v30.json).

| Metric | Result | Required |
| --- | ---: | ---: |
| Final XY | 0.01983 m | `< 0.020 m` |
| Final height | 0.00100 m | `< 0.010 m` |
| Final full SO(3) | 0.01367 rad | `< 0.100 rad` |
| Strict pose + dwell | yes | yes |
| C1 violations | 0 | 0 |
| Legal formal pushes | 10/10 | all |

Randomized robustness is still open.  The first eight-scene batch evaluated
749 candidates and executed 41/41 legal C1 pushes with zero C1 violations,
yet achieved 0/8 strict pose successes.  Final XY ranged from 2.65 cm to
9.96 cm.  The +63 degree scene that succeeds alone took a different finite
candidate branch in the batched GPU PhysX run and exhausted after six pushes.
The present bottleneck is therefore search coverage and sensitivity of finite
rollout ranking, not the contact gate or a loose terminal predicate.

## Reproduce

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=1 \
  bash scripts/run_contact_planner_m2.sh
```

Physics rollouts and video recording now share one simulator instance.  The
selective recorder emits only formal execution frames, so the MP4 and result
JSON above audit the same run.  Replaying stored candidate ranks is retained
only as a diagnostic because contact-rich GPU simulation is not bitwise
reproducible.

## Next milestone

Keep the accepted semantic/IK/C1 layer and first close the bounded +/-90 degree
randomized search gate.  Then replace the straight contact-to-push segment
with a short contact-phase trajectory optimization where necessary:

1. sample a legal safe contact and a small set of continuous-contact arc or
   two-segment paths;
2. roll out the entire contact trajectory in physics;
3. constrain protected clearance throughout the path;
4. optimize terminal SE(2)/SO(3) error and robustness to replay variation;
5. require multi-scene strict success before adding clutter, C2, or C3.

This adds the missing control freedom directly.  It is preferable to adding
more reward terms, stage-specific waypoints, or looser success thresholds.
