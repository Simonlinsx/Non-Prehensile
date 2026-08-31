# Contact Planner M2: Restored Physics Rollouts

## Status

M2 is an **implemented but not accepted** research prototype.  It improves the
M1 candidate ranker without weakening the C1 contract, but it has not yet
demonstrated robust simultaneous XY and orientation success.  It must not be
reported as a successful full-task planner.

The current scope remains one oracle-labelled DOMINO hammer, one goal, and no
clutter.  RGB-D perception, C2, and C3 are intentionally excluded until the
single-object contact dynamics are solved.

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

This remains sampling plus physics-based MPC.  It is not SCSP, contact-implicit
trajectory optimization, or a learned world model.

## Safety and restoration evidence

The targeted Python suite contains 21 M1/M2 planner tests, and the repository
suite passes 131 tests plus 7 subtests.  In the seed-17 scene-0 M2 run:

- 64 candidate rollouts were evaluated;
- 55 produced legal safe contact;
- all 7 formally executed pushes passed the contact gate;
- C1 violations, forbidden-hand contacts, and arm-target contacts were all 0;
- a one-candidate repeatability probe measured 0.73 mm position and 0.0075 rad
  rotation disagreement between the shadow rollout and formal replay.

The restore check itself is exact at the exposed articulation/rigid-body state.
Residual replay disagreement is caused by contact solver state/warm-start and
grows in contact-rich multi-candidate runs, so it must be included in the
planner margin rather than ignored.

## Strict result and bottleneck

The best current horizon-2 diagnostic is:

`outputs/contact_planner_m2/m2_rollout8_mpc2_actionreset_seed17_scene0_v16.json`

It is **not a success**:

| Metric | Result | Required |
| --- | ---: | ---: |
| Final XY | 0.0243 m | `< 0.020 m` |
| Final full SO(3) | 0.1490 rad | `< 0.100 rad` |
| Strict pose reached | no | yes |
| C1 violations | 0 | 0 |
| Legal formal pushes | 7/7 | all |

The best stable one-step multi-distance run ended at XY 0.0325 m and SO(3)
0.0926 rad, also outside the joint acceptance set.  Separate minimum errors
are not task success.

The important failure mode is now localized: a straight push at the safe
handle couples translation and rotation.  Near the goal, candidates that
improve XY rotate the hammer out of tolerance; candidates that correct the
rotation move it away in XY or cannot establish the intended safe contact.
Wider sampling, more scalar penalties, and larger contact penetration did not
remove this controllability limitation.  A two-step composition of local
effects delayed but did not eliminate the dead end.

## Reproduce

```bash
OMNI_KIT_ACCEPT_EULA=YES GPU_ID=0 NUM_ENVS=1 \
  bash scripts/run_contact_planner_m2.sh
```

Physics rollouts and video recording intentionally do not share one simulator
instance.  Once a planner passes quantitative acceptance, its selected action
sequence should be replayed separately for presentation video.

## Next milestone

Keep the accepted M1 semantic/IK/C1 layer, but replace the straight
contact-to-push segment with a short contact-phase trajectory optimization:

1. sample a legal safe contact and a small set of continuous-contact arc or
   two-segment paths;
2. roll out the entire contact trajectory in physics;
3. constrain protected clearance throughout the path;
4. optimize terminal SE(2)/SO(3) error and robustness to replay variation;
5. require multi-scene strict success before adding clutter, C2, or C3.

This adds the missing control freedom directly.  It is preferable to adding
more reward terms, stage-specific waypoints, or looser success thresholds.
