# Affordance-aware non-prehensile teacher

This document fixes the scope and acceptance contract for the oracle-affordance
teacher.  RGB-D affordance prediction and deployment dynamics estimation are
deliberately out of scope until this teacher passes the gates below.

## Actor contract

The actor receives only quantities that a later RGB-D student can reproduce:

- 512 target points with aligned `[x, y, z, safe, protected]` features;
- 512 clutter points with `[x, y, z]` geometry;
- Franka proprioception, previous action, relative SE(3) goal, and noisy target
  twist.

The flat observation has 4,141 values.  Exact mass, friction, restitution,
future state, and simulator contact flags are not actor inputs.  Oracle DOMINO
semantics are the only actor privilege.

PPO uses a separate 4,146-value critic observation during training.  It has
the same scene/state layout plus five exact dynamics scalars.  The critic has
its own PointNets, attention blocks, state encoder, and fusion network; no
privileged critic feature pathway is shared with the actor.  The critic is
discarded for deterministic evaluation and later distillation, so the
deployed teacher action remains a function of the 4,141 recoverable values
only.  Simulator-only contact flags remain confined to rewards, termination,
and metrics.

## Safety contract

- **C1 robot--forbidden:** the hand point cloud may contact only points whose
  safe score is at least 0.25.  Any proximal Franka-link/target PhysX contact is
  forbidden independently of the sampled semantic cloud.  The dense margin
  uses the same complete non-safe partition as hard C1 (neutral plus
  protected), and ramps only from 20 mm to the audited 10 mm contact boundary.
  The exact semantic/PhysX predicate remains the violation label.
- **C2 clutter--protected:** a violation requires both protected-region
  clearance below 5 mm and a filtered target/clutter PhysX contact.
- **C3 robot--clutter:** any filtered PhysX contact between any Franka rigid
  body and an active obstacle is a violation. A conservative sampled Franka
  link-centerline/capsule proxy plus the hand cloud supplies a continuous 5 cm
  pre-contact clearance gradient; it never replaces the PhysX violation label.

All physical events use a 0.5 N force threshold.  Episode metrics report C1,
C2, and C3 independently, and decompose C1 into hand-neutral,
hand-protected, and proximal-arm physical events.  Constrained success means
strict pose success with no typed violation during the whole episode.

## Task profiles

| Profile | Task ID | Purpose |
|---|---|---|
| T0 soft | `Isaac-AffordanceTeacher-T0-Soft-Franka-v0` | learn a legal hammer push from scratch without clutter |
| T0 frozen-v7 soft | `Isaac-AffordanceTeacher-T0-FrozenV7-Soft-Franka-v0` | reproduce the accepted forward-v7 learning contract without the later dense C1-clearance term, for controlled manifest-distribution audits |
| T0 frozen-v7 goal-wrench | `Isaac-AffordanceTeacher-T0-FrozenV7-GoalWrench-Soft-Franka-v0` | keep the frozen-v7 environment/reward/action/PPO contract and change only the actor/critic point tokens by adding recoverable goal-translation support and signed-yaw moment relations |
| T0 unified progress | `Isaac-AffordanceTeacher-T0-UnifiedProgress-C1-Soft-Franka-v0` | train waypoint-free reaching and simultaneous XY/Z/SO(3) object motion from signed transition progress |
| T0 unified distance | `Isaac-AffordanceTeacher-T0-UnifiedDistance-C1-Soft-Franka-v0` | controlled ablation replacing only safe-set transition progress with continuous absolute safe-distance cost |
| T0 distance + DAPL goal | `Isaac-AffordanceTeacher-T0-DistanceDAPLGoal-C1-Soft-Franka-v0` | retain continuous reaching and replace only signed object progress with DAPL's current-state coarse/fine pose tracking |
| T0 leaky-signed joint goal | `Isaac-AffordanceTeacher-T0-LeakySignedInitialRelativeJointGoalAction010-C1-Soft-Franka-v0` | interpolate the rejected zero/full wrong-direction slopes with a fixed 0.25 regression slope at the unchanged 0.10 action scale |
| T0 no-C1 diagnostic | `Isaac-AffordanceTeacher-T0-PositiveInitialRelativeJointGoalAction010-NoC1Diagnostic-Franka-v0` | isolate whether C1 learning costs prevent the otherwise identical movement-preserving v10 baseline from learning the strict pose task; never a selectable safe teacher |
| T0 DyWA multi-query no-C1 diagnostic | `Isaac-AffordanceTeacher-T0-DyWAMatchedPotentialsMultiQuery16Action010-NoC1Diagnostic-Franka-v0` | change only the state-dependent attention query count from one to 16 after the matched single-query arm-div control fails |
| T0 DyWA Cartesian no-C1 diagnostic | `Isaac-AffordanceTeacher-T0-DyWAMatchedPotentialsCartesian-NoC1Diagnostic-Franka-v0` | change only the action/control space from seven joint-position residuals to bounded six-dimensional Cartesian delta pose with DLS IK |
| T0 weighted component-progress no-C1 diagnostic | `Isaac-AffordanceTeacher-T0-WeightedComponentProgressAction010-NoC1Diagnostic-Franka-v0` | replace only v13's strict-normalized smooth-max goal scalar by simultaneous signed XY/Z/SO(3) progress, retaining the no-C1 control to test the diagnosed cross-component dead zone |
| T0 goal-side soft | `Isaac-AffordanceTeacher-T0-GoalSide-Soft-Franka-v0` | ablate object-centric safe-side shaping for difficult push directions |
| T0 goal-side C1 soft | `Isaac-AffordanceTeacher-T0-GoalSide-C1-Soft-Franka-v0` | combine endpoint safe-set shaping with the complete soft C1 cost, without termination |
| T0 goal-side C1 explore | `Isaac-AffordanceTeacher-T0-GoalSide-C1-Explore-Soft-Franka-v0` | replace ambiguous nearest-safe shaping with the goal-side set and use bounded 0.10 exploration |
| T0 semantic corridor C1 soft | `Isaac-AffordanceTeacher-T0-SemanticCorridor-C1-Soft-Franka-v0` | optimize a point-cloud free-space contact potential around non-safe target geometry, without exposing route points to the actor |
| T0 semantic log-barrier C1 soft | `Isaac-AffordanceTeacher-T0-SemanticCorridorBarrier-C1-Soft-Franka-v0` | retain corridor geometry but make vanishing free margin dominate unsafe straight-line progress |
| T0 semantic geodesic C1 soft | `Isaac-AffordanceTeacher-T0-SemanticGeodesic-C1-Soft-Franka-v0` | choose the shortest legal direct/support-ring route computed from the live non-safe point cloud; route nodes remain reward-only |
| T0 semantic geodesic conservative | `Isaac-AffordanceTeacher-T0-SemanticGeodesicConservative-C1-Soft-Franka-v0` | preserve scalar geodesic route-quality differences with conservative PPO updates |
| T0 semantic vector field | `Isaac-AffordanceTeacher-T0-SemanticVectorField-C1-Soft-Franka-v0` | reward actual hand displacement along a live point-cloud free-space field only while the direct semantic route is blocked; no field or route is an actor input |
| T0 semantic vector-field explore | `Isaac-AffordanceTeacher-T0-SemanticVectorFieldExplore-C1-Soft-Franka-v0` | remove competing scalar progress, strengthen local flow alignment, and use bounded 0.15 exploration to test lateral-route discovery |
| T0 relation vector field | `Isaac-AffordanceTeacher-T0-RelationVectorField-C1-Soft-Franka-v0` | internally derive point-to-hand, object-local, and goal-conditioned side relations from the unchanged recoverable observation before PointNet/attention |
| T0 relation scratch | `Isaac-AffordanceTeacher-T0-Relation-C1-Soft-Franka-v0` | train the recoverable relation policy jointly from random initialization under the original waypoint-free teacher reward |
| T0 relation vector-field scratch | `Isaac-AffordanceTeacher-T0-RelationVectorFieldScratch-C1-Soft-Franka-v0` | controlled from-scratch comparison that changes only local Euclidean approach shaping to the reward-only semantic free-space field |
| T0 relation balanced-field scratch | `Isaac-AffordanceTeacher-T0-RelationVectorFieldBalancedScratch-C1-Soft-Franka-v0` | preserve the original direct-route progress slope while retaining a stronger reward-only lateral field on semantically obstructed routes |
| T0 relation committed-field scratch | `Isaac-AffordanceTeacher-T0-RelationVectorFieldCommittedScratch-C1-Soft-Franka-v0` | rejected diagnostic: latch detour-strength reward guidance through the final direct edge, introducing history not recoverable from the actor observation |
| T0 relation clearance-blend scratch | `Isaac-AffordanceTeacher-T0-RelationVectorFieldClearanceBlendScratch-C1-Soft-Franka-v0` | continuously blend detour/direct guidance from the current semantic-route clearance, eliminating the reward cliff without a waypoint or hidden phase |
| T0 relation clearance-recovery scratch | `Isaac-AffordanceTeacher-T0-RelationVectorFieldClearanceRecoveryScratch-C1-Soft-Franka-v0` | retain the continuous blend and replace an illegal straight-line fallback with a current-geometry outward-plus-boundary-tangent recovery field until a legal semantic route exists again |
| T0 relation full-safe contact-gate scratch | `Isaac-AffordanceTeacher-T0-RelationFullSafeContactGateScratch-C1-Soft-Franka-v0` | route continuously to any C1-legal safe handle point, gate navigation after legal contact, and leave goal-conditioned contact choice to the recoverable policy and joint pose progress |
| T0 relation full-safe joint-pose-cost scratch | `Isaac-AffordanceTeacher-T0-RelationFullSafeJointPoseCostScratch-C1-Soft-Franka-v0` | retain full-safe reaching and add a bounded simultaneous XY/Z/SO(3) current-state cost so a wrong one-sign rotation mode remains continuously penalized |
| T0 relation full-safe post-contact pose-cost scratch | `Isaac-AffordanceTeacher-T0-RelationFullSafePostContactPoseCostScratch-C1-Soft-Franka-v0` | preserve full-safe reaching, retire the pre-contact vector field after first legal contact, and activate the simultaneous XY/Z/SO(3) state cost only in the post-contact portion of training |
| T0 relation full-safe post-contact improvement scratch | `Isaac-AffordanceTeacher-T0-RelationFullSafePostContactImprovementScratch-C1-Soft-Franka-v0` | use first legal contact as a zero-reward pose baseline, then reward joint XY/Z/SO(3) improvement without imposing a negative cost cliff at contact |
| T0 relation yaw-compatible post-contact improvement scratch | `Isaac-AffordanceTeacher-T0-RelationYawCompatiblePostContactImprovementScratch-C1-Soft-Franka-v0` | while observable yaw error remains material, route to the near-best safe-handle subset for signed moment arm; retain contact-relative joint pose improvement and no waypoint/hidden phase |
| T0 relation yaw-positive post-contact improvement scratch | `Isaac-AffordanceTeacher-T0-RelationYawPositivePostContactImprovementScratch-C1-Soft-Franka-v0` | retain the full safe-handle halfspace above a positive signed-moment floor, preserving reachability diversity while excluding wrong-sign yaw contacts |
| Relation C1 eval | `Isaac-AffordanceTeacher-Relation-C1-Franka-v0` | instantiate the relation policy in the identical hard-C1 audit environment so relation checkpoints cannot be evaluated with the legacy network by mistake |
| Relation C1 soft refine | `Isaac-AffordanceTeacher-Relation-C1-Soft-Franka-v0` | low-noise soft-C1 continuation preserving the selected relation architecture |
| Relation C2/C3/clutter soft | `Isaac-AffordanceTeacher-Relation-{C2,C3,Clutter}-Soft-Franka-v0` | typed soft-cost transfer tasks that retain the C1 relation checkpoint structure |
| Relation C2/C3/combined hard | `Isaac-AffordanceTeacher-Relation-{C2,C3,Combined}-Franka-v0` | matched-architecture deterministic audits and final hard-constraint refinement |
| C1 soft refine | `Isaac-AffordanceTeacher-C1-Soft-Franka-v0` | behavior-preserving hard-C1-aligned clearance/contact adaptation without termination |
| C1 | `Isaac-AffordanceTeacher-C1-Franka-v0` | hard forbidden-region audit without clutter |
| C2 soft | `Isaac-AffordanceTeacher-C2-Soft-Franka-v0` | one-blocker protected-part transfer with soft C1+C2 costs and no C3 objective |
| C3 soft | `Isaac-AffordanceTeacher-C3-Soft-Franka-v0` | one-blocker whole-arm routing transfer with soft C1+C3 costs and no C2 objective |
| clutter soft | `Isaac-AffordanceTeacher-Clutter-Soft-Franka-v0` | adapt to two obstacle point clouds with soft C1/C2/C3 costs |
| C2 | `Isaac-AffordanceTeacher-C2-Franka-v0` | one-blocker protected-part sweep audit |
| C3 | `Isaac-AffordanceTeacher-C3-Franka-v0` | one-blocker whole-arm routing audit |
| combined | `Isaac-AffordanceTeacher-Combined-Franka-v0` | two-blocker hard C1+C2+C3 benchmark |

The policy uses direct relative joint-position actions and dense,
waypoint-free safe-region approach/pose-progress shaping.
During soft exploration, signed pose progress is available whenever the hand
reaches the safe patch, while any simultaneous forbidden contact is penalized
separately. The one-time contact bonus requires a fully legal contact, strict
success is violation-free, and hard profiles terminate the mixed contact. This
preserves the contact-rich learning path without weakening the final contract.
Hard profiles use the same 4,141-value actor and network as T0/clutter-soft,
but cap resumed exploration noise at 0.005 so hard terminations do not erase
the learned contact skill.  Their PPO update is also deliberately conservative
(`1e-5` learning rate, `0.1` clipping, no entropy bonus, and checkpoints every
two iterations) because the earlier aggressive hard continuation drifted
toward a no-contact/no-push policy.  Deterministic evaluation is unaffected by
these training-only settings.

## Training and acceptance gates

1. Train T0 soft from scratch with 1,024 environments.  Gate: at least 90%
   strict pose success and at most 2% C1 violation on held-out scenes.
2. Evaluate the same checkpoint on C1 hard.  If the C1 gate is missed, use the
   C1-soft refine profile first and select dense five-iteration checkpoints on
   disjoint constrained evaluation; only then run a short hard fine-tune.
   Gate: at least 85% constrained success and at most 1% C1.  For the audited
   full-direction set, additionally require at least 12/16 = 75% constrained
   success and zero C1 in the `[70, 90]` positive endpoint bin so the aggregate
   cannot hide the known protected-head dead point.
3. Transfer to clutter soft.  Gate: at least 80% strict success and declining
   C2/C3 rates on direction-balanced validation scenes.
4. Fine-tune and evaluate combined hard for seeds 17, 23, and 41.  Gate:
   mean constrained success at least 70%, every seed at least 60%, and each of
   C1/C2/C3 at most 2%.

These are research gates, not claims; the reported numbers must come from
fresh multi-episode evaluation JSON files.

Quantitative evaluation uses a balanced-per-environment protocol.  With 128
held-out scenes, run 128 environments and count the first terminal episode
from each environment exactly once.  For larger episode counts, every
environment receives a fixed quota differing by at most one.  Do not stop as
soon as the aggregate terminal count is reached without this quota: easy
scenes terminate, reset, and get counted repeatedly while difficult scenes
are still running, which creates a severe optimistic bias.  `scripts/eval.py`
records `episode_allocation`, `episode_quota_min`, and `episode_quota_max` in
every result JSON so this protocol can be audited.

## Current audited evidence (2026-08-25)

- The unified-progress T0 ablation is the minimal follow-up to the rejected
  contact-gated DAPL-progress runs.  It keeps the same DAPL/DyWA scene and PPO
  settings, 4,141/4,146 actor/critic contracts, strict XY/Z/SO(3)+dwell
  success, no clutter, and soft C1 costs.  Its only reward-contract change is
  to replace the instantaneous legal-contact-gated pose progress and one-time
  contact bonus with the existing always-observable signed joint-pose progress.
  Thus the active learning signals are safe-set distance progress, joint-pose
  progress, local forbidden-contact/clearance costs, and sparse success; no
  waypoint, phase latch, desired contact side, or absolute pose cost is used.
  The real four-environment PPO smoke passes at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_17-48-38_seed17_t0_unified_progress_c1_config_smoke_v3`.
  Three 1,024-environment runs now train from scratch: seed 17
  (`https://wandb.ai/simonlsx/non-prehensile-affordance/runs/eo71i4oi`), seed
  23 (`https://wandb.ai/simonlsx/non-prehensile-affordance/runs/0dn3fdpp`),
  and seed 41
  (`https://wandb.ai/simonlsx/non-prehensile-affordance/runs/j4375vbz`).
  These three runs are rejected at model 250/300/300 after a manifest audit
  exposes an infeasible target contract rather than a reward failure.  In the
  16,384-task training manifest, only 11.8% of initial-to-goal relative
  rotations are within one degree of the table normal, 76.6% change support
  height, and 61.3% change it by more than the strict 1 cm threshold.  The
  forward teacher's accepted evaluation set is 100% same-support with zero
  height change.  The next single-variable experiment keeps the unified reward
  and actor unchanged and regenerates DAPL-range tasks with the explicitly
  named `dapl-planar-push` same-support contract.
  The resulting train/eval manifests contain 1,024/128 scenes and
  16,384/2,048 unique tasks at
  `data/manifests/domino_hammer_dapl_planarpush_train1024_seed1701.jsonl` and
  `data/manifests/domino_hammer_dapl_planarpush_eval128_seed1801.jsonl`.
  Both preserve nine randomized hammer support signatures, have exactly zero
  initial-to-goal height change, 100% table-normal relative rotation axes, and
  retain the DAPL minimum 15 cm XY displacement (train/eval mean 29.64/29.25
  cm).  Thus this is a support-manifold correction, not the old 6--10 cm
  directional proof distribution.

  The corrected same-support v4 seed-17 run is
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_19-24-27_seed17_t0_unified_progress_planarpush_fromscratch_v4`.
  Identical deterministic 128-scene evaluations at model 50/100/200 all have
  0% strict and constrained success.  Mean terminal XY error is
  29.53/29.56/28.91 cm, while legal safe-contact episode rate is
  4.69/7.81/3.91%.  Model 200 also has 8.59% C1, 1.56 cm mean Z error, and
  1.82 rad mean full-rotation error.  Its small 6.6 mm XY improvement from
  model 100 is not a pushing skill, and the deterministic policy still stays
  about 10 cm from the target at its closest point.  The run and queued later
  seeds were therefore stopped at the preregistered model-200 decision gate.
  Evaluation artifacts are under
  `outputs/teacher_eval/seed17_unified_progress_planarpush_v4_model{50,100,200}_balanced128`.

  v5 changes one variable in response: `safe_region_progress` (weight +8) is
  replaced by `safe_region_distance` (weight -2, 0.50 m normalization).  This
  produces an absolute linear approach cost with a far-field slope and zero
  cost continuously at the contact boundary.  Sparse success, unified
  XY/Z/SO(3) progress, both soft C1 terms, actor/critic observations, PPO,
  manifests, and strict termination are unchanged.  It adds no waypoint,
  contact latch, event reward, or hidden phase.  A real four-environment PPO
  smoke confirms the 4,141/4,146 observation contract and exactly five active
  rewards at
  `logs/rsl_rl/franka_affordance_teacher_seed17005/2026-08-25_19-59-11_seed17005_t0_unified_distance_planarpush_smoke_v5`.
  The controlled 1,024-environment seed-17 run is training from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_20-12-42_seed17_t0_unified_distance_planarpush_fromscratch_v5_r1`;
  its online W&B run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/je3q9ymx`.
  Seeds 23 and 41 are deliberately not queued until this run demonstrates
  deterministic reaching and pushing on the disjoint 128-scene evaluation.

  v5 is rejected at model 200.  Models 50/100/200 all score 0/128 strict and
  constrained success; legal safe-contact peaks at 13.28% at model 50 and
  falls to 6.25%/2.34%, while C1 is 24.22%/14.84%/11.72%.  Mean terminal XY
  is 29.53/28.89/29.26 cm, so the hammer does not acquire sustained goal
  progress.  The model-50 video shows the deterministic hand approaching and
  parking near the green safe handle while the hammer remains separated from
  its cyan goal ghost:
  `outputs/teacher_demos/seed17_v5_model50_scene0020_legal/seed17_v5_model50_scene0020_legal-step-0.mp4`.
  Thus absolute safe-distance fixes the v4 mean-reaching failure, but signed
  one-step object-pose progress does not provide persistent pushing credit.
  Evidence is under
  `outputs/teacher_eval/seed17_unified_distance_planarpush_v5_model{50,100,200}_balanced128`.

  v6 changes only the object-motion shaping family: the signed joint-pose
  progress term is replaced by DAPL's coarse/fine current-state pose kernels
  (`std=0.6/0.3`, weights `5/16`) behind the original observable
  `d_safe < 0.10 m` gate.  v5's absolute safe-distance cost remains the sole
  approach objective; DAPL's positive proximity living reward is not restored.
  A four-environment PPO smoke verifies the unchanged 4,141/4,146 observation
  contract and the intended six-term reward table at
  `logs/rsl_rl/franka_affordance_teacher_seed17006/2026-08-25_20-50-42_seed17006_t0_distance_dapl_goal_planarpush_smoke_v6`.
  The 1,024-environment seed-17 run is
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_20-55-58_seed17_t0_distance_dapl_goal_planarpush_fromscratch_v6`
  with online W&B run
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/vj81mpj2`.

  v6 is rejected at model 200.  Models 50/100/200 all score 0/128 strict and
  constrained success.  Their mean terminal XY errors are 30.56/29.39/29.52
  cm, legal safe-contact rates are 15.63%/16.41%/17.19%, and C1 episode
  violation rates are 21.09%/18.75%/21.09%.  A deterministic trace confirms
  that the hammer stays essentially stationary while the hand remains near
  the safe handle and receives the uncentered positive goal score.  The
  roughly +14 training return is therefore a living-reward exploit, not
  learned pushing.  Evidence is under
  `outputs/teacher_eval/seed17_distance_dapl_goal_planarpush_v6_model{50,100,200}_balanced128`.

  v7 makes one correction to that diagnosed failure: it uses the identical
  weighted DAPL full-pose score but subtracts its episode-initial value.  A
  stationary hammer earns exactly zero, a pose improvement earns persistent
  positive credit, and regression earns persistent negative credit.  The
  reference is an episode-constant scalar rather than a waypoint, phase,
  latch, action branch, or actor input.  Reaching, C1 terms, observations,
  PPO, manifests, and success criteria remain unchanged.  The pure metric
  tests pass (37/37), and a real four-environment PPO smoke confirms the
  unchanged 4,141/4,146 observation contract and exactly five active rewards
  at
  `logs/rsl_rl/franka_affordance_teacher_seed17007/2026-08-25_21-34-30_seed17007_t0_initial_relative_dapl_goal_planarpush_smoke_v7`.
  The 1,024-environment seed-17 run is training from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_21-39-54_seed17_t0_initial_relative_dapl_goal_planarpush_fromscratch_v7`;
  its online W&B run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/my4lrg77`.
  An independent GPU-7 watcher evaluates models 50/100/200/300/400/500/700/1000
  on all 128 held-out scenes.  Seeds 23 and 41 remain gated on that seed-17
  evidence.

  The first v7 decision point, model 50, has 0/128 strict and constrained
  successes, 15.62% legal safe-contact episodes, and 18.75% C1 violations.
  Mean terminal XY/Z/SO(3) errors are 30.04 cm / 1.85 cm / 1.89 rad.  Its
  deterministic diagnostic scene reduces hand-to-target distance only to
  18.19 cm and moves the hammer by less than 0.12 mm, so model 50 is still a
  reaching checkpoint rather than evidence of pushing.  Unlike v6, the
  relative pose term remains approximately zero for a stationary target and
  becomes negative when exploratory contacts worsen the pose.  Training and
  the pre-registered model-100 evaluation therefore continue without a reward
  change.  Artifact:
  `outputs/teacher_eval/seed17_initial_relative_dapl_goal_planarpush_v7_model50_balanced128/eval_summary.json`.

  Model 100 also has 0/128 strict and constrained successes.  Relative to
  model 50, its diagnostic hand-to-target distance improves from 18.19 cm to
  12.85 cm, but target translation remains below 0.12 mm.  Across all held-out
  scenes it has 7.81% legal safe-contact episodes, 14.84% C1 violations, and
  mean terminal XY/Z/SO(3) errors of 29.43 cm / 1.33 cm / 1.82 rad.  Thus the
  controlled change has not yet produced pushing, but deterministic reaching
  is still improving; the run continues to the pre-registered model-200
  decision gate without another change.  Artifact:
  `outputs/teacher_eval/seed17_initial_relative_dapl_goal_planarpush_v7_model100_balanced128/eval_summary.json`.

  A rendered model-100 legal-contact probe adds an important mechanical
  diagnosis: the hand reaches the green safe handle without C1, begins moving
  the hammer around step 240, and produces 7.22 cm target translation by step
  299.  However, planar goal error worsens from about 35.05 cm to 42.35 cm.
  Thus the policy can reach and transfer force through the allowed region, but
  has not learned the goal-conditioned push direction.  This is not an asset,
  collision, or immovable-object failure, and the signed relative reward gives
  the wrong-direction motion negative credit.  Transparent goal and semantic
  overlay video plus sidecar:
  `outputs/teacher_demos/seed17_v7_model100_scene0040_legal/`.

  A supplemental model-150 balanced evaluation remains at 0/128, with 8.59%
  legal safe-contact episodes and 14.06% C1 violations.  Its diagnostic
  hand-to-target minimum regresses to 15.62 cm and the target is stationary;
  mean terminal XY/Z/SO(3) errors are 30.09 cm / 1.24 cm / 1.79 rad.  This
  removes the apparent model-50-to-100 reaching improvement as evidence of
  monotonic policy learning, but the run is retained through the registered
  model-200 decision point.  Artifact:
  `outputs/teacher_eval/seed17_initial_relative_dapl_goal_planarpush_v7_model150_balanced128/eval_summary.json`.

  v7 is rejected at its model-200 gate.  Model 200 remains at 0/128 strict and
  constrained success, with 5.47% legal safe-contact episodes, 10.16% C1
  violations, and mean terminal XY/Z/SO(3) errors of 29.03 cm / 1.14 cm /
  1.85 rad.  Its diagnostic hand reaches 7.88 cm from the target but does not
  move it.  Across iterations 0/25/50/75/100/125/150, training safe distance
  improves from 35.15 cm to roughly 13--15 cm while the relative pose reward
  is consistently non-positive and strict success stays zero.  Combined with
  the wrong-direction legal push in the model-100 video, this shows a specific
  avoidance failure: most exploratory pushes regress pose, the signed term
  penalizes them, and the easiest policy response is to stop making contact.
  The run and later-checkpoint watcher were stopped after preserving model 200
  and all held-out evidence.  Artifact:
  `outputs/teacher_eval/seed17_initial_relative_dapl_goal_planarpush_v7_model200_balanced128/eval_summary.json`.

  v8 changes only that diagnosed sign treatment.  It applies the positive
  part to the same episode-initial-relative, multiscale DAPL score: stationary
  and regressed poses earn zero from the goal term, while improvements retain
  persistent positive credit.  The five-term reward table, DAPL score and
  weights, safe-distance gate, observations, PPO, manifests, C1 coefficients,
  and strict success predicate are unchanged.  This removes the incentive to
  avoid all exploratory object contact without adding a reward, waypoint,
  phase, latch, or privileged actor feature.

  The pure metric suite passes 37/37 and a real four-environment PPO smoke
  confirms `positive_only: true`, the unchanged 4,141/4,146 observation
  contract, and exactly five active rewards at
  `logs/rsl_rl/franka_affordance_teacher_seed17008/2026-08-25_22-21-23_seed17008_t0_positive_initial_relative_dapl_goal_planarpush_smoke_v8`.
  The seed-17, 1,024-environment v8 run is training from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_22-26-33_seed17_t0_positive_initial_relative_dapl_goal_planarpush_fromscratch_v8`;
  its online W&B run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/8aicq6vq`.
  An independent GPU-7 watcher evaluated models 50/100/150/200 on the
  identical 128-scene held-out manifest and was stopped at the rejection gate.

  v8 is rejected at model 200.  Its held-out models 50/100/150/200 all obtain
  0/128 strict and constrained successes.  Model 200 has 6.25% legal
  safe-contact episodes, 14.84% C1 violations, and mean terminal XY/Z/SO(3)
  errors of 30.04 cm / 1.40 cm / 1.83 rad.  Across the 128 scenes, terminal
  XY error worsens by 1.54 cm on average; only six scenes improve XY by more
  than 2 cm, while 28 regress by more than 2 cm.  The model-100 scene-73 video
  confirms that a nominal 3.5-mm batch improvement is contact jitter rather
  than a push: the single-scene replay reaches 9.66 cm from the target and
  moves the hammer only 0.25 mm.  Artifact:
  `outputs/teacher_demos/seed17_v8_model100_scene0073_legal/seed17_v8_model100_scene0073_legal_t0_positive_initial_relative_dapl_goal_planarpush-step-0.mp4`.

  The controlled diagnosis is score geometry, not missing reach or force
  transfer.  DAPL's published scalar uses `position + rotation / 5`; held-out
  v8 trajectories include cases where yaw improvement compensates for XY
  regression and still raises that score, whereas the benchmark accepts a
  pose only when XY, Z, and SO(3) all pass their own thresholds.  v9 therefore
  changes only the geometry inside the same reset-relative, positive-only,
  safe-distance-gated goal term.  It uses the normalized joint XY/Z/SO(3)
  smooth maximum already used by the strict metrics, so the worst condition
  remains the optimization bottleneck.  The other four rewards, all weights,
  actor/critic observations, PPO, manifests, C1 coefficients, and terminal
  predicate remain unchanged.

  Pure metric and manifest tests pass 49/49 plus three subtests.  A real
  four-environment smoke confirms the unchanged 4,141/4,146 observation
  contract and exactly five active rewards at
  `logs/rsl_rl/franka_affordance_teacher_seed17009/2026-08-25_23-21-26_seed17009_t0_positive_initial_relative_joint_goal_planarpush_smoke_sentinel_v9`.
  Its saved `env.yaml` records the joint-error selector and models 0/1 were
  produced successfully.  The formal seed-17, 1,024-environment v9 run is
  training from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-25_23-27-40_seed17_t0_positive_initial_relative_joint_goal_planarpush_fromscratch_v9`
  under tmux session `aff_teacher_planarpush_v9_joint_s17_20260825`.  Its
  online W&B run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/wr197w1u`.
  An independent watcher under
  `aff_teacher_planarpush_v9_joint_s17_eval_20260825` evaluates models
  50/100/150/200/300/400/500/700/1000 on the same 128-scene held-out manifest.

  The first formal audit at `model_50` remains an early failure signal, not a
  selection candidate: strict/constrained success is 0/128, legal safe contact
  is 7.03%, and C1 is 14.84%.  Mean terminal XY/Z/SO(3) error is
  29.71 cm / 1.71 cm / 1.92 rad.  Relative to each manifest initial pose, mean
  XY error worsens by 1.22 cm and mean SO(3) error worsens by 0.24 rad; 9/128
  scenes improve XY by more than 2 cm, 34/128 regress by more than 2 cm, and
  none simultaneously improves XY by more than 2 cm and SO(3) by more than
  0.1 rad.  The scene-0 diagnostic still shows only 0.12 mm target
  translation.  This is recorded without changing the experiment: the frozen
  watcher will next audit models 100/150/200 before the v9 keep/reject gate.
  Artifact:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_planarpush_v9_model50_balanced128/eval_summary.json`.

  `model_100` does not show an emerging push trend: strict/constrained success
  remains 0/128, legal safe contact remains 7.03%, and C1 rises to 21.88%.
  Mean terminal XY/Z/SO(3) is 29.55 cm / 1.72 cm / 1.91 rad.  Relative to the
  manifest initial poses, XY and SO(3) still regress by 1.06 cm and 0.23 rad
  on average; 9/128 improve XY by more than 2 cm, 34/128 regress by more than
  2 cm, and again no scene simultaneously improves XY by more than 2 cm and
  SO(3) by more than 0.1 rad.  Scene 0 remains stationary at the 0.12-mm
  settling scale.  The frozen experiment therefore proceeds to its registered
  model-150/model-200 audits without a reward or configuration change.
  Artifact:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_planarpush_v9_model100_balanced128/eval_summary.json`.

  `model_150` shows a small but still non-credible trend: success remains
  0/128, legal safe contact rises to 9.38%, and C1 is 17.19%.  Mean terminal
  XY/Z/SO(3) is 29.09 cm / 1.52 cm / 1.84 rad.  Mean relative regression
  shrinks to 0.59 cm XY and 0.16 rad SO(3); 13/128 scenes improve XY by more
  than 2 cm, 24/128 regress by more than 2 cm, and only 2/128 simultaneously
  improve XY by more than 2 cm and SO(3) by more than 0.1 rad.  The fixed
  scene-0 trace remains stationary at 0.12-mm settling scale.  Because the
  batch distribution improved relative to models 50/100, v9 is retained to
  the preregistered model-200 decision point and the two joint-improvement
  scenes are reserved for visual/trajectory validation.
  Artifact:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_planarpush_v9_model150_balanced128/eval_summary.json`.

  v9 is rejected and stopped at its preregistered `model_200` gate.  It still
  obtains 0/128 strict and constrained success, 5.47% legal safe contact, and
  15.62% C1.  Mean terminal XY/Z/SO(3) is 28.70 cm / 1.38 cm / 1.86 rad;
  there is no held-out scene that simultaneously improves XY by more than
  2 cm and SO(3) by more than 0.1 rad.  An enhanced, metric-equivalent replay
  records the actual runtime task index, initial error and maximum object
  motion for every environment.  It confirms all 128 episodes used task 0 and
  the actual mean initial XY error is 28.49 cm.  The hammer moves more than
  2 cm in 39/128 episodes, but only 6/128 combine that motion with any legal
  safe contact.  Thus v9 is not purely stationary: it learns uncontrolled
  object motion without a safe, goal-conditioned pushing skill.  Artifacts:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_planarpush_v9_model200_balanced128/eval_summary.json` and
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_planarpush_v9_model200_motionaudit_balanced128/eval_summary.json`.

  The next controlled experiment, v10, changes no reward or policy contract.
  The source-of-truth task settings declare a 0.10 residual-action scale, but
  the formal v9 `env.yaml` records 0.03 inherited from the tightly initialized
  6--10-cm proof task.  v10 restores only `arm_action.scale=0.10`; the five
  rewards, joint score, 0.40 PPO exploration noise, 4,141/4,146 observations,
  broad joint reset, manifests, C1 costs, and strict terminal predicate remain
  unchanged.  A real four-environment, two-update smoke writes models 0/1 and
  its saved configuration confirms `scale: 0.1`, `coarse_weight: -1.0`,
  `positive_only: true`, and the unchanged observation dimensions at
  `logs/rsl_rl/franka_affordance_teacher_seed17010/2026-08-26_00-10-14_seed17010_t0_positive_initial_relative_joint_goal_action010_planarpush_smoke_v10`.
  The formal seed-17, 1,024-environment run trains from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_00-16-15_seed17_t0_positive_initial_relative_joint_goal_action010_planarpush_fromscratch_v10`;
  online W&B is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/8otszloy`.
  The independent GPU-7 watcher evaluates models
  50/100/150/200/300/400/500/700/1000 with the enhanced per-scene task,
  initial-error, initial-joint and maximum-target-motion evidence on the same
  held-out manifest.

  The first preregistered held-out checkpoint, `model_50`, is not yet a
  goal-conditioned policy: strict/constrained success is 0/128, legal safe
  contact is 15/128 (11.72%), and C1 is 22/128 (17.19%).  It moves the hammer
  by more than 2 cm in 49/128 scenes, but mean terminal XY and SO(3) errors are
  respectively 1.18 cm and 0.146 rad worse than their actual reset values;
  only 1/128 scenes improves both XY by more than 2 cm and SO(3) by more than
  0.1 rad.  A matched 128-scene zero-action replay records zero contact and
  zero violations, mean maximum target translation 5.0 mm, and only 3/128
  scenes above 2 cm.  Thus the extra v10 motion is predominantly induced by
  the policy rather than reset drift, while three zero-action outliers remain
  flagged for later manifest inspection.  Artifacts:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_action010_planarpush_v10_model50_balanced128/eval_summary.json` and
  `outputs/teacher_diagnostics/dapl_planarpush_eval128_zeroaction_evalpath_seed1801_v10/eval_summary.json`.
  This is an early diagnostic, not a v10 accept/reject decision; the frozen
  single-variable run remains unchanged through the model-100/150/200 gate.

  `model_100` also has 0/128 strict and constrained success.  Legal safe
  contact remains 15/128 (11.72%) while C1 rises to 24/128 (18.75%).  The
  policy moves the hammer by more than 2 cm in 59/128 scenes, but terminal
  XY/Z/SO(3) errors regress from their actual reset values by 1.27 cm /
  1.94 cm / 0.289 rad on average; only 1/128 scenes improves both XY by more
  than 2 cm and SO(3) by more than 0.1 rad.  Reflecting the relative-goal yaw
  changes the deterministic action by mean L2 0.321/0.286 for negative/
  positive yaw samples, so the actor is goal-sensitive rather than ignoring
  the goal observation.  A one-environment legal, C1-free replay of scene 49
  transfers 11.1 cm through the safe handle and improves XY from 50.6 cm to
  46.0 cm, but raises the hammer 6.4 cm off the support manifold and worsens
  SO(3) from 1.56 to 2.01 rad.  This directly confirms uncontrolled force
  transfer rather than failure to reach or move the object.  Quantitative and
  rendered artifacts:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_action010_planarpush_v10_model100_balanced128/eval_summary.json` and
  `outputs/teacher_diagnostics/seed17_v10_model100_scene0049_legal_wrongpush/`.
  The frozen experiment continues to model 150/200; any follow-up must be
  justified against DAPL's published proximity plus coarse/fine pose reward,
  rather than adding a waypoint or hidden phase.

  `model_150` still has 0/128 strict/constrained success, 12/128 (9.38%) legal
  safe contact, and 32/128 (25.00%) C1.  Movement above 2 cm rises to 65/128,
  but terminal XY/Z/SO(3) errors regress by 0.91 cm / 2.25 cm / 0.202 rad on
  average and only 2/128 scenes jointly improve XY and SO(3) by the diagnostic
  margins.  Recomputing the exact joint reward geometry shows that every one
  of the 40 scenes with positive reset-relative credit nevertheless regresses
  at least one pose component.  This is expected from a smooth maximum but is
  harmful when its positive-only form maps the other 88 wrong-direction
  trajectories to the same zero as a stationary object.  The model-200 gate
  remains the frozen v10 decision point.  Artifact:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_action010_planarpush_v10_model150_balanced128/eval_summary.json`.

  The preregistered v11 candidate is a single-variable follow-up:
  keep v10's 0.10 action scale and entire five-term contract, but make the
  normalized reset-relative joint error signed and bounded in `[-1, 1]` so
  regressions receive negative credit.  It does not add a term, phase,
  waypoint, contact side, or actor feature.  Its initial pure metric suite
  passed 39/39, and no v11 training started before the v10 model-200 decision.

  v10 is rejected and stopped at that registered decision point.  `model_200`
  remains at 0/128 strict/constrained success, with 10/128 (7.81%) legal safe
  contact and 26/128 (20.31%) C1.  It moves the hammer by more than 2 cm in
  69/128 scenes, but mean terminal XY/Z/SO(3) errors regress from reset by
  1.12 cm / 2.31 cm / 0.270 rad, and only 1/128 jointly improves XY and SO(3)
  by the diagnostic margins.  The exact terminal joint geometry assigns
  positive-only credit to 36 scenes (all with at least one regressed pose
  component), while a signed version averages -0.135 over all scenes.  This
  closes the v10 action-scale ablation: 0.10 restores force transfer but does
  not repair the missing wrong-direction feedback.  Artifact:
  `outputs/teacher_eval/seed17_positive_initial_relative_joint_goal_action010_planarpush_v10_model200_balanced128/eval_summary.json`.

  The v11 signed candidate then passes an actual four-environment, two-update
  PPO smoke.  It writes `model_0.pt` and `model_1.pt`; the saved configuration
  confirms `scale: 0.1`, `coarse_weight: -1.0`, `positive_only: false`, exactly
  five rewards, actor/critic dimensions 4,141/4,146, and unchanged 0.40-capped
  exploration noise.  The expanded pure metric/representation/actor suite
  passes 47/47.  Smoke artifact:
  `logs/rsl_rl/franka_affordance_teacher_seed17011/2026-08-26_00-54-57_seed17011_t0_signed_initial_relative_joint_goal_action010_planarpush_smoke_v11`.
  The formal seed-17, 1,024-environment run now trains from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_01-01-03_seed17_t0_signed_initial_relative_joint_goal_action010_planarpush_fromscratch_v11`;
  online W&B is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/t7nfuumz`.
  Its GPU-7 watcher evaluates models 50/100/150/200/300/400/500/700/1000 on
  the unchanged held-out 128-scene manifest with the same motion-aware audit.

  The first fixed held-out audit at `model_50` remains an early rejection
  signal, not a selectable checkpoint: strict/constrained success is 0/128,
  legal safe contact is 28/128 (21.88%), and C1 is 33/128 (25.78%).  The
  policy moves the hammer by more than 2 cm in 77/128 scenes, including 16
  legal-contact and C1-free scenes, so signed feedback has not collapsed to a
  stationary policy.  However, mean terminal XY/Z/SO(3) errors regress from
  reset by 3.17 cm / 2.20 cm / 0.233 rad; only 8 scenes improve XY by more
  than 2 cm, only 10 improve SO(3) by more than 0.1 rad, and none does both.
  Compared with v10 `model_50`, contact and force transfer appear earlier but
  are less controlled.  The frozen run continues through model 100/150/200
  before the signed single-variable hypothesis is accepted or rejected.
  Artifact:
  `outputs/teacher_eval/seed17_signed_initial_relative_joint_goal_action010_planarpush_v11_model50_balanced128/eval_summary.json`.

  At `model_100`, strict/constrained success remains 0/128.  Legal safe
  contact is 15/128 (11.72%) and C1 falls to 11/128 (8.59%), while movement
  above 2 cm falls to 51/128 and legal-contact/C1-free movement to 8/128.
  Mean terminal XY/Z/SO(3) still regress from reset by 1.26 cm / 1.75 cm /
  0.225 rad; 15 scenes improve XY by more than 2 cm, only two improve SO(3)
  by more than 0.1 rad, and only one does both.  Relative to `model_50`, the
  policy is becoming less unsafe and less indiscriminately forceful, but this
  is not yet evidence of goal-conditioned convergence.  The experiment stays
  frozen through the model-150/model-200 decision gate.  Artifact:
  `outputs/teacher_eval/seed17_signed_initial_relative_joint_goal_action010_planarpush_v11_model100_balanced128/eval_summary.json`.

  `model_150` remains at 0/128 strict/constrained success, with 11/128
  (8.59%) legal safe contact and 15/128 (11.72%) C1.  Mean XY/Z/SO(3)
  regression is 1.33 cm / 1.63 cm / 0.195 rad; only 9/128 improve XY by more
  than 2 cm, 4/128 improve SO(3) by more than 0.1 rad, and none does both.
  Thus the safety recovery at model 100 does not become goal-conditioned
  pushing.  Artifact:
  `outputs/teacher_eval/seed17_signed_initial_relative_joint_goal_action010_planarpush_v11_model150_balanced128/eval_summary.json`.

  v11 is rejected and stopped at the registered `model_200` decision gate.
  It still obtains 0/128 strict/constrained success, 10/128 (7.81%) legal
  safe contact, and 12/128 (9.38%) C1.  Mean terminal XY/Z/SO(3) regress from
  reset by 0.74 cm / 1.28 cm / 0.152 rad; movement above 2 cm declines from
  77 scenes at model 50 to 38, only six scenes improve XY by more than 2 cm,
  only one improves SO(3) by more than 0.1 rad, and none does both.  The
  full-slope signed penalty therefore suppresses the uncontrolled v10 motion
  by converging toward less contact rather than finding a correct push.
  Artifact:
  `outputs/teacher_eval/seed17_signed_initial_relative_joint_goal_action010_planarpush_v11_model200_balanced128/eval_summary.json`.

  A rendered model-100 replay confirms the mechanism rather than a simple
  reach failure: the hand reaches the green handle, rapidly tips/rotates the
  hammer, and sends it away from the translucent goal.  The replay uses the
  same manifest scene but a one-environment robot-reset stream, so it is a
  qualitative failure example rather than a bit-identical replay of batched
  row 43.  Video and sidecar:
  `outputs/teacher_diagnostics/seed17_v11_model100_scene0043_legal_xyimprove_rotregress/`.

  The next controlled candidate interpolates the two falsified endpoints
  without adding a reward or state: v10 gives regressions zero slope and
  preserves exploration but produces uncontrolled pushes; v11 gives them full
  negative slope and collapses toward no push.  A leaky signed score keeps the
  identical positive-improvement branch and applies a fixed 0.25 slope only
  to regressions.  Action scale, score geometry, reward count and weights,
  observations, PPO, manifests, C1 terms, and strict terminal predicate stay
  unchanged.

  The implementation's metric/representation/actor suite passes 59/59.  A
  real four-environment, two-update Isaac/PPO smoke writes models 0/1; its
  saved configuration verifies `scale: 0.1`, `coarse_weight: -1.0`,
  `positive_only: false`, `regression_scale: 0.25`, exactly five rewards, and
  the same `ActorCriticAffordance` policy.  Smoke artifact:
  `logs/rsl_rl/franka_affordance_teacher_seed17012/2026-08-26_01-38-31_seed17012_t0_leaky_signed_initial_relative_joint_goal_action010_planarpush_smoke_v12`.

  The formal seed-17, 1,024-environment v12 run trains from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_01-45-59_seed17_t0_leaky_signed_initial_relative_joint_goal_action010_planarpush_fromscratch_v12`;
  online W&B is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/msb5qzu4`.
  Its independent GPU-7 watcher uses the unchanged deterministic held-out 128
  scenes and the same checkpoint grid.  Because `/data1` had only about 2.9
  GiB free at launch, a separate guard sends a graceful interrupt only to
  this trainer below 2 GiB; no prior artifact or unrelated user data is
  deleted.

  The first held-out `model_50` remains 0/128 strict/constrained success, with
  20/128 (15.62%) legal safe contact and 26/128 (20.31%) C1.  It moves the
  hammer by more than 2 cm in 66 scenes; 11 improve XY by more than 2 cm, 12
  improve SO(3) by more than 0.1 rad, and two do both.  Mean terminal XY/Z/
  SO(3) nevertheless regress from reset by 1.82 cm / 1.99 cm / 0.195 rad.
  This is behaviorally between the zero- and unit-regression-slope endpoints
  but not yet goal-conditioned success, so the frozen run proceeds to the
  model-100 gate.  Artifact:
  `outputs/teacher_eval/seed17_leaky_signed_initial_relative_joint_goal_action010_planarpush_v12_model50_balanced128/eval_summary.json`.

  `model_100` also remains 0/128 strict/constrained success.  Legal safe
  contact falls to 17/128 (13.28%), C1 rises to 29/128 (22.66%), and movement
  above 2 cm falls from 66 to 47 scenes.  Eleven scenes improve XY by more
  than 2 cm, five improve SO(3) by more than 0.1 rad, and none improves both;
  mean terminal XY/Z/SO(3) errors regress from reset by 1.05 cm / 1.60 cm /
  0.200 rad.  The run was therefore rejected and gracefully stopped at
  iteration 120 after preserving `model_{0,50,100}`.  This closes the scalar
  wrong-direction-slope sweep: zero slope preserves uncontrolled motion,
  unit slope suppresses motion, and 0.25 interpolates the two without creating
  goal-conditioned progress.  Further slope/weight sweeps are not justified.
  Artifact:
  `outputs/teacher_eval/seed17_leaky_signed_initial_relative_joint_goal_action010_planarpush_v12_model100_balanced128/eval_summary.json`.

  A policy-structure audit rules out a disconnected goal/point-cloud path.
  `ActorCriticAffordance` embeds every target point from the external
  `[x,y,z,safe,protected]` contract, encodes the complete recoverable state
  (including the relative SE(3) goal) as the query, and cross-attends that
  query over all 512 point tokens before fusion.  v12 intentionally leaves
  the optional hand/object/contact-side relation residual disabled, but that
  residual is not required for goal--point joint encoding; the completed v39
  from-scratch relation control also reached only 7.81% constrained success.
  The next useful control is therefore not another architecture or shaping
  addition.  It removes C1 learning costs from the otherwise identical
  movement-preserving baseline while retaining the strict pose/dwell task,
  to separate a safety-induced exploration failure from a base pushing
  failure before any further teacher design is considered.

  The resulting v13 no-C1 diagnostic is registered separately and cannot be
  mistaken for a selectable safe teacher.  A real four-environment,
  two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17013/2026-08-26_02-18-47_seed17013_t0_positive_initial_relative_joint_goal_action010_noc1_planarpush_smoke_v13`.
  Its saved configuration and live manager tables verify the unchanged
  4,141/4,146 actor/critic observations, `ActorCriticAffordance`, 0.10 action
  scale, 0.40-capped exploration, 30-second/300-step horizon, and strict
  XY/Z/SO(3)+five-step termination.  Exactly three rewards remain: strict
  pose success, absolute safe-set distance, and v10's positive reset-relative
  joint-pose score.  Both C1 learning costs are null, while all C1 metric
  channels remain live.  Models 0 and 1 are written successfully.  The
  formal seed-17 control is fixed to checkpoints 50/100/200/300/500 on the
  unchanged held-out 128 manifest.  Before launch, explicitly rejected
  baselines, v3--v12 sweeps, and disposable smoke checkpoints were removed;
  selected checkpoints, evaluation summaries, videos, manifests, W&B data,
  and the v13 smoke were retained.  This reduced `logs/rsl_rl` from about
  4.2 GiB to 2.1 GiB (checkpoint payload from about 3.23 GiB to 1.65 GiB).

  The formal 1,024-environment seed-17 run started from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_04-17-25_seed17_t0_positive_initial_relative_joint_goal_action010_noc1_planarpush_fromscratch_v13`.
  Its online W&B run is `simonlsx/non-prehensile-affordance/0pq363na`.
  Iteration 0 confirms 4,141/4,146 observations, the three frozen rewards,
  zero strict success as expected before learning, and live C1/C2/C3 metric
  channels.  A separate GPU watcher evaluates models 50/100/200/300/500 on
  `domino_hammer_dapl_planarpush_eval128_seed1801.jsonl`; no training metric is
  used as a selection substitute for those held-out results.

  v13 is rejected at model 200, which is sufficient to answer the no-C1
  control.  Models 50/100/200 all obtain 0/128 strict and constrained
  successes.  Their legal safe-contact rates are 28.91%/49.22%/37.50%, C1
  rates are 37.50%/37.50%/35.94%, and 79/117/103 scenes move the hammer by
  more than 2 cm.  Despite that increased interaction, mean terminal XY error
  regresses from the common 28.49 cm reset mean to 31.59/35.50/33.85 cm;
  mean SO(3) error likewise regresses from 1.676 rad to 1.918/2.046/2.003 rad.
  Only 3/3/4 scenes improve both XY by 2 cm and SO(3) by 0.1 rad.  Thus soft
  C1 is not what prevented base pushing: removing it creates more motion but
  no goal-conditioned skill.  Training and the watcher were stopped after
  preserving models 0/50/100/150/200 and all three held-out summaries.

  The next control targets one identified reward-geometry defect.  At reset,
  the held-out mean XY and SO(3) errors are about 14.2 and 16.8 strict-threshold
  units.  v13's smooth maximum therefore makes improvement in a non-maximum
  component nearly invisible.  v14 keeps every v13 setting but replaces that
  scalar with the accepted forward teacher's simultaneous signed component
  progress: XY/Z/SO(3) are normalized by 1 cm/5 mm/0.05 rad and weighted
  20/4/8.  A 1-cm wrong-way XY move therefore cannot be paid for by a
  0.05-rad rotation improvement (-20 + 8 = -12), while either correct
  component still supplies a gradient before it becomes the worst error.
  Strict success still requires XY, Z, and full SO(3) together for five steps;
  the term has no waypoint, contact latch, or hidden phase.

  The pure reward/actor regression suite passes 55/55.  A real four-env,
  two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17014/2026-08-26_05-02-12_seed17014_t0_weighted_component_progress_action010_noc1_planarpush_smoke_v14`.
  Its saved `env.yaml` verifies 0.10 action scale, 30-second horizon, null
  initial-relative and C1 reward terms, and exactly three active rewards:
  strict success, absolute safe-set distance, and the weighted component
  progress.  Models 0 and 1 are written successfully.

  The formal 1,024-env seed-17 v14 run started from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_05-08-05_seed17_t0_weighted_component_progress_action010_noc1_planarpush_fromscratch_v14`.
  Online W&B run `simonlsx/non-prehensile-affordance/nouigrwn` and the
  independent evaluator were stopped after model 100 because the unchanged
  held-out set gave a decisive negative result. Model 50 obtains 0/128 strict
  success, 6.25% legal contact, 25.78% C1, and moves only 49/128 hammers by
  more than 2 cm. Model 100 remains at 0/128, with 2.34% legal contact, 13.28%
  C1, and only 40/128 moved hammers. Its terminal-minus-reset mean errors are
  +0.77 cm XY, +1.23 cm Z, and +0.148 rad SO(3): less destructive than v13,
  but still wrong and substantially less interactive. Thus component
  visibility is useful, while a unit-slope signed transition cost makes early
  contact unattractive. Models 0/50/100 and both held-out summaries are
  preserved; v14 is rejected rather than subjected to a weight sweep.

  v15 completes that two-factor diagnosis with one scalar substitution. It
  keeps v14's independently visible XY/Z/SO(3) components but compares each
  with its episode-reset value and rectifies regressions to zero as in v13.
  The 20/4/8 component weights are normalized internally, bounding the whole
  term to [0, 1] and preserving v13's return scale. This adds no waypoint,
  contact latch, hidden phase, actor feature, or reward term; strict success,
  safe-distance shaping, action scale, PPO, manifests, and no-C1 ablation are
  unchanged. The pure metric/actor suite passes 56/56. A real four-env,
  two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17015/2026-08-26_05-42-47_seed17015_t0_positive_component_improvement_action010_noc1_planarpush_smoke_v15`.
  Runtime output and saved configuration verify 4,141/4,146 actor/critic
  dimensions, action scale 0.10, 30-second horizon, and exactly three rewards:
  strict success, safe-region distance, and positive component improvement.
  The formal 1,024-env seed-17 run then started from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_05-49-02_seed17_t0_positive_component_improvement_action010_noc1_planarpush_fromscratch_v15`.
  Online curves are at W&B run `simonlsx/non-prehensile-affordance/w1vcxln4`.
  A separate GPU watcher is fixed to models 50/100/200 on the unchanged
  balanced 128-scene seed-1801 manifest. Iterations 0--3 run normally at
  about 900--1,000 simulation steps/s; zero initial success and near-zero
  component credit are expected before the reaching policy enters the 10-cm
  shaping neighborhood.

  The model-50 balanced-128 audit rejects the hypothesis that independently
  rectified components are sufficient. It obtains 0/128 strict/constrained
  success, 25.78% legal safe-contact episodes, and 35.16% C1. Although 97/128
  hammers move more than 2 cm, only 13 improve XY by 2 cm, 18 improve SO(3)
  by 0.1 rad, and only two do both. Relative to the common reset distribution,
  mean terminal XY/Z/SO(3) errors regress by 3.64 cm / 2.67 cm / 0.292 rad.
  This is worse goal control than v13 model 50 despite more object motion:
  rectifying each component separately permits transient improvement in one
  component to pay while the joint pose deteriorates. The trainer is stopped
  after model 100 (still zero training successes). The final unchanged
  held-out trend check also obtains 0/128, with 35.94% legal contact and
  41.41% C1. It moves 85/128 hammers, but only 15 improve XY, 15 improve
  rotation, and only two improve both. Mean terminal XY/Z/SO(3) regressions
  remain +3.16 cm / +2.16 cm / +0.303 rad. Models 0/50/100 and both summaries
  are preserved, and the trainer/watcher are stopped; v15 is rejected.

  v16 changes only the scalar conjunction exposed by that result. It computes
  the minimum of reset-relative XY improvement, reset-relative full-SO(3)
  improvement, and the remaining margin to the strict 1-cm Z tolerance, then
  clips to [0, 1]. Therefore neither goal component can compensate for the
  other and leaving the support-height band cannot earn credit, while
  exploratory regression still receives zero instead of v14's contact-entry
  penalty. The pure function/actor suite passes 57/57 and all Python, shell,
  and whitespace checks pass. The task still has one goal scalar, the same
  continuous safe-distance term and sparse strict success, with no waypoint,
  latch, hidden phase, or observation change. A real four-environment,
  two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17016/2026-08-26_06-15-17_seed17016_t0_pareto_pose_improvement_action010_noc1_planarpush_smoke_v16`.
  Runtime output verifies the unchanged 4,141/4,146 actor/critic dimensions,
  action scale 0.10, 30-second horizon, no C1 term, and exactly the intended
  three rewards: strict success, continuous safe-region distance, and joint
  Pareto pose improvement. The formal 1,024-environment seed-17 run starts
  from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_06-21-59_seed17_t0_pareto_pose_improvement_action010_noc1_planarpush_fromscratch_v16`.
  Online curves are at W&B run
  `simonlsx/non-prehensile-affordance/xeiy89h7`; an independent GPU-7 watcher
  is fixed to models 50/100/200 on the unchanged balanced 128-scene held-out
  manifest. The first updates run normally at about 900 simulation steps/s.

  The model-50 and model-100 held-out audits reject v16 decisively. Both have
  0/128 strict/constrained success. Model 50 moves 98/128 hammers by more
  than 2 cm, but only 15 improve XY by 2 cm, 19 improve SO(3) by 0.1 rad,
  and three improve both; its mean terminal-minus-reset XY/Z/SO(3) errors are
  +5.06 cm / +3.05 cm / +0.317 rad. Model 100 moves 116/128 but only
  10/29/two improve XY/rotation/both, while the mean regressions grow to
  +8.78 cm / +3.02 cm / +0.377 rad. Legal contact rises from 25.78% to
  60.94%, proving that reaching and interaction are present; the minimum
  conjunction nevertheless gives useful joint improvement too rarely to
  control those interactions. Trainer and watcher are stopped after model
  100, with checkpoints and both balanced summaries preserved.

  v17 replaces only the rejected object-goal scalar with the reward geometry
  used by DyWA's released arm-diverse task. The same 512 hammer surface
  points represented to the actor are transformed by the current and goal
  poses, their corresponding distances receive DyWA's per-keypoint
  exponential potential (`0.302`, `243.12`, base `0.995`), and the mean
  potential is shaped with the PPO-matched discount `0.99`. RewardManager
  weight `1.6` accounts for its 0.1-second multiplication and recovers DyWA's
  per-step coefficient `0.16`. This one geometry jointly represents XYZ and
  full SO(3) without component rectification, a contact gate, waypoint,
  latch, hidden phase, or actor-input change. Continuous safe-region distance,
  sparse strict success, action scale, PPO, manifests, and the no-C1 root-cause
  control remain unchanged. The pure metric/actor suite passes 58/58, and
  Python, shell, and whitespace checks pass. A real four-environment,
  two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17017/2026-08-26_06-49-59_seed17017_t0_dywa_keypoint_potential_action010_noc1_planarpush_smoke_v17`.
  Runtime output verifies the frozen 4,141/4,146 actor/critic dimensions,
  seven-dimensional action, and exactly three rewards with weights 2,000,
  -2, and 1.6; both PPO updates remain finite. The formal seed-17 run then
  starts from scratch with 1,024 environments and 500 updates in tmux session
  `aff_teacher_planarpush_v17_dywa_s17_20260826` at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_06-55-27_seed17_t0_dywa_keypoint_potential_action010_noc1_planarpush_fromscratch_v17`.
  Its online W&B run id is `habuub4p`. An independent GPU-7 watcher evaluates
  models 50/100/200 on the unchanged balanced 128-scene manifest, writing
  `outputs/teacher_eval/seed17_dywa_keypoint_potential_action010_noc1_planarpush_v17_model{50,100,200}_balanced128`.

  Model 50 does not recover goal control: strict/constrained success is
  0/128, legal contact is 38.28%, and C1 is 53.12%. Although 125/128 hammers
  move by more than 2 cm, only 20 improve XY by 2 cm, 32 improve SO(3) by
  0.1 rad, and three improve both. Mean terminal-minus-reset XY/Z/SO(3)
  errors regress by +5.92 cm / +3.14 cm / +0.295 rad. Model 100 confirms the
  same failure with 0/128 success: 115 hammers move, but only 18 improve XY,
  21 improve SO(3), and five improve both; mean error regressions remain
  +4.14 cm / +2.72 cm / +0.331 rad. Training and the model-200 watcher are
  stopped, preserving models 0/50/100 and both held-out summaries.

  The exact released DyWA code audit identifies the controlled replacement
  for v18. DyWA uses canonical bounding-box keypoints rather than 512 surface
  samples, a temporal factor of 0.995, and its exponential branch does not
  multiply the configured `pot_coef=0.16`. v18 therefore changes only the
  internals of the same object-goal scalar to eight AABB corners and an
  effective scale of 1.0 (RewardManager weight 10 at 0.1-s dt), while all
  task, input, PPO, manifest, reaching, strict-success, and no-C1 settings
  remain frozen. The pure metric/actor suite passes 59/59 and all static
  checks pass. A real four-environment, two-update Isaac/PPO smoke completes
  at
  `logs/rsl_rl/franka_affordance_teacher_seed17018/2026-08-26_07-18-42_seed17018_t0_dywa_bbox_fullscale_action010_noc1_planarpush_smoke_v18`;
  runtime output verifies the frozen 4,141/4,146 observations, seven actions,
  exactly three rewards, bbox-potential weight 10, and finite PPO updates.
  The formal 1,024-environment seed-17 run starts from scratch at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_07-24-19_seed17_t0_dywa_bbox_fullscale_action010_noc1_planarpush_fromscratch_v18`;
  online curves are W&B run `c8kqz67p`. An independent GPU-7 watcher evaluates
  models 50/100/200 on the frozen balanced 128-scene held-out manifest and
  writes
  `outputs/teacher_eval/seed17_dywa_bbox_fullscale_action010_noc1_planarpush_v18_model{50,100,200}_balanced128`.

  Balanced model 50 remains at 0/128. It moves 118 hammers, but only 19
  improve XY, 22 improve SO(3), and four improve both; mean terminal-minus-
  reset XY/Z/SO(3) errors regress by +4.16 cm / +2.84 cm / +0.314 rad.
  Model 100 is worse: 0/128 success, 119 moved, only 14/16/one improve
  XY/rotation/both, and mean regressions reach +7.41 cm / +2.97 cm /
  +0.439 rad. Training and the model-200 watcher are stopped, preserving
  models 0/50/100 and both held-out results. v18 proves that bbox geometry and
  the full released exponential scale alone do not resolve goal control.

  v19 keeps v18's object-goal potential fixed and changes only the unmatched
  reaching term. The absolute safe-distance cost is replaced by DyWA's same
  exponential temporal potential applied to minimum safe-affordance distance,
  with amplitude `0.302 * 0.2 = 0.0604`, the released hand/object-to-goal
  relative scale. Thus both continuous relations use matched temporal form;
  there are still exactly three rewards and no waypoint, gate, latch, phase,
  or observation change. The pure metric/actor suite and static checks remain
  green at 59/59. A real four-environment, two-update Isaac/PPO smoke completes
  at
  `logs/rsl_rl/franka_affordance_teacher_seed17019/2026-08-26_07-48-53_seed17019_t0_dywa_matched_potentials_action010_noc1_planarpush_smoke_v19`.
  Runtime output verifies the frozen 4,141/4,146 actor/critic observations,
  seven-dimensional action, exactly three rewards with weights 2,000/10/10,
  no curriculum, and finite PPO updates. The formal seed-17 run starts from
  scratch with 1,024 environments and 500 updates at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_07-59-29_seed17_t0_dywa_matched_potentials_action010_noc1_planarpush_fromscratch_v19`;
  its online W&B run is `simonlsx/non-prehensile-affordance/fqn5bfv3`. An
  independent GPU-7 watcher is fixed to models 50/100/200 on the unchanged
  balanced 128-scene held-out manifest, writing
  `outputs/teacher_eval/seed17_dywa_matched_potentials_action010_noc1_planarpush_v19_model{50,100,200}_balanced128`.

  Model 50 remains at 0/128 strict and constrained success. Legal contact is
  31.25%, but only seven scenes improve XY by 2 cm, 11 improve SO(3) by
  0.1 rad, and none improve both. Mean terminal-minus-reset XY/Z/SO(3)
  errors regress by +3.36 cm / +2.32 cm / +0.234 rad. The matched reaching
  potential reduces v18's uncontrolled displacement but has not recovered
  goal-directed pushing; the preregistered model-100 audit remains active to
  distinguish delayed learning from a structural failure. Model 100 confirms
  the structural failure with 0/128 strict/constrained success. It moves 115
  hammers by more than 2 cm, but only 33 improve XY by 2 cm, 25 improve SO(3)
  by 0.1 rad, and eight improve both. Mean terminal-minus-reset XY/Z/SO(3)
  errors regress by +2.50 cm / +3.36 cm / +0.418 rad; legal contact reaches
  45.31% while C1 reaches 52.34%. Trainer and model-200 watcher are stopped,
  preserving models 0/50/100 and both balanced summaries. v19 therefore
  proves that matching DyWA's temporal reward forms repairs reaching but not
  goal-conditioned contact selection on the frozen planarpush distribution.

  A source-level distribution audit then identifies the next single-variable
  root control. Released DyWA `arm_div_base.yaml` has a 128-step horizon and
  task `margin_scale=0`: its XY goal lies on the current-object-to-table-centre
  ray, starting at `1.1 * 0.05 = 0.055 m`, rather than being an independent
  second point in DAPL's central box. The explicit
  `dywa-arm-div-planar-push` generator setting changes only that task
  distribution while retaining the benchmark's same support face, full yaw,
  strict XY/Z/SO(3)+dwell criterion, 300-step horizon, DOMINO hammer, actor,
  PPO, v19 rewards, and no-C1 diagnosis. Its train/eval manifests contain
  1,024/128 scenes and 16,384/2,048 unique tasks with zero overlap; every goal
  is centre-ray aligned, every relative rotation is table-normal, height
  change is exactly zero, and mean displacement is 9.74/9.78 cm. Generator
  tests pass 12 tests plus three subtests. This distribution is a named DyWA
  root-cause control and is not substituted for the final DAPL-wide gate.

  The real four-environment v20 smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17020/2026-08-26_08-31-01_seed17020_t0_dywa_matched_potentials_action010_noc1_armdiv_planarpush_smoke_v20`.
  The formal seed-17 run then starts from scratch with 1,024 environments at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_09-11-46_seed17_t0_dywa_matched_potentials_action010_noc1_armdiv_planarpush_fromscratch_v20`;
  its online W&B run is
  `simonlsx/non-prehensile-affordance/li5nolje`. Runtime output verifies the
  frozen 4,141/4,146 actor/critic observations, seven-dimensional action,
  exactly three rewards with weights 2,000/10/10, no curriculum, and finite
  PPO updates at roughly 850--985 simulation steps/s. An independent watcher
  evaluates models 50/100/200 on the disjoint 128-scene arm-div manifest and
  writes
  `outputs/teacher_eval/seed17_dywa_matched_potentials_action010_noc1_armdiv_planarpush_v20_model{50,100,200}_balanced128`.

  The balanced model-50 audit has 0/128 strict/constrained successes despite
  50/128 legal-safe-contact episodes. It moves 74/128 hammers by more than
  2 cm, but only seven improve XY by 2 cm, 12 improve SO(3) by 0.1 rad, and
  one improves both. Mean terminal-minus-reset XY/Z/SO(3) errors regress by
  +5.13 cm / +2.53 cm / +0.295 rad. Counterfactual yaw reflection changes
  actions strongly for both yaw signs, so the goal input is live; the policy
  has not yet converted that sensitivity into goal-conditioned contact and
  object motion. Model 100 also obtains 0/128. Legal safe contact falls to
  14/128 and only 47 hammers move by more than 2 cm; two improve XY by 2 cm,
  eight improve SO(3) by 0.1 rad, and one improves both. Mean terminal-minus-
  reset errors remain worse by +2.60 cm / +1.65 cm / +0.166 rad. The trainer
  and model-200 watcher are therefore stopped under the preregistered stop
  rule, preserving models 0/50/100 and both balanced summaries. Matching the
  released DyWA centre-ray distance distribution is not sufficient for the
  current single-query actor.

  The next control changes only that actor bottleneck. Local DyWA source shows
  that its teacher uses 16 state-dependent queries over object tokens, whereas
  this repository's legacy affordance actor compresses all state into one
  64-dimensional query. The v21 policy makes the query count configurable and
  sets it to 16; external target input remains exactly
  `[x,y,z,safe,protected]`, and the environment, rewards, PPO, action, critic
  privilege boundary, and v20 train/eval manifests are unchanged. Query count
  one remains exactly backward compatible. The focused actor/generation/metric
  suite passes 73 tests plus three subtests, and Python, shell, and whitespace
  checks pass. A real four-environment, two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17021/2026-08-26_09-38-23_seed17021_t0_dywa_matched_potentials_multiquery16_action010_noc1_armdiv_planarpush_smoke_v21`.
  Its saved agent contract explicitly records 16 attention queries, 45/50
  actor/critic state dimensions, and 0.4 bounded exploration; models 0 and 1
  are written successfully. The disposable smoke directory is then removed
  under `outputs/CLEANUP_2026-08-26.md` before the formal run to preserve the
  seven-GiB disk safety line.

  The formal v21 seed-17 run starts from scratch with 1,024 environments at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_09-47-22_seed17_t0_dywa_matched_potentials_multiquery16_action010_noc1_armdiv_planarpush_fromscratch_v21`;
  its online W&B run is `simonlsx/non-prehensile-affordance/x7qrtxnl`. The
  first PPO updates are finite at roughly 1,089 simulation steps/s. An
  independent watcher is fixed to models 50/100/200 on the unchanged held-out
  arm-div manifest, writing
  `outputs/teacher_eval/seed17_dywa_matched_potentials_multiquery16_action010_noc1_armdiv_planarpush_v21_model{50,100,200}_balanced128`.

  Model 50 remains at 0/128 strict/constrained success. It has 43/128 legal
  safe-contact episodes and moves 70 hammers by more than 2 cm. Eleven scenes
  improve XY by 2 cm, 13 improve SO(3) by 0.1 rad, and three improve both;
  mean terminal-minus-reset XY/Z/SO(3) regressions are +3.10 cm / +2.25 cm /
  +0.254 rad. This is a small improvement over single-query v20 model 50
  (7/12/one improvements and +5.13/+2.53 cm/+0.295 rad regressions), but zero
  instantaneous strict poses and zero successes are not evidence of a solved
  representation. Model 100 confirms the rejection with 0/128 success,
  36/128 legal-safe-contact episodes, and only 66 hammers moved by more than
  2 cm. Four scenes improve XY by 2 cm, ten improve SO(3) by 0.1 rad, and none
  improve both; mean terminal-minus-reset errors regress by +3.43 cm /
  +2.21 cm / +0.185 rad. Even the released DyWA 5-cm/0.1-rad pose criterion is
  satisfied by zero scenes at both models 50 and 100. The trainer and
  model-200 watcher are stopped, so increasing query capacity is rejected.

  The next single-variable v22 control aligns the remaining major action-space
  mismatch with released DyWA `arm_div`. It replaces seven direct relative
  joint-position actions by six bounded end-effector delta-pose actions:
  translation scale 0.06 m, rotation-axis scale 0.1 rad, raw action clipping
  to `[-1,1]`, and IsaacLab damped-least-squares differential IK. This is still
  a direct policy action, not a waypoint or scripted contact side. v20's
  single-query actor, environment, manifests, rewards, PPO, and no-C1 contract
  remain fixed; the actor/critic observations become 4,140/4,145 only because
  previous action changes from seven to six values. The focused suite passes
  74 tests plus three subtests, and static checks pass.

  The real four-environment, two-update Isaac/PPO smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17022/2026-08-26_10-14-08_seed17022_t0_dywa_matched_potentials_cartesian_noc1_armdiv_planarpush_smoke_v22`.
  Runtime manager tables verify a six-dimensional action, 4,140/4,145
  actor/critic observations, exactly the unchanged 2,000/10/10 reward terms,
  and no curriculum. Both `model_0.pt` and `model_1.pt` contain 384,973 model
  parameters and all floating-point tensors are finite; the two PPO updates
  also report finite value, surrogate, entropy, action, and reward metrics.
  Thus the Cartesian DLS/TCP/action-observation chain is accepted for the
  formal 1,024-environment seed-17 checkpoint gate.

  The formal v22 seed-17 run starts from scratch with 1,024 environments at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_10-20-51_seed17_t0_dywa_matched_potentials_cartesian_noc1_armdiv_planarpush_fromscratch_v22`;
  its online W&B run is
  `simonlsx/non-prehensile-affordance/i4wvhpnx`. Runtime output re-verifies the
  six-dimensional action and 4,140/4,145 observation contract, and the first
  PPO updates are finite at roughly 1,089 simulation steps/s. An independent
  watcher is fixed to models 50/100/200 on the unchanged held-out arm-div
  manifest, writing
  `outputs/teacher_eval/seed17_dywa_matched_potentials_cartesian_noc1_armdiv_planarpush_v22_model{50,100,200}_balanced128`.

  Model 50 has 0/128 strict/constrained successes, 20/128 legal-safe-contact
  episodes, and 25/128 diagnostic C1 episodes (C1 is measured but neither
  rewarded nor terminating in this no-C1 control). It moves 37/128 hammers by
  more than 2 cm, but only seven improve XY by 2 cm, ten improve SO(3) by
  0.1 rad, and one improves both. Mean terminal-minus-reset XY/Z/SO(3) errors
  regress by +1.94 cm / +1.47 cm / +0.101 rad. Counterfactual goal-yaw
  reflection changes the action by more than 0.05 in 86.0%/88.6% of negative/
  positive-yaw samples, so goal conditioning is live but not yet converted to
  goal-directed object motion. The single-scene diagnostic trace moves its
  hammer only 0.246 mm, while the balanced aggregate contains 37 moving
  scenes; those two intentionally different scopes must not be conflated.

  Model 100 confirms rejection rather than competence: it still has 0/128
  strict/constrained successes. Legal-safe-contact rises to 70/128 and
  93/128 hammers move by more than 2 cm, but only 14 improve XY by 2 cm, 12
  improve SO(3) by 0.1 rad, and two improve both. Mean terminal-minus-reset
  XY/Z/SO(3) errors regress more severely by +4.52 cm / +3.03 cm /
  +0.343 rad. Only two scenes end below the 2-cm XY threshold, two below the
  0.1-rad rotation threshold, and none satisfy both. Thus Cartesian control
  teaches more reaching/contact and object motion, but not goal-directed
  pushing. The trainer and model-200 watcher are stopped at iteration 134,
  preserving models 0/50/100 and both balanced summaries. This rejects the
  action-space change as an early-gate fix; no C1 reward or hard termination
  was active in this test. It is not a transition-budget-matched final
  comparison to the accepted forward v7: v7 collects 16 steps/environment per
  PPO iteration whereas v22 collects eight, so their equally numbered
  checkpoints contain twice-different experience counts.

  A post-v22 frozen-config audit identifies that the accepted forward v7 and
  arm-div v22 were never a one-factor safety comparison. The forward held-out
  distribution uses initial target x=0.46--0.50 m, y within +/-1.5 cm, goal
  direction within roughly +/-12 degrees, 6.5--9.5 cm XY displacement, and
  0.07--0.23 rad relative rotation. Arm-div expands initial target placement
  to x=0.31--0.69 m and y within +/-23.7 cm, full 360-degree goal direction,
  5.5--29.4 cm displacement, and 0--pi relative rotation. V7 additionally
  uses a fixed robot reset, seven joint actions, 16-step PPO rollouts, and 11
  active waypoint-free reward terms; v22 uses a DAPL joint-reset box, six
  Cartesian actions, eight-step rollouts, and only two dense potentials. The
  local DyWA `arm_div_base.yaml` itself specifies `franka.init_type: home`, so
  importing DAPL's uniform joint-reset box into v22 was not DyWA-aligned.

  The next v23 control therefore changes no code-side learning mechanism. It
  runs the frozen-v7
  `Isaac-AffordanceTeacher-T0-FrozenV7-Soft-Franka-v0` configuration from
  scratch on the existing arm-div train/eval manifests.
  Relative to the accepted v7, only the manifest distribution changes; fixed
  robot reset, 16-step PPO rollout, joint action, original soft-C1
  waypoint-free reward, strict 2-cm/0.1-rad/five-step success, actor, critic,
  and PPO are retained. Evaluation profile
  `t0_v7_reward_armdiv_planarpush` fixes the held-out manifest.

  A first launch against the generic T0 task was intentionally aborted before
  any checkpoint was written: its manager table exposed the post-v7 dense C1
  clearance as a twelfth reward. The frozen task removes exactly that term;
  current inactive-obstacle handling remains because it changes only absent
  C2/C3 checks, while C1 remains part of both sparse success and the historical
  soft forbidden-contact term.

  The real four-environment, two-update frozen-v7 smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17023/2026-08-26_11-01-05_seed17023_t0_frozenv7_fixedhome_joint_armdiv_planarpush_smoke_v23`.
  Runtime tables verify no randomized-joint reset event, seven-dimensional
  joint action, 4,141/4,146 actor/critic observations, exactly the 11 archived
  v7 reward terms, and no curriculum. Models 0 and 1 each contain 385,295
  parameters and all tensors are finite; both PPO updates have finite value,
  surrogate, and entropy losses. Two 16-step smoke updates are shorter than
  the 300-step episode horizon, so the absence of an episode-return scalar is
  expected.

  The formal seed-17 v23 run started from scratch with 1,024 environments at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_11-09-40_seed17_t0_frozenv7_fixedhome_joint_armdiv_planarpush_fromscratch_v23`.
  Its runtime manager tables independently verify a seven-dimensional joint
  action, 4,141/4,146 actor/critic observations, no randomized robot-joint
  reset, exactly the 11 archived v7 rewards, and no curriculum. Online curves
  are available in W&B run `oc4ahg3r`. A held-out watcher evaluates iterations
  50, 100, 200, and 300 on the fixed 128-scene arm-div evaluation manifest;
  iteration 300 is the first transition-budget-matched comparison with the
  accepted forward-v7 result.

  The model-50 held-out result is 0/128 strict and constrained successes. It
  nevertheless reaches legal safe contact in 113/128 episodes and moves the
  hammer by at least 2 cm in 113/128 scenes. Only 16 scenes improve terminal
  XY error, 27 improve rotation error, and six improve both; mean terminal
  error is 5.51 cm and 0.481 rad worse than reset. C1 occurs in 79/128
  episodes. This rejects a reaching/contact failure at this checkpoint and
  isolates the early failure to goal-conditioned contact dynamics under the
  expanded full-direction/full-yaw manifest. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_fixedhome_joint_armdiv_planarpush_v23_model50_balanced128/eval_summary.json`.

  Model 100 also obtains 0/128 strict and constrained successes. Relative to
  model 50, legal-safe-contact episodes decrease from 113 to 84 and C1
  episodes decrease from 79 to 65, while scenes moved by at least 2 cm increase
  from 113 to 127. XY-improved and rotation-improved scenes change only from
  16/27 to 18/30, scenes improving both decrease from six to four, and mean XY
  degradation grows from 5.51 cm to 9.41 cm. The actor therefore produces more
  object motion without a corresponding goal-directed trend. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_fixedhome_joint_armdiv_planarpush_v23_model100_balanced128/eval_summary.json`.

  Model 200 is the formal v23 rejection gate: 0/128 strict and constrained
  successes, 73/128 legal-safe-contact episodes, and 51/128 C1 episodes. It
  moves the hammer by at least 2 cm in 107 scenes; 38 improve XY, 23 improve
  rotation, and eight improve both, but zero scenes jointly satisfy XY below
  2 cm and rotation below 0.1 rad. Mean terminal error remains 3.99 cm and
  0.435 rad worse than reset. At the identical 200-iteration/16-step/1,024-env
  transition budget, accepted forward-v7 training has 50.74% cumulative,
  83.44% recent, and 48.69% constrained success, whereas arm-div v23 has
  0.027%, 0%, and 0.009%. This is sufficient evidence against a delayed but
  otherwise matched learning curve. The trainer and model-300 watcher were
  intentionally stopped after preserving models 0/50/100/200, all three
  held-out summaries, and online W&B run `oc4ahg3r`. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_fixedhome_joint_armdiv_planarpush_v23_model200_balanced128/eval_summary.json`.

  The only pre-registered fallback is the unlaunched
  `Isaac-AffordanceTeacher-T0-FrozenV7-GoalWrench-Soft-Franka-v0` control. It
  uses the identical frozen-v7 environment, arm-div manifest, observations,
  seven-dimensional action, 16-step rollout, reward, and PPO configuration.
  Its sole delta is an internal point-token residual derived from the existing
  actor observation: point-to-hand and object-local coordinates plus separate
  goal-translation support and signed-yaw moment channels. It adds neither a
  waypoint nor a privileged actor input. Python compilation, evaluator shell
  validation, diff validation, and all 18 affordance-actor unit tests pass.
  This control must not launch until the v23 competence gate is resolved. The
  model-200 evidence above resolves that gate as failed, so it is now the only
  authorized next training delta.

  A real four-environment, two-update v24 GoalWrench smoke completes at
  `logs/rsl_rl/franka_affordance_teacher_seed17024/2026-08-26_12-18-49_seed17024_t0_frozenv7_goalwrench_armdiv_planarpush_smoke_v24`.
  Runtime tables verify the unchanged frozen-v7 contract: fixed robot reset,
  seven-dimensional joint action, 4,141/4,146 actor/critic observations,
  exactly 11 reward terms, and no curriculum. Both PPO updates have finite
  value, surrogate, entropy, reward, and action statistics. Models 0 and 1
  each contain 394,767 parameters, including 9,472 relation-encoder
  parameters, and every checkpoint tensor is finite. Across the second
  update, eight target-relation tensors (4,800 scalar elements) change with a
  maximum absolute delta of 0.002679. The obstacle-relation tensors remain
  unchanged because this no-clutter task supplies no active obstacle signal.
  This confirms that the single GoalWrench delta is instantiated and receives
  optimization gradients before the formal 1,024-environment run.

  The formal seed-17 v24 GoalWrench control starts from scratch with 1,024
  environments at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_12-26-48_seed17_t0_frozenv7_goalwrench_armdiv_planarpush_fromscratch_v24`.
  Runtime tables again verify fixed robot reset, a seven-dimensional joint
  action, 4,141/4,146 observations, the same 11 rewards, and no curriculum.
  Online curves are available in W&B run `cxdy49th` at
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/cxdy49th`. A
  disjoint held-out watcher evaluates models 50, 100, 200, and 300 on all 128
  arm-div scenes under profile
  `t0_frozenv7_goalwrench_armdiv_planarpush`; it changes no training state.

  Model 50 obtains 0/128 strict and constrained successes, 111/128 legal-safe
  contact episodes, and 53/128 C1 episodes. Relative to the matched v23 model
  50, legal contact is nearly unchanged (113 to 111) and C1 improves (79 to
  53), but competence does not: XY-improved scenes change only from 16 to 18,
  rotation-improved scenes fall from 27 to 20, both-improved scenes fall from
  six to four, and the joint 2-cm/0.1-rad count remains zero. Mean terminal
  degradation changes from +5.51 cm/+0.481 rad in v23 to +5.70 cm/+0.488 rad
  in v24. Thus GoalWrench has not solved directional pushing at the first gate;
  model 100 remains necessary to exclude delayed representation learning.
  Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_armdiv_planarpush_v24_model50_balanced128/eval_summary.json`.

  Model 100 remains at 0/128 strict and constrained successes, with no scene
  jointly below 2 cm and 0.1 rad, but it supplies enough directional evidence
  to continue to the matched model-200 gate. Relative to v23 model 100,
  legal-safe-contact episodes improve from 84 to 118 and C1 episodes decrease
  from 65 to 34. XY-improved scenes increase from 18 to 29,
  rotation-improved scenes from 30 to 38, and both-improved scenes from four
  to 12. Mean terminal degradation shrinks from +9.41 cm/+0.491 rad to
  +7.10 cm/+0.390 rad; five scenes individually satisfy the XY threshold and
  one individually satisfies the rotation threshold. This is an incomplete
  but nontrivial representation gain, so stopping at model 100 would leave a
  delayed-learning ambiguity. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_armdiv_planarpush_v24_model100_balanced128/eval_summary.json`.

  Model 200 is the formal v24 rejection gate. It remains at 0/128 strict and
  constrained successes, with zero scenes below the joint 2-cm/0.1-rad
  threshold and zero scenes individually below 2 cm in terminal XY error.
  GoalWrench nevertheless improves the matched diagnostics relative to v23
  model 200: legal-safe-contact episodes rise from 73 to 92, C1 episodes fall
  from 51 to 32, XY-improved scenes rise from 38 to 46,
  rotation-improved scenes from 23 to 30, and both-improved scenes from eight
  to 17. Mean terminal degradation shrinks from +3.99 cm/+0.435 rad to
  +2.05 cm/+0.334 rad. This proves that the relation features are connected
  and useful, but also proves that they are insufficient for from-scratch
  full-direction/full-yaw competence at the matched budget. The trainer and
  model-300 watcher were stopped after preserving models 0/50/100/150/200,
  all three held-out summaries, and online W&B run `cxdy49th`. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_armdiv_planarpush_v24_model200_balanced128/eval_summary.json`.

- The next experiment, v25, is a distribution-only root-cause control. It keeps
  the v24 GoalWrench actor/critic, frozen-v7 rewards, 7-D joint action, PPO
  settings, strict joint XY+yaw success definition, and random initialization.
  The sole training-distribution change is from the one-shot arm-div manifest
  back to the historical forward v7 manifest that already supported high
  from-scratch competence. Held-out evaluation uses the disjoint
  `teacher_heldout_forward_v9` manifest through profile
  `t0_frozenv7_goalwrench_forward`. If v25 recovers competence, later runs will
  expand only the sampling distribution (forward -> +/-45 -> +/-90 -> wider
  yaw/full direction); if it does not, the GoalWrench relation architecture is
  rejected rather than adding further reward or waypoint machinery.

  The formal seed-17 v25 run started from scratch with 1,024 environments at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_13-37-52_seed17_t0_frozenv7_goalwrench_forward_fromscratch_v25`.
  Runtime audit confirms a 7-D action, 4,141-D actor observation, 4,146-D
  critic observation, 16 rollout steps, the frozen set of 11 reward terms, no
  curriculum, and the forward-v7 training manifest. Online W&B run:
  `dwv0uv8f`. A held-out watcher evaluates models 50, 100, 200, and 300 on 128
  disjoint forward scenes with seed 7829.

  The model-50 held-out audit obtains 8/128 strict and constrained successes
  (6.25%), 128/128 legal safe-contact episodes, and zero C1/C2/C3 violations.
  Twenty-one scenes finish below 2 cm XY error, 33 below 0.1 rad rotation
  error, and exactly the eight successes satisfy both. Mean terminal XY and
  rotation errors are 4.03 cm and 0.186 rad. This is already nonzero relative
  to v24 model 50 on arm-div (0/128), so the recoverable action relation is not
  disconnected; however, 6.25% is far below the competence gate and does not
  authorize a distribution expansion. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_forward_v25_model50_balanced128/eval_summary.json`.

  Model 100 does not improve held-out competence: it obtains 7/128 pose
  successes (5.47%) and only 3/128 constrained successes (2.34%). Although
  all scenes establish legal safe-region contact, 49/128 episodes (38.28%)
  also incur C1, dominated by protected-hand contact (48 episodes), with 19
  proximal-arm physical events. Thirty-one scenes satisfy terminal XY below
  2 cm and 18 satisfy rotation below 0.1 rad, but only seven satisfy both.
  This non-monotonic model-50-to-100 result and the simultaneous training
  recent-success value above 50% expose a train/held-out and safety-drift gap;
  training therefore continues unchanged to the predeclared model-200 gate,
  without treating online success as competence. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_forward_v25_model100_balanced128/eval_summary.json`.

  A read-only model-100 diagnostic on the 128 training-manifest scenes obtains
  only 12/128 pose successes (9.38%), 8/128 constrained successes (6.25%),
  and 65/128 C1 episodes (50.78%). The held-out manifest was also audited
  against training: both contain only negative yaw with nearly identical
  direction, distance, initial-position, yaw-magnitude, and friction ranges.
  Thus the large online/strict gap is not explained by an opposite-yaw eval
  set or primarily by unseen-scene overfitting. It shows that stochastic PPO
  rollouts with 0.4 action noise can succeed while the deterministic mean
  actor remains unstable and unsafe at this checkpoint. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_forward_v25_model100_trainmanifest128/eval_summary.json`.

  An additional read-only model-150 held-out diagnostic shows that the
  deterministic actor subsequently absorbs the successful exploration. It
  obtains 81/128 pose successes (63.28%), 75/128 constrained successes
  (58.59%), and 10/128 C1 episodes (7.81%). Every C1 event is protected-hand
  contact; proximal-arm physical C1 is zero. Mean terminal XY and rotation
  errors fall to 2.02 cm and 0.0528 rad. This large recovery from model 100
  identifies temporary deterministic mean-policy lag rather than a
  disconnected relation encoder, but remains below the approximately 80%
  competence gate; the unchanged run continues to model 200. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_forward_v25_model150_balanced128/eval_summary.json`.

  Model 200 passes the forward competence gate decisively: 121/128 held-out
  scenes succeed in both strict pose and constrained metrics (94.53%), all
  128 establish legal safe-region contact, and C1/C2/C3 are all zero. Mean
  terminal XY and rotation errors are 0.790 cm and 0.0459 rad; even p95 is
  1.39 cm and 0.0998 rad. This proves that the GoalWrench actor can learn the
  task from scratch and that v24's full arm-div failure was principally a
  one-shot distribution jump, not a disconnected action relation. Model 200
  is frozen as the accepted forward checkpoint; the next authorized change
  is sampling-distribution expansion to +/-45 degrees, with the same task,
  reward, action, observations, success predicate, and PPO. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_forward_v25_model200_balanced128/eval_summary.json`.

  Before adaptation, the accepted model 200 obtains 57/128 strict and
  constrained successes (44.53%) zero-shot on the disjoint +/-45-degree set,
  with zero C1 and 116/128 legal safe-contact episodes. This provides a
  nontrivial transfer baseline while leaving sufficient headroom for the
  distribution stage. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_forward_v25_model200_dir45_zeroshot_balanced128/eval_summary.json`.

  The seed-17 v26 +/-45-degree continuation starts from the complete v25
  model-200 state (actor, critic, optimizer, and iteration counter) at
  `logs/rsl_rl/franka_affordance_teacher_seed17/2026-08-26_14-43-39_seed17_t0_frozenv7_goalwrench_dir45_from_v25m200_v26`.
  Runtime audit confirms 1,024 environments, the unchanged 7-D/4,141/4,146
  contract, the same 11 rewards and no curriculum-manager term; only the
  training manifest changes to the audited 256-scene +/-45-degree set. Online
  W&B run: `mzvzoein`. A disjoint 128-scene watcher evaluates models 250,
  300, 350, and 400 through profile `t0_frozenv7_goalwrench_dir45`.

  Model 250 improves the frozen model-200 zero-shot baseline but does not yet
  pass the direction-expansion gate. It obtains 76/128 pose successes
  (59.38%), 74/128 constrained successes (57.81%), and 4/128 C1 episodes
  (3.12%); all four are protected-hand semantic events and proximal-arm
  physical C1 remains zero. The improvement over zero-shot is +14.84 points
  pose and +13.28 points constrained, so adaptation is working. Direction
  bins expose the remaining asymmetry: the +35 to +45 degree slice reaches
  11/16 constrained successes (68.75%) with zero C1, whereas the -45 to -35
  degree slice remains 0/14 with one C1 event. The unchanged run therefore
  continues to the predeclared model-300/350/400 gates; no reward, PPO, or
  architecture change is authorized from this checkpoint. Artifacts:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_dir45_v26_model250_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed17_goalwrench_dir45_v26_model250_direction_summary.json`.

  Model 300 passes the overall competence/safety gate with 123/128 pose
  successes (96.09%), 122/128 constrained successes (95.31%), 128/128 legal
  safe-contact episodes, and one protected-hand C1 episode (0.78%). Mean
  terminal XY and rotation errors are 0.714 cm and 0.0460 rad. It is not yet
  frozen because the direction-bin audit finds 15/16 constrained successes
  (93.75%) and zero C1 at +35 to +45 degrees, but only 9/14 (64.29%) plus the
  one C1 event at -45 to -35 degrees. The middle 98 scenes are all successful,
  so the 95.31% aggregate is not allowed to hide the negative-endpoint gap.
  Training continues unchanged to model 350. Artifacts:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_dir45_v26_model300_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed17_goalwrench_dir45_v26_model300_direction_summary.json`.

  A matching hard-C1 task/profile is now registered as
  `Isaac-AffordanceTeacher-FrozenV7-GoalWrench-C1-Franka-v0` /
  `c1_frozenv7_goalwrench_dir45`. It changes only the evaluation environment:
  the actor remains 4,141-D GoalWrench and `forbidden_region_contact` is an
  active fourth termination. A real one-scene runtime smoke passes, followed
  by a balanced 128-scene model-300 audit with 122/128 pose and constrained
  successes (95.31%), 128/128 legal safe contacts, and the same one protected
  C1 event (0.78%). Thus hard early termination does not expose any hidden
  aggregate failure beyond the already recorded soft-trajectory violation;
  the remaining rejection is specifically the negative endpoint gate.
  Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_dir45_v26_model300_hardc1_balanced128/eval_summary.json`.

  Model 350 passes both the aggregate and endpoint gates and is frozen as the
  first +/-45-degree seed-17 candidate. It obtains 127/128 pose successes
  (99.22%), 126/128 constrained successes (98.44%), 127/128 legal
  safe-contact episodes, and one protected-hand C1 episode (0.78%) with zero
  proximal-arm physical C1. Crucially, the -45 to -35 degree slice recovers
  from model 300's 9/14 to 14/14 constrained successes with zero C1; the +35
  to +45 degree slice remains 15/16 (93.75%) with zero C1. The single C1 lies
  in the easier 0 to +35 degree bin, so neither endpoint is hidden by the
  aggregate. Artifacts:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_dir45_v26_model350_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed17_goalwrench_dir45_v26_model350_direction_summary.json`.

  RSL-RL writes the final checkpoint of this 200-update continuation as
  `model_399.pt`, not `model_400.pt`. The predeclared final comparison obtains
  the same 127/128 pose and 126/128 constrained successes with one C1 event,
  but its weaker endpoint is 13/14 (92.86%) versus model 350's 15/16 (93.75%).
  Model 350 therefore remains the first-passing and endpoint-better selected
  checkpoint; the final-iteration filename is not silently substituted.
  Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_dir45_v26_model399_balanced128/eval_summary.json`.

  The selected model 350 also passes the exact hard-C1 environment on the
  same balanced 128 scenes: 126/128 strict and constrained successes (98.44%),
  126/128 legal safe-contact episodes, and one protected-hand C1 episode
  (0.78%), with no proximal-arm physical contact. Mean terminal XY and
  rotation errors are 0.680 cm and 0.0335 rad. Artifact:
  `outputs/teacher_eval/seed17_frozenv7_goalwrench_dir45_v26_model350_hardc1_balanced128/eval_summary.json`.

  Two deterministic held-out endpoint demos verify the same hard-C1 contract.
  Scene 16 pushes at -41.19 degrees and scene 15 at +38.04 degrees; each is a
  1/1 strict/constrained success with legal safe-region contact and zero
  C1/C2/C3 events. Both show the cyan transparent goal hammer and the green
  safe/red protected overlays. The positive recording's renderer warm-up
  frames were removed before producing the accepted 4x slow-motion exports:
  `outputs/teacher_demos/seed17_v26_model350_dir45_hardc1/negative_scene16/seed17_model350_negative41deg_goal_affordance_hardc1_trimmed_slowmo4x.mp4`
  and
  `outputs/teacher_demos/seed17_v26_model350_dir45_hardc1/positive_scene15/seed17_model350_positive38deg_goal_affordance_hardc1_trimmed_slowmo4x.mp4`.
  Their auditable sidecars are the adjacent `eval/eval_summary.json` files.

  Independent seed-23 and seed-41 replications use the same two-stage
  from-scratch forward-to-+/-45-degree protocol. Their durable experiment
  roots are `logs/rsl_rl/franka_affordance_goalwrench_seed23` and
  `logs/rsl_rl/franka_affordance_goalwrench_seed41`; these names deliberately
  avoid legacy experiment-name symlinks into `/tmp`. The forward runs use
  W&B IDs `r4jemdeu` and `5ysjodye`, respectively. Because these launches use
  exactly 200 updates, RSL-RL writes their final checkpoints as
  `model_199.pt` (the seed-17 v25 run had a larger configured budget and was
  stopped at its interval checkpoint `model_200.pt`). The fixed watcher was
  corrected to the observed checkpoint metadata rather than waiting for a
  nonexistent file.

  Balanced deterministic evaluation on the same 128 held-out forward scenes
  gives seed 23 125/128 and seed 41 126/128 strict/constrained successes
  (97.66% and 98.44%). Both establish legal safe-region contact in 128/128
  episodes. Seed 23 records one protected-hand C1 episode (0.78%) and seed 41
  records zero; neither records proximal-arm physical C1. Together with seed
  17's accepted 121/128 result, the independent three-seed forward gate is
  372/384 constrained successes (96.88%, seed range 94.53--98.44%), 384/384
  legal safe contacts, and 1/384 C1 episodes (0.26%). Artifacts:
  `outputs/teacher_eval/goalwrench_forward_three_seed_v27b_summary.json`,
  `outputs/teacher_eval/seed23_frozenv7_goalwrench_forward_v27b_model199_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed41_frozenv7_goalwrench_forward_v27b_model199_balanced128/eval_summary.json`.

  Each policy then starts its own full-state continuation from `model_199.pt`;
  the first trainer line is `Learning iteration 199/399`, proving that the
  iteration/optimizer state is restored rather than applying a weights-only
  restart. The only distribution change is the audited +/-45-degree manifest.
  Runtime then writes `model_200.pt`, confirming that periodic saves remain
  aligned to absolute multiples of 50. The equivalent checkpoint grid is
  therefore 250/300/350/398: its intermediate gates match seed 17 exactly,
  while only the final checkpoint is one lower because the parent state is
  iteration 199 and the continuation budget is exactly 200 updates.
  Seed-23 and seed-41 continuation W&B IDs are `lq0f4gu9` and `p9jh436e`.
  No checkpoint from seed 17 is copied into either run.

  The first hard-C1 +/-45-degree gate at model 250 is diagnostic and does not
  yet pass. Seed 23 obtains 80/128 constrained successes (62.50%) with 20 C1
  episodes (15.62%); its -35 to -45 and +35 to +45 endpoint slices are 4/14
  and 7/16. Seed 41 obtains 102/128 (79.69%) with 21 C1 episodes (16.41%);
  its endpoints are 14/14 and 2/16. All 41 C1 events are hand-semantic and
  proximal-arm physical C1 remains zero. The complementary endpoint failures
  show why neither aggregate nor one signed endpoint is sufficient for
  selection. Training therefore continues unchanged to the predeclared
  model-300/350/398 gates. Artifacts are
  `outputs/teacher_eval/seed{23,41}_frozenv7_goalwrench_dir45_v28b_hardc1_model250_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/goalwrench_dir45_v28b_hardc1_direction_summary.json`.

  Model 300 remains below the joint competence/safety gate. Seed 23 obtains
  77/128 constrained successes (60.16%) with 14 C1 episodes (10.94%); its
  negative and positive endpoint slices are 8/14 and 5/16. Seed 41 obtains
  103/128 (80.47%) with 18 C1 episodes (14.06%); its endpoint slices are
  12/14 and 3/16. Seed 23 has one proximal-arm physical C1 event; all other
  C1 events at this gate are hand-semantic. The run remains the same
  distribution-only continuation rather than introducing a reward or
  architecture response to this intermediate checkpoint.

  Model 350 improves aggregate competence but still fails the safety and
  positive-endpoint gates. Seed 23 reaches 101/128 constrained successes
  (78.91%) with 9 C1 episodes (7.03%); its negative endpoint is 14/14 while
  the positive endpoint is only 3/16. Seed 41 reaches 112/128 (87.50%) with
  13 C1 episodes (10.16%); its endpoints are 13/14 and 6/16. Proximal-arm
  physical C1 is zero for seed 41. Thus seed 41 crosses the 85% aggregate
  competence threshold, but neither seed satisfies C1 <= 1% or the required
  >=75% success with zero C1 at both signed endpoints. Neither checkpoint is
  selected; both unchanged runs continue to their predeclared final
  `model_398.pt` comparison.

  The final model-398 audit confirms that longer training under the weak
  frozen-v7 soft cost is not sufficient. Seed 23 obtains 104/128 constrained
  successes (81.25%) with 20 C1 episodes (15.62%); its endpoints are 14/14
  and 5/16. Seed 41 remains at 112/128 (87.50%) with 14 C1 episodes (10.94%);
  its endpoints are 13/14 and 6/16. Relative to model 350, seed 23 gains only
  three constrained successes while C1 more than doubles, and seed 41 is
  unchanged in aggregate/endpoint competence while C1 worsens by one scene.
  Neither seed passes, so no clutter transfer is authorized.

  Matching no-termination evaluations of the same final actors isolate the
  failure. Seed 23 reaches 116/128 pose successes (90.62%) and seed 41 reaches
  124/128 (96.88%), while still recording 16/128 and 7/128 C1 episodes. Thus
  both policies have learned the +/-45-degree pose task; immediate hard C1
  primarily exposes an illegal-contact shortcut rather than missing pushing
  competence. Artifacts:
  `outputs/teacher_eval/seed23_frozenv7_goalwrench_dir45_v28b_model398_soft_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed41_frozenv7_goalwrench_dir45_v28b_model398_soft_balanced128/eval_summary.json`.

  The v29 refinement therefore changes exactly one curriculum objective: the
  frozen-v7 weak C1 cost is replaced by the repository's existing complete
  C1-soft profile. The binary contact weight changes from -5 to -25 and the
  predicate-aligned 10--20 mm clearance term is enabled at weight -4; hard
  C1 termination remains disabled. GoalWrench, actor/critic observations,
  action space, PPO, 1,024 environments, and the +/-45-degree manifest are
  unchanged. A real one-environment runtime smoke verifies 7-D actions,
  4,141/4,146 observations, 12 rewards, three non-C1 terminations, and
  checkpoint compatibility. The three full-state continuations all begin at
  `Learning iteration 399/499`; online W&B IDs are `spqdfw12`, `f28a6ybi`,
  and `gflemelx` for seeds 17, 23, and 41. The predeclared strict comparisons
  are model 450 and the final model 498.

  Balanced Hard-C1 evaluation of model 450 gives seed 17 127/128 (99.22%,
  0.78% C1), seed 23 115/128 (89.84%, 0.78% C1), and seed 41 124/128
  (96.88%, zero C1). Although all three pass the aggregate competence gate,
  seed 23 reaches only 9/16 (56.25%) on the positive >=35-degree endpoint,
  while seed 17's sole arm-target physical event lies in its positive endpoint.
  Neither issue is hidden by reporting only the 366/384 aggregate result.

  Final model 498 removes the seed-23 directional failure: seed 23 obtains
  121/128 (94.53%), zero C1, and 14/14 negative plus 13/16 positive endpoint
  successes. Seed 41 obtains 125/128 (97.66%), zero C1, and 14/14 plus 14/16
  endpoint successes. Seed 17 model 498 is rejected despite 123/128 (96.09%)
  success because five proximal-arm physical C1 episodes make its C1 rate
  3.91% and reduce the negative endpoint to 9/14. The six-checkpoint audit is
  `outputs/teacher_eval/goalwrench_c1soft_dir45_v29_hardc1_three_seed_summary.json`.

  To retain one consistent complete-C1-soft contract without relaxing the
  endpoint gates, seed 17 is rewound to its already endpoint-competent v26
  `model_350.pt` and receives exactly ten updates under the same complete
  non-terminating C1 cost. No observation, action, PPO, manifest, or reward
  term changes. The resulting v30 `model_359.pt` reaches 127/128 (99.22%)
  constrained success, zero C1, 13/14 negative endpoints, and 16/16 positive
  endpoints. Its W&B run is `0yg5nxy8`, and its gate artifact is
  `outputs/teacher_eval/seed17_goalwrench_c1soft_dir45_v30_hardc1_summary.json`.

  The frozen three-seed C1 checkpoint set is therefore seed-17 v30 model 359,
  seed-23 v29 model 498, and seed-41 v29 model 498. Together they obtain
  373/384 constrained successes (97.14%), 383/384 legal safe-contact episodes,
  zero C1 events, 41/42 negative endpoint successes, and 43/48 positive
  endpoint successes. Every seed independently passes >=85% overall, <=1%
  overall C1, and >=75% with zero C1 on both signed >=35-degree endpoints.
  The selected paths and source summaries are recorded in
  `outputs/teacher_eval/goalwrench_c1soft_dir45_v30_selected_three_seed_summary.json`.

  Two additional rendered endpoint demos verify the selected seed-23 and
  seed-41 checkpoints. Each sidecar is a 1/1 deterministic Hard-C1 success
  with legal safe-region contact and zero C1/C2/C3. Renderer warm-up frames
  are removed from the accepted 4x slow-motion exports:
  `outputs/teacher_demos/seed23_v29_model498_dir45_hardc1/positive_scene7/seed23_model498_positive44deg_goal_affordance_hardc1_trimmed_slowmo4x.mp4`
  and
  `outputs/teacher_demos/seed41_v29_model498_dir45_hardc1/negative_scene16/seed41_model498_negative41deg_goal_affordance_hardc1_trimmed_slowmo4x.mp4`.

  C2 begins only after freezing that C1 set. The earlier v31 12 cm eraser
  transfer is rejected as a matched comparison: its no-blocker control obtains
  0/36, so the forward-only subset no longer measures the already proved C1
  policy. Its W&B run `7middocu` was stopped and is retained only as a negative
  diagnostic. A subsequent scale-0.007 eraser proposal is also rejected: only
  16/74 held-out objects satisfy the reset/settling audit because the resulting
  support thickness is sub-millimetre. Artifact:
  `outputs/teacher_diagnostics/c2_matched_v32_eval_settling_audit_scale0007.json`.

  The provisional v35 phone construction is rejected after a direct rerun of
  its supposedly final manifests. Fifteen of 81 held-out blockers change pose
  during zero-action settling, so the apparent 50/81 = 61.73% zero-shot result
  does not measure the intended fixed initial condition and is non-reportable.
  Its soft-C2 run `t5e4dk40` was stopped. The v36--v39 proposals are likewise
  rejected because their blocker support pose or non-uniform scaling does not
  pass the same direct stability audit.

  The accepted v40 construction preserves the C1 hammer start/goal states and
  target dynamics while adding one physical, collidable, contact-sensed but
  kinematic blocker: DOMINO `062_plasticbox:0`, uniformly scaled by `0.006` and
  placed at the protected-region straight-sweep midpoint. Kinematic applies
  only to the typed obstacle; the hammer remains dynamic. Geometry filtering
  requires the complete target and protected region to remain more than 10 mm
  from the blocker at both endpoints while the protected sweep midpoint is at
  most 5 mm away. It retains 158/293 training and 81/136 disjoint-seed
  evaluation candidates. The final manifests are
  `data/manifests/teacher_c2_matched_v40/hammer_teacher_dir45_c2_plasticbox_uniform006_kinematic_train_clear10_seed8831.jsonl`
  and
  `data/manifests/teacher_c2_matched_v40/hammer_teacher_dir45_c2_plasticbox_uniform006_kinematic_eval_clear10_seed9833.jsonl`;
  geometry evidence is in
  `outputs/teacher_diagnostics/c2_matched_v40_kinematic_{train293,eval136}_clear10_geometry_audit.json`.

  Direct 30-step PhysX audits pass all 158 training and 81 evaluation scenes at
  3 mm translation, 0.03 rad rotation, 0.01 m/s linear-speed, 0.10 rad/s
  angular-speed, and a deliberately strict 0.01 N raw-contact threshold. The
  blocker has exactly zero pose change and speed; the hammer's maximum terminal
  speed is 0.000168 m/s and 0.00601 rad/s in training and 0.000100 m/s and
  0.00491 rad/s in evaluation. Both audits record zero reset C1, C2, or raw
  target--obstacle contacts. Evidence:
  `outputs/teacher_diagnostics/c2_matched_v40_kinematic_train158_clear10_settling_audit.json`
  and
  `outputs/teacher_diagnostics/c2_matched_v40_kinematic_eval81_clear10_settling_audit.json`.

  On that accepted hard-C2 evaluation set, the frozen seed-17 v30 model-359
  zero-shot baseline obtains 5/81 = 6.17% constrained success and 77/81 legal
  safe-contact episodes, with 3 C1, 72 C2, and zero C3 violation episodes. This
  is a clean C2 learning gap: the competent C1 policy pushes the protected sweep
  through the blocker rather than avoiding it. Artifact:
  `outputs/teacher_eval/seed17_v30_model359_c2matched_v40_clear10_zeroshot_hard_balanced81/eval_summary.json`.

  The v41 1,024-environment soft-C2 continuation restores the complete
  model-359 optimizer state and changes only the C2 training condition; C3
  reward and termination remain disabled. Its online W&B ID is `eay6mv4k`.
  Hard-C2 evaluation is predeclared at models 400, 450, 500, 550, and 558 on all
  81 held-out scenes using profile `c2_goalwrench_matched_box_clear10`.
  Selection requires at least 75% constrained success, ideally with zero C1 and
  zero C2; training-window success is diagnostic only. No C3 or combined-clutter
  run is authorized before this gate passes.

  The first v41 checkpoint does not pass. Hard-C2 `model_400` obtains 2/81 pose
  successes and 1/81 = 1.23% constrained success, with 6 C1 and 67 C2 episodes;
  the zero-shot source had 5/81 constrained successes, 3 C1, and 72 C2. Its
  mean terminal XY error worsens from 5.21 cm to 6.18 cm even as C2 decreases
  slightly. A matched no-obstacle Hard-C1 audit isolates substantial competence
  forgetting: the same `model_400` obtains only 69/128 = 53.91% constrained
  success with 3 C1, versus 127/128 = 99.22% and zero C1 for the selected source
  checkpoint. Artifacts:
  `outputs/teacher_eval/seed17_v41_c2matched_v40_clear10_hard_model400_balanced81/eval_summary.json`
  and
  `outputs/teacher_eval/seed17_v41_model400_noobstacle_hardc1_balanced128/eval_summary.json`.
  The unchanged run is retained through the predeclared model-450 audit to test
  whether this is transient. It is not: `model_450` obtains 0/81 pose or
  constrained successes, 21 C1, and 22 C2 episodes. C2 decreases only because
  the policy stops making useful progress: mean terminal XY error reaches
  8.03 cm and mean yaw-progress ratio is -0.97. v41 and its watcher are stopped;
  models 500/550/558 are not run. Artifact:
  `outputs/teacher_eval/seed17_v41_c2matched_v40_clear10_hard_model450_balanced81/eval_summary.json`.

  v42 is the minimal replay control, not a new policy or waypoint. It retains
  the identical actor, rewards, PPO, one kinematic plastic-box blocker, and hard
  held-out C2 set. The only changed factor is the training manifest: 123
  geometry/PhysX-audited scenes place the same blocker 50 mm laterally so the
  protected straight-sweep midpoint has at least 20 mm clearance, mixed once
  with all 158 v40 conflicting scenes (43.8% easy, 56.2% hard). Every easy scene
  also has more than 10 mm whole-target endpoint clearance and zero 0.01 N reset
  C1/C2/raw contact. Artifacts:
  `data/manifests/teacher_c2_matched_v42/hammer_teacher_dir45_c2_plasticbox_uniform006_easy123_hard158_mixed281_seed38831.jsonl`,
  `outputs/teacher_diagnostics/c2_matched_v42_lateral50_train256_clear20_geometry_audit.json`,
  and
  `outputs/teacher_diagnostics/c2_matched_v42_lateral50_train123_clear10_midclear20_settling_audit.json`.
  The 1,024-environment seed-17 run again restores the selected model-359 full
  optimizer state; W&B ID `teccd5hp`. Hard-C2 models 400/450/500/508 are
  predeclared on the unchanged 81-scene v40 evaluation set.

  A subsequent evaluation-entry audit found that the shell wrapper accepted
  custom training manifests through `DAPL_CLUTTER_MANIFEST`, but its evaluation
  side resolved only `MANIFEST` and otherwise silently selected the profile
  default.  Consequently, the old reports labelled as the lateral-50 control
  (`1/68`) and lateral-100 full-sweep-clear control (`0/73`) actually evaluated
  the default 81-scene v40 center-blocker manifest and are rejected.  The v40
  zero-shot/model-400 reports remain valid because v40 was their intended
  profile default.  The wrapper now resolves explicit `MANIFEST`, then
  `DAPL_CLUTTER_MANIFEST`, then the profile default; `eval.py` also persists the
  resolved `clutter_manifest` in every new summary.

  Corrected hard-C2 controls establish a smooth, learnable difficulty axis.
  The frozen v30 model-359 obtains 66/73 = 90.41% on the lateral-100 set with
  one C1 and zero C2 episodes, and 47/68 = 69.12% on the lateral-50 set with
  three C1 and one C2 episode.  The already-trained v42 model-400 improves the
  latter to 53/68 = 77.94% with three C1 and zero C2 episodes; on lateral-100 it
  retains 58/73 = 79.45% with zero C1 and one C2 episode.  Evidence:
  `outputs/teacher_eval/seed17_v30_model359_fullsweepclear30_easyblocker_hardc2_balanced73_manifestfix_v44/eval_summary.json`,
  `outputs/teacher_eval/seed17_v30_model359_lateral50_hardc2_balanced68_manifestfix_v44/eval_summary.json`,
  `outputs/teacher_eval/seed17_v42_model400_lateral50_hardc2_balanced68_manifestfix_v44/eval_summary.json`,
  and
  `outputs/teacher_eval/seed17_v42_model400_fullsweepclear30_far_hardc2_balanced73_manifestfix_v45/eval_summary.json`.

  v44 then changes only the blocker lateral offset to the round values 40, 30,
  20, and 10 mm.  Geometry filtering requires 10 mm whole-target endpoint
  clearance, and direct 30-step PhysX audits at a strict 0.01 N contact
  threshold accept all retained training/evaluation scenes with zero reset C1,
  C2, or raw target--obstacle contact.  On the disjoint hard-C2 sets, v42
  model-400 obtains respectively 51/65 = 78.46%, 47/62 = 75.81%, 29/42 =
  69.05%, and 18/43 = 41.86%; C2 counts are 1, 2, 1, and 14.  Together with
  the unchanged center-blocker result 4/81 = 4.94%, this localizes the remaining
  gap to the 10-to-0-mm transition rather than an obstacle-input failure.
  Geometry/PhysX evidence is under
  `outputs/teacher_diagnostics/c2_curriculum_v44_*`; evaluation evidence is
  under `outputs/teacher_eval/seed17_v42_model400_c2_lateral*_v44/`.

  v45 is therefore a blocker-offset curriculum, not a new architecture,
  waypoint, or reward.  Its 731-scene soft-C2 replay contains audited
  lateral-100/50/40/30/20 scenes once and lateral-10 scenes twice.  It resumes
  v42 model-400 with 1,024 environments for 50 iterations, saving every ten
  iterations.  W&B run `4d5fie2c`; checkpoint selection is by disjoint
  lateral-10 hard-C2 evaluation at models 410/420/430/440/449, followed by the
  unchanged v40 center-blocker and lateral-100 competence gates.

  Held-out selection stops v45 early at model-430.  Lateral-10 hard-C2 improves
  from the v42 source's 18/43 = 41.86% to 20/43 = 46.51% at model-410 and peaks
  at 27/43 = 62.79% at model-420; C2 falls from 14 to 7 while all 43 episodes
  retain legal safe contact.  Model-410 also restores the lateral-100 control
  to 66/73 = 90.41% with zero C2, proving replay prevents competence collapse.
  The never-trained center blocker improves only modestly at model-420, from
  4/81 = 4.94% to 6/81 = 7.41%.  Model-430 then regresses to 23/43 = 53.49%
  with ten C2 episodes, so models 440/449 are not run and model-420 is selected.

  v46 remains the same soft-C2 task and starts from v45 model-420.  It changes
  only replay proportions: audited lateral-100, lateral-40, and lateral-20
  scenes appear once, while lateral-10 appears five times (390/727 scenes).
  The 1,024-environment continuation is limited to 20 iterations with a
  five-iteration checkpoint interval; W&B run `ti7t43am`.  Models
  425/430/435/439 are selected first on lateral-10 hard-C2.  The center blocker
  remains excluded until that disjoint gate reaches at least 75%.

  C2-v46 is rejected after its first strict checkpoint: model-425 obtains only
  20/43 = 46.51%, with seven C1 and ten C2 episodes.  Increasing exploration
  noise in a matched hard-C2 continuation likewise never exceeds the frozen
  model-420 source (the best checked model is 25/43 = 58.14%).  A six-update
  fixed-rate critic-only warm-up changes zero actor tensors and improves critic
  conditioning, but the subsequent hard continuation only ties the source at
  27/43 before regressing.  These controls rule out checkpoint loading,
  exploration standard deviation, and an immediately stale critic as the
  primary 10-mm bottleneck.

  Per-scene inspection reveals a data contract problem hidden by the aggregate
  62.79%: the selected model-420 obtains 0/7 on positive-side blockers and
  27/36 on negative-side blockers.  The corresponding replay is also strongly
  skewed (632 negative versus 95 positive scenes).  Simply repeating the
  minority class is not sufficient.  C2-side-v54 balances multiplicity inside
  every offset but its strict models 425/430/435/439 obtain 19/43, 19/43,
  15/43, and 21/43; the positive side remains 0/7 throughout while the learned
  negative side is forgotten.  Evidence is under
  `outputs/teacher_eval/seed17_v54_sidebalanced_lateral10_hard_model*_balanced43/`,
  with machine-readable `side_summary.json` files generated by
  `scripts/summarize_teacher_c2_side_eval.py`.

  The corrected paired construction emits both blocker sides for every
  identical base task and keeps blocker mass, friction, orientation, target,
  goal, and parked objects fixed within each pair.  Geometry filtering is
  followed by `scripts/filter_complete_teacher_c2_pairs.py`, so a task is kept
  only when both counterfactual sides pass the same 10-mm endpoint-clearance
  gate.  The first audited set contains 29 training pairs (58 scenes) and 14
  disjoint evaluation pairs (28 scenes).  Both sets pass 30 zero-action PhysX
  steps with zero C1, C2, or raw target--obstacle contacts.  Artifacts:
  `data/manifests/teacher_c2_paired_v55/hammer_dir45_c2_box_lateral10_train58_completepairs_stable_endpointclear10_seed55010.jsonl`,
  `data/manifests/teacher_c2_paired_v55/hammer_dir45_c2_box_lateral10_eval28_completepairs_stable_endpointclear10_seed55011.jsonl`, and
  `outputs/teacher_diagnostics/c2_paired_v55_lateral10_{train,eval}_completepairs_physx.json`.
  Frozen model-420 obtains 5/28 = 17.86% on this intentionally unbiased hard
  set: 0/14 positive and 5/14 negative.  This is the authoritative paired
  baseline, not a replacement for the older difficulty-curve set.  The
  1,024-environment C2-paired-v57 run changes only this training manifest and
  is evaluated at models 425/430/435/439 on all 28 paired held-out scenes.
  Online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/kssllkqq`.

  v57 is rejected as a 10-mm direct jump.  Models 425/430/435/439 obtain
  3/28, 3/28, 5/28, and 5/28 strict successes, respectively.  Models 435 and
  439 recover only the frozen model-420 paired baseline (5/28), while the
  positive-side slice remains 0/14 at every checkpoint.  Model 439 removes
  C1 from the paired set but still records C2 in 20/28 episodes.  Thus simple
  paired replay fixes the data balance but does not by itself provide a
  discoverable positive-side avoidance behavior at 10 mm.  Machine-readable
  evidence is under
  `outputs/teacher_eval/seed17_v57_completepairs_lateral10_paired_hard_model*_balanced28/`.

  v58 is the next single-variable curriculum control: retain the same task,
  reward, actor, optimizer, 1,024 environments, and model-420 initialization,
  but begin with a 20-mm paired blocker set.  The original 10-mm geometry
  admission threshold leaves too few easier pairs, so this stage uses a 5-mm
  sampled endpoint-clearance gate followed by the authoritative 0.01-N,
  30-step PhysX contact audit.  All 26 training scenes and all 10 disjoint
  evaluation scenes pass with zero C1, C2, or raw target--obstacle contact.
  Frozen model-420 obtains 2/10 on the 20-mm paired hard set: 0/5 positive and
  2/5 negative, with positive-side C2 in 4/5.  This small set is a curriculum
  diagnostic only; the 28-scene 10-mm paired set remains the final selection
  gate.  Artifacts are under
  `data/manifests/teacher_c2_paired_curriculum_v58/`,
  `outputs/teacher_diagnostics/c2_paired_v58_lateral20_*`, and
  `outputs/teacher_eval/seed17_v58_source_model420_lateral20_paired_hard_balanced10/`.
  Online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/1w6qy60e`.

  For +/-45-degree audits, run the direction summarizer with
  `--endpoint-min-abs-deg 35`, `--require-negative-endpoint`, and
  `--skip-bidirectional-yaw-gate`. This makes both observed endpoint slices
  explicit while correctly deferring bidirectional goal-yaw randomization to
  its later curriculum axis. The default 70-degree behavior remains
  compatible with +/-90-degree evaluations.

- The legacy 45-value forward warm-up actor at `model_150` obtains 68/128
  strict and constrained successes (53.12%) with zero typed violations on the
  disjoint forward set.  Artifact:
  `outputs/teacher_eval/seed17_v6_forward_model150_balanced128/eval_summary.json`.
- The asymmetric-critic seed-41 actor at `model_100` obtains 89/128 pose
  successes (69.53%), 88/128 constrained successes (68.75%), and 16.41% C1
  episode violations.  It demonstrates pose learning but does not pass the C1
  gate.  Artifact:
  `outputs/teacher_eval/seed41_v7_forward_model100_balanced128/eval_summary.json`.
- Three independent asymmetric-critic runs evaluated at `model_250` on the
  128-scene disjoint forward set obtain, respectively: seed 17, 81.25% pose /
  80.47% constrained / 0.78% C1; seed 23, 91.41% / 85.94% / 6.25%; and seed
  41, 97.66% / 97.66% / 1.56%.  Thus seed 41 passes the forward T0 gate, but
  the three-seed result is not yet stable.  Artifacts are under
  `outputs/teacher_eval/seed{17,23,41}_v7_forward_model250_balanced128/`.
- At `model_300`, the same three seeds obtain 92.19% / 89.06% / 3.91%,
  95.31% / 89.84% / 5.47%, and 96.09% / 95.31% / 1.56% pose / constrained /
  C1, respectively.  More iterations improve pose coverage but do not by
  themselves satisfy the safety gate for seeds 17 and 23.  Artifacts are
  under
  `outputs/teacher_eval/seed{17,23,41}_v7_forward_model300_balanced128/`.
- Re-evaluating the original seed-23 `model_300` with immediate hard-C1
  termination obtains 91.41% constrained success and 6.25% C1.  All eight
  violating episodes are hand-semantic events and none is a proximal-arm
  physical collision.  Artifact:
  `outputs/teacher_eval/seed23_v7_original_model300_hard_c1_audit/eval_summary.json`.
- Direct hard-C1 continuation was rejected: dense checkpoints `model_155` and
  `model_165` reached only 60.94% and 67.19% constrained success, despite
  training-window recent success near 90%.  This is policy drift toward a
  safer but less useful behavior, and motivates the explicit C1-soft
  clearance stage rather than longer hard training.
- The protected-only C1-soft continuation from seed-23 `model_300` is also
  rejected.  Hard-C1 evaluation of `model_325`, `model_340`, and `model_349`
  falls from 79.69% to 77.34% to 73.44% constrained success while C1 rises
  from 17.19% to 20.31% to 23.44%.  Every violation is hand-semantic and none
  is proximal-arm contact.  This demonstrates that the earlier 4 cm
  protected-only hinge was not aligned with the complete C1 predicate.
  Artifacts are under
  `outputs/teacher_eval/seed23_v16_c1soft_model{325,340,349}_hard_c1_audit/`.
- The first ±45-degree transfer checkpoint (`seed41_v12`, `model_200`) obtains
  51.56% pose success, 42.19% constrained success, and 19.53% C1.  Success is
  concentrated near the original forward direction, so it does not pass the
  direction-expansion gate.
- By `model_350`, v12 reaches 90.62% pose success but only 79.69% constrained
  success with 14.06% C1.  Its +45-degree endpoint remains weak at 42.9% pose
  success.  In contrast, the protected-clearance v15 `model_350` obtains
  73.44% pose, 71.88% constrained, and 2.34% C1; the three C1 episodes include
  two neutral and two protected contacts (one episode contains both), with no
  proximal-arm collision.  The complementary results motivate v17: a new
  seed-41 transfer from accepted `model_250` using the complete neutral plus
  protected 10--20 mm C1 margin.  Its online run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/mv6prqmb`.
- The complete-C1 safety-refine run v18 improves its selected soft checkpoint
  (`model_428`) to 89.84% pose success, 87.50% constrained success, and 3.12%
  C1 on the balanced 128-scene +/-45-degree set.  The remaining four C1
  episodes are hand-semantic events (four neutral, two protected, with overlap)
  and there is no proximal-arm physical contact.  Artifact:
  `outputs/teacher_eval/seed41_v18_dir45_model428_balanced128/eval_summary.json`.
- Conservative hard-C1 continuation must be selected densely rather than by
  final iteration.  v19 checkpoints range from 25.00% to 87.50% strict success;
  the final `model_443` has collapsed to 32.03%, while the transient
  `model_438` obtains 112/128 = 87.50% strict/constrained success with zero C1
  events.  A second evaluation seed obtains 118/128 = 92.19%, again with zero
  C1; the two-run mean is 89.84%.  The first evaluation's +35 to +45 degree
  bin remains the weakest at 56.2%, so this checkpoint is accepted for the
  +/-45 safety gate but direction expansion continues before clutter.  Artifacts:
  `outputs/teacher_eval/seed41_v19_dir45_model438_hardc1_balanced128/` and
  `outputs/teacher_eval/seed41_v19_dir45_model438_hardc1_repeat_seed7002/`.
- The +/-90-degree expansion run v20 starts from the selected v19 `model_438`
  under the soft T0 profile with 1,024 environments.  Its online run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/ay9lxan7`.  The
  first audited checkpoint, `model_500`, is rejected: it obtains 57.03% pose,
  53.91% constrained success, and 8.59% C1 on the balanced held-out set.
  Artifact:
  `outputs/teacher_eval/seed41_v20_dir90_model500_balanced128/eval_summary.json`.
  `model_550` improves to 63.28% pose / 62.50% constrained / 3.12% C1 but is
  also below gate.  Its +70 to +90 degree bin remains 0/16 while the -35 to
  0 degree bin is 21/21, making the remaining limitation a directional skill
  imbalance rather than uniform pose noise.  Artifact:
  `outputs/teacher_eval/seed41_v20_dir90_model550_balanced128/eval_summary.json`.
  `model_600` subsequently regresses to 58.59% pose/constrained success with
  1.56% C1 and still has 0/16 success in the +70 to +90 degree bin, so v20 is
  rejected rather than extended.
- A terminal-reward audit found a configuration bug shared by the no-clutter
  T0 and hard-C1 profiles.  Their reward term still evaluated C2/C3 against
  inactive obstacle slots in the common scene schema.  Consequently every
  successful training episode logged `Episode_Reward/task_success=0` even
  when `Episode_Termination/reached` and the independent constrained-success
  metric were true.  A replay of the accepted scene-71 trajectory records
  `reached=true`, zero C1/C2/C3, but a zero terminal bonus before the fix in
  `outputs/teacher_diagnostics/seed41_model438_scene71_success_reward_predicate_probe/eval_summary.json`.
  After disabling only the inactive C2/C3 predicates (C1 remains active), the
  identical trajectory receives the intended +200 step contribution and a
  total terminal reward of 199.88; see
  `outputs/teacher_diagnostics/seed41_model438_scene71_success_reward_fixed_probe/eval_summary.json`.
  Episode-end traces now persist the weighted reward-term breakdown and the
  individual success predicates, making this contract directly auditable.
- Reward-fixed v23 continues the waypoint-free +/-90-degree curriculum from
  v20 `model_550` with 1,024 environments.  Its online run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/2oflspoi`; balanced
  held-out evaluation is scheduled at `model_{600,650,700,749}`.  `model_600`
  obtains 60.94% pose / 49.22% constrained success with 21.09% C1;
  `model_650` obtains 53.91% / 51.56% with 14.84% C1.  Both are rejected, but
  the decline in C1 shows that the repaired sparse terminal reward is active
  and the remaining issue is safe directional coverage.  `model_700` improves
  to 68.75% pose / 66.41% constrained with 9.38% C1, but remains 0/16 in the
  +70 to +90 degree bin and 3/13 in the -90 to -70 degree bin.  No clutter
  transfer checkpoint will be selected until this direction gate passes.
- The explicit v27 goal-side ablation adds a continuous object-centric set
  potential, computed from the safe-point subset and remaining goal
  displacement.  It selects the trailing-side safe surface band online and
  never introduces a world-frame waypoint or changes the 4,141-value actor
  observation.  The baseline T0 task is unchanged.  A four-environment real
  Isaac/PPO update passes, all 63 unit tests plus 7 subtests pass.  The full
  1,024-environment 24-update transfer from v23 `model_700` is nevertheless
  rejected: `model_724` obtains 70.31% pose success, 60.16% constrained
  success, and 17.97% C1 on the balanced 128-scene set.  The +70 to +90 degree
  bin remains 0%, while the apparent improvement in the -90 to -70 degree bin
  is mostly unsafe (46.2% pose versus 15.4% constrained, 76.9% C1).  Artifact:
  `outputs/teacher_eval/seed41_v27r2_dir90_goalside_model724_balanced128/eval_summary.json`.
  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/lm2dnf94`.
- v28 therefore changes the training distribution rather than the reward: it
  filters the stable direction-randomized manifest to 128 scenes whose planar
  goal direction is 45 to 90 degrees away from the forward axis in either
  sign.  This is still direct goal-conditioned control, not a waypoint.  The
  1,024-environment 50-update transfer from v23 `model_700` is rejected on the
  disjoint balanced full-direction set: `model_749` obtains 69.53% pose but
  only 48.44% constrained success with 29.69% C1.  The -90 to -70 degree bin
  is 38.5% pose / 15.4% constrained / 84.6% C1, and the +70 to +90 degree bin
  remains effectively unsolved at 6.2% pose / 0% constrained.  Focused
  sampling therefore discovers unsafe protected-part shortcuts and does not
  justify a full-direction replay or clutter transfer.  The next diagnostic
  is an IK/contact-clearance audit of the endpoint-direction safe contact
  poses.  Artifact:
  `outputs/teacher_eval/seed41_v28_dir90_outer_model749_balanced128/eval_summary.json`.
  The focused manifest is
  `data/manifests/teacher_direction_endpoint_v16/hammer_teacher_dir90_outer_abs45_train128_seed10837.jsonl`.
  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/4zufllsh`.
- A semantic hand-cloud plus joint-limit reachability audit rules out endpoint
  infeasibility as the explanation for v28.  On all 29 held-out scenes with
  absolute direction at least 70 degrees, the search finds a sampled hand pose
  that contacts the trailing safe subset while maintaining more than 10 mm
  clearance to every non-safe target point; all 29 poses have a Franka IK
  solution within 1 mm / 0.01 rad.  However, a 41-step direct joint
  interpolation from reset to that legal IK pose is semantic-C1-free in only
  14/29 scenes: 13/13 negative endpoints but just 1/16 positive endpoints.
  The endpoint is reachable, but the robot must route around target forbidden
  geometry.  This remains a necessary-condition hand-cloud audit, not a PhysX
  collision certificate for proximal links.  Artifacts:
  `outputs/teacher_diagnostics/dir90_endpoint_safe_contact_ik_audit_v29.json`
  and
  `outputs/teacher_diagnostics/dir90_endpoint_safe_contact_ik_path_audit_v29r1.json`;
  reproducible script: `scripts/audit_teacher_endpoint_reachability.py`.
- v30 therefore combines the object-centric goal-side set potential with the
  complete non-terminating C1-soft cost (`-25` illegal-contact weight) on the
  focused endpoint manifest.  It still uses direct joint actions and the same
  4,141-value actor contract; no waypoint or privileged actor input is added.
  The 1,024-environment transfer starts again from v23 `model_700`, rather
  than inheriting v28's unsafe shortcut.  It is early-stopped after 19/50
  updates: hard-C1 `model_710` has only 46.09% constrained success and 2.34%
  C1, with 0/13 and 0/16 successes in the two endpoint bins.  Strong safety
  refinement made the policy conservative without discovering a new contact
  side.  Inspection shows that the original arbitrary-nearest-safe progress
  (weight 12) still competed with the added goal-side progress (weight 8),
  while the behavior-preserving runner capped exploration at 0.02.  Artifact:
  `outputs/teacher_eval/seed41_v30_goalside_c1soft_endpoint_hardc1_model710_balanced128/eval_summary.json`.
  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/7mehd9mw`.
- v31 removes that conflict rather than adding another reward: the general
  nearest-safe distance/progress terms are disabled during endpoint discovery,
  so the single active approach set is computed from goal direction and safe
  semantics.  The complete soft C1 terms remain active, and exploration is
  bounded at 0.10 (between v30's 0.02 refinement and v28's 0.40).  The actor,
  success predicate, direct actions, and absence of waypoints are unchanged.
  A real four-environment PPO update and all 63 unit tests plus 7 subtests
  pass.  The run is nevertheless early-stopped and rejected.  Hard-C1
  `model_710` obtains 49.22% constrained success with 9.38% C1;
  `model_720` improves to 73/128 = 57.03% constrained success with 6/128 =
  4.69% C1, but the positive endpoint remains 0/16.  Removing the conflicting
  objective helps middle directions but still rewards a Euclidean shortcut
  whose route intersects forbidden geometry.  Artifacts:
  `outputs/teacher_eval/seed41_v31_goalside_c1explore_endpoint_hardc1_model{710,720}_balanced128/`.
  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/j7c1z2du`.
- v32 introduces a waypoint-free semantic corridor navigation potential.  It
  selects the goal-compatible safe surface from the oracle point cloud, samples
  only the open route toward it, morphologically inflates all non-safe target
  points by a 30 mm hand-sweep radius, and optimizes distance plus corridor
  obstruction as one scalar.  No route point or new privilege enters the
  4,141-value actor observation.  The far-field distance remains linear, so
  the reset state retains a reaching gradient.  On the balanced 128-scene
  reset audit, the -70 to -90 degree endpoint bin has 0% obstruction whereas
  the +35 to +90 degree bins have 100%; mean potential is about 1.25 versus
  2.16 at the positive endpoint.  Artifact:
  `outputs/teacher_diagnostics/dir90_semantic_corridor_reset_audit_v32r1.json`;
  reproducible script: `scripts/audit_teacher_semantic_corridor.py`.
  A real Isaac smoke test and all 64 unit tests plus 7 subtests pass.  The
  1,024-environment v32r1 transfer is paused and rejected at `model_710`:
  hard-C1 full-direction evaluation obtains 87/128 = 67.97% constrained
  success and 12/128 = 9.38% C1, with 0/16 success and 4/16 C1 in the positive
  endpoint bin.  The linear obstruction saturates near the constraint, so
  direct-distance progress can still pay the unsafe shortcut.  Artifact:
  `outputs/teacher_eval/seed41_v32r1_semanticcorridor_endpoint_hardc1_model710_balanced128/eval_summary.json`.
  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/xfbh3jc2`.
- v33 preserves the same point-cloud corridor and actor contract, but replaces
  only the saturated linear obstruction with a finite log barrier.  At the
  reset state, mean potential remains about 1.24 in the unobstructed negative
  endpoint and rises to 4.22 in the positive endpoint; 13/16 positive endpoint
  scenes already lie within the inflated route's 10 mm contact-risk margin.
  The barrier is finite at contact (free-fraction floor 0.05), so PPO remains
  numerically stable, while detouring can reduce substantially more cost than
  a blocked straight approach can gain from Euclidean progress.  Artifact:
  `outputs/teacher_diagnostics/dir90_semantic_corridor_logbarrier_reset_audit_v33.json`.
  The bounded 1,024-environment 10-update probe is rejected.  Its final
  `model_709` obtains only 27/128 = 21.09% constrained success with 78/128 =
  60.94% C1 on the balanced hard-C1 set.  A finite barrier still saturates
  after the route enters collision, and the scalar straight-line objective
  supplies no side-selection gradient.  This is a reward-geometry failure,
  not evidence that a longer continuation is warranted.  Artifact:
  `outputs/teacher_eval/seed41_v33_semanticcorridor_logbarrier_hardc1_model709_balanced128/eval_summary.json`.
  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/lc3rkl7v`.
- v34 replaces the blocked straight segment with a small semantic visibility
  graph: one direct route and 12 support-ring detours are constructed online
  from all non-safe target points, segment clearance is evaluated after a
  30 mm hand-sweep inflation, and an illegal route can never beat an available
  legal detour.  Only the selected scalar geodesic potential shapes PPO; the
  actor still receives the unchanged 4,141 values and no waypoint/route label.
  The balanced reset audit selects a legal route in 128/128 scenes.  It uses a
  detour in 100% of +35 to +90 degree scenes, exactly where the direct route is
  obstructed, while retaining the direct route in the unobstructed negative
  endpoint bin.  All 66 tests plus 7 subtests pass.  Artifact:
  `outputs/teacher_diagnostics/dir90_semantic_geodesic_reset_audit_v34.json`;
  reproducible script: `scripts/audit_teacher_semantic_geodesic.py`.
  The focused 1,024-environment probe is nevertheless rejected: hard-C1
  `model_705` and `model_709` obtain only 51.56% / 36.72% C1 and 52.34% /
  35.16% C1 constrained-success / violation rates, respectively, with 0/16
  positive-endpoint success.  A selected scalar route cost does not tell PPO
  which local displacement realizes the detour.  Artifacts:
  `outputs/teacher_eval/seed41_v34_semanticgeodesic_hardc1_model{705,709}_balanced128/`;
  online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/axs3krzs`.
- v35 tests whether v34 failed only because of aggressive adaptation.  It
  widens the scalar-progress normalization, caps exploration at 0.05, uses
  conservative PPO, and trains on a 512-scene 50/50 mixture of the full
  direction set and endpoint replay.  Final `model_719` recovers to 84/128 =
  65.62% constrained success with 7/128 = 5.47% C1, but the +70 to +90 degree
  bin is still 0/16.  Intermediate `model_715` is worse at 58.59% / 14.84%
  constrained success / C1.  The scalar-geodesic line is therefore rejected
  rather than extended.  Artifacts:
  `outputs/teacher_eval/seed41_v35_semanticgeodesic_conservative_hardc1_model{705,715,719}_balanced128/`;
  online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/vwjnt3so`.
- v36 adds a reward-only local semantic vector field while preserving the
  4,141-value actor contract.  At each state, the field is the first free-space
  direction of the currently shortest legal point-cloud route; the signed
  reward measures actual hand displacement along the previous field and is
  active only while the direct route is blocked.  This is live state feedback,
  not a stored scene waypoint, and neither the route index nor field direction
  enters the actor.  On the 128-scene reset audit, unobstructed negative bins
  align with the direct approach (mean cosine approximately 1.0), while every
  selected detour has at least 0.10 lateral fraction.  The +35 to +90 degree
  bins select legal detours in every scene and have mean lateral fractions of
  0.99 or greater.  All 67 regression tests pass.  Artifact:
  `outputs/teacher_diagnostics/dir90_semantic_vector_field_reset_audit_v36.json`;
  online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/7z496t0u`.
  Geometry alone is not acceptance: hard-C1 `model_705` obtains 86/128 =
  67.19% constrained success with 10/128 = 7.81% C1, and `model_710`
  obtains 85/128 = 66.41% with 12/128 = 9.38% C1.  Both remain 0/16 at the
  positive endpoint.  The logged vector-field contribution is only about one
  quarter of scalar geodesic progress, while exploration is capped at 0.05;
  v36 is rejected and stopped at `model_710`.  Artifacts:
  `outputs/teacher_eval/seed41_v36_semanticvectorfield_hardc1_model{705,710}_balanced128/`.
- v37 removes scalar geodesic progress, raises the vector-field weight from 8
  to 40, halves its displacement normalization to 5 mm, and increases bounded
  exploration from 0.05 to 0.15.  The logged field contribution rises from
  about 0.009 to 0.071 without numerical instability, but hard-C1
  `model_705` remains 0/16 at the positive endpoint despite 67.97% overall
  constrained success and 4.69% C1.  Final `model_709` regresses to 65.62% /
  9.38% and also remains 0/16.  Reward scale and action exploration are thus
  not the missing mechanism; v37 is rejected.  Artifacts:
  `outputs/teacher_eval/seed41_v37_semanticvectorfield_strongexplore_hardc1_model{705,709}_balanced128/`;
  online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/u2z3b1tj`.
- v38 keeps the exact 4,141-value actor observation and per-point external
  contract, but adds zero-initialized residual relation encoders inside the
  policy.  For each target point they deterministically derive point-to-hand,
  object-local, and goal-conditioned trailing-side features from quantities
  already present in the observation; clutter points receive point-to-hand and
  point-to-target-center relations.  Old `model_700` loads with bitwise-equivalent
  deterministic actions, while the new residual receives non-zero gradients.
  No route, field direction, waypoint, simulator contact, or physics parameter
  enters the actor.  All 69 regression tests pass.
  Its held-out hard-C1 result rejects the short residual continuation:
  `model_705` and `model_710` both obtain 74/128 = 57.81% constrained success,
  with 25.00% and 28.12% C1 respectively; both remain 0/16 in the +70 to +90
  degree endpoint.  Artifacts:
  `outputs/teacher_eval/seed41_v38_relationvectorfield_hardc1_model{705,710}_balanced128/`;
  online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/ucdq1g7g`.
- A rendered v23 `model_700` positive-endpoint failure shows the actual local
  minimum rather than an unreachable reset: the hand descends to 26.8 mm from
  the target, stops in front of the protected head with zero C1, and leaves the
  hammer at essentially its initial 87 mm planar error.  Artifact:
  `outputs/teacher_demos/seed41_v23_model700_positive_endpoint_scene0023/seed41_v23_model700_positive_endpoint_scene0023-step-0.mp4`.
- v39 removes the residual-continuation confound.  It jointly trains the
  relation encoder and policy from random initialization with the original
  Euclidean safe-region reward, 1,024 environments, and the same 50/50
  full-direction/endpoint replay.  Its online run is
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/7m8i1ay3`.
  The first v39 deterministic hard-C1 checkpoint, `model_25`, is intentionally
  not promoted: 0/128 success and 0/128 C1.  It reaches 60.7 mm finger-target
  clearance and the target relation residual is already non-zero (final-layer
  norm 0.286), proving that this is an early-policy limitation rather than a
  disconnected relation branch.  Artifact:
  `outputs/teacher_eval/seed41_v39_relation_fromscratch_hardc1_model25_balanced128/eval_summary.json`.
  `model_50` also remains an unselected early checkpoint at 0/128 strict and
  0/128 C1; it reaches 48.6 mm minimum finger-target distance and produces
  target motion, but has not yet formed deterministic legal contact.  Artifact:
  `outputs/teacher_eval/seed41_v39_relation_fromscratch_hardc1_model50_balanced128/eval_summary.json`.
  `model_75` remains 0/128, while `model_100` obtains the first 1/128 strict
  deterministic success with zero C1; the positive endpoint is still 0/16.
  These are learning-progress checkpoints, not C1 selections.  Artifacts:
  `outputs/teacher_eval/seed41_v39_relation_fromscratch_hardc1_model{75,100}_balanced128/`.
  `model_150` reaches 2/128 = 1.56% constrained success with zero C1 but remains
  0/16 at the positive endpoint, so it also fails both selection gates.  The
  completed run never resolves the endpoint: its best scheduled aggregate is
  `model_250` at 10/128 = 7.81% constrained success with zero C1, while final
  `model_299` regresses to 5/128 = 3.91%; both remain 0/16 in `[70,90]`.
  The original Euclidean reward is therefore a completed negative control,
  not an under-trained candidate.  Artifacts:
  `outputs/teacher_eval/seed41_v39_relation_fromscratch_hardc1_model{250,299}_balanced128/`.
  The
  complete live direction-bin summary is
  `outputs/teacher_eval/dir90_relation_euclidean_v39_live.json`.
- The original v40 semantic-vector run is a rejected, confounded comparison:
  it inherited a `-25` forbidden-contact weight while v39 used the intended
  exploratory `-5` weight.  Its `model_25` has 0/128 success and 13/128 C1,
  but that result cannot be attributed to the vector field.  The run is kept
  only for provenance at
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/1eopjkhl` and must
  not be used in the ablation table.
- v40r1 is the corrected controlled counterpart.  It changes only the
  pre-contact approach shaping from local Euclidean attraction to the
  reward-only semantic free-space vector potential while matching v39's
  actor observation, relation network, seed, 1,024 environments, manifest,
  PPO settings, and `-5` soft-C1 contact weight.  The hard-C1 watcher evaluates
  the same balanced 128-episode profile at checkpoints 25, 50, 75, 100, 125,
  and 149.  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/8trid98y`.
  Its first hard-C1 checkpoint confirms the missing-direct-signal concern:
  `model_25` is 0/128 strict and 0/128 C1 but remains 251.4 mm from the target,
  i.e. its deterministic mean policy has not learned to approach.  Artifact:
  `outputs/teacher_eval/seed41_v40r1_relation_semanticvector_matchedsoftc1_hardc1_model25_balanced128/eval_summary.json`.
  `model_50` is also 0/128 with zero C1 and 0/16 at the positive endpoint;
  its 80.1 mm mean terminal planar error is essentially the reset error.
  Training and its watcher are stopped at this checkpoint, with all artifacts
  retained as a rejected detour-only ablation:
  `outputs/teacher_eval/seed41_v40r1_relation_semanticvector_matchedsoftc1_hardc1_model50_balanced128/eval_summary.json`.
- The post-launch reward audit found one remaining asymmetry in v40r1: its
  local field is active only for detours, so unobstructed scenes lose v39's
  signed approach-progress term.  The independently registered balanced-field
  variant fixes this without changing v40r1 in place.  With global weight 40,
  5 mm normalization, and direct-route scale 0.15, its direct progress slope
  is exactly `40 * 0.15 / 0.005 = 1200`, matching v39's
  `12 / 0.010 = 1200`; detours retain the stronger lateral signal.  A unit
  test covers detour/direct scaling and the legacy default remains zero.  A
  one-environment real Isaac Lab smoke completed one PPO update with the
  intended 4,141/4,146 observation split and 12 reward terms at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_05-02-30_seed41_t0_relation_balancedfield_config_smoke_v42`.
  The repository task suite passes 76 tests plus 7 subtests after the v47
  illegal-route recovery regression is added.
  The 1,024-environment v42 controlled run and identical hard-C1 watcher are
  active; online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/atni3o1y`.
  Its first hard-C1 checkpoint, `model_25`, is 0/128 strict and 0/128 C1, but
  already reduces mean terminal planar error to 65.5 mm.  This is earlier
  object-motion learning than the Euclidean run, accompanied by a still-large
  0.456 rad mean rotation error; the positive endpoint remains 0/16.  It is
  therefore retained through the scheduled checkpoint grid but not selected.
  Artifact:
  `outputs/teacher_eval/seed41_v42_relation_balancedfield_matchedsoftc1_hardc1_model25_balanced128/eval_summary.json`.
  At `model_50` it regresses to 0/128 with 3/128 protected-hand C1 events;
  all three occur in the positive endpoint (3/16), whose mean planar and
  rotation errors worsen to 164.7 mm and 0.698 rad.  This confirms that merely
  extending the discontinuous field can turn the boundary oscillation into an
  unsafe push.  v42 is retained only through `model_75` for the scheduled
  trend check.  Artifact:
  `outputs/teacher_eval/seed41_v42_relation_balancedfield_matchedsoftc1_hardc1_model50_balanced128/eval_summary.json`.
  The scheduled `model_75` audit is again 0/128 constrained success with zero
  C1 and a 92.3 mm minimum finger-to-target distance, so training and its
  watcher are stopped rather than spending more samples on a structural
  reward discontinuity.  Artifact:
  `outputs/teacher_eval/seed41_v42_relation_balancedfield_matchedsoftc1_hardc1_model75_balanced128/eval_summary.json`.
- A rendered v42 `model_25` positive-endpoint rollout resolves its early
  failure mechanism: the hand moves from 263 mm to 31.7 mm from the target,
  reaches the safe-handle side without C1, then retreats through 42.8, 55.1,
  and 73.7 mm while the hammer moves less than 1 mm.  This coincides with the
  discrete route transition where the balanced field drops from detour scale
  1.0 to direct scale 0.15.  Video and sidecar:
  `outputs/teacher_demos/seed41_v42_model25_positive_endpoint_scene0127/seed41_v42_model25_positive_endpoint_scene0127-step-0.mp4` and
  `outputs/teacher_eval/seed41_v42_model25_positive_endpoint_scene0127/eval_summary.json`.
- v43 tested latching detour-strength guidance through the final direct edge.
  Its real one-environment PPO smoke passes at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_05-23-18_seed41_t0_relation_committedfield_config_smoke_v43`.
  A subsequent contract audit rejected the design: the per-episode latch is
  history that cannot be reconstructed from the actor's current semantic
  point cloud, so identical actor observations can receive different reward
  scales.  The 1,024-environment run and watcher were therefore stopped early,
  with provenance retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_05-24-50_seed41_t0_relation_committedfield_matchedsoftc1_dir90_fullendpoint_fromscratch_v43`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/fqgxuxdn`.
- v44 replaces that hidden latch with a continuous scale derived only from
  the current straight-route clearance to the non-safe semantic cloud.  The
  scale is 1.0 at or below 10 mm clearance, falls linearly through 0.575 at
  25 mm, and reaches the matched direct baseline 0.15 at 40 mm.  Thus the
  strong field persists across the geometrically tight final approach but
  fades smoothly in genuinely open space.  It adds no waypoint, route label,
  phase bit, or actor feature; the clearance is recoverable from the existing
  `[x,y,z,safe,protected]` target cloud and hand state.  A real one-environment
  PPO smoke completed successfully with the unchanged 4,141/4,146 actor/critic
  contract at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_05-33-41_seed41_t0_relation_clearanceblend_config_smoke_v44`.
  The matched 1,024-environment from-scratch run and held-out hard-C1 watcher
  are active at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_05-34-43_seed41_t0_relation_clearanceblend_matchedsoftc1_dir90_fullendpoint_fromscratch_v44`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/hfloxogr`.
  A live 512-scene reset-state audit verifies that the blend is applied to the
  intended geometry rather than only matching its unit-test formula.  The
  `[70,90]` endpoint has 100% direct obstruction and detour selection, 100%
  legal selected routes, mean route scale 0.934, and mean lateral field
  fraction 0.993.  In contrast, the two unobstructed negative bins use the
  matched 0.15 baseline on every scene.  This is a reset-geometry check, not a
  learned-trajectory claim; checkpoint evaluation remains the selection
  evidence.  Artifact:
  `outputs/teacher_diagnostics/dir90_fullendpoint_clearanceblend_reset_audit_v44.json`.
  The first learned checkpoint nevertheless rejects v44 as a complete fix:
  `model_25` obtains 0/128 constrained success with 39/128 = 30.47% C1
  (14 neutral-hand and 25 protected-hand episodes).  On +75.47-degree scene
  127 it reaches 33.6 mm from the target, then terminates at step 66 on a
  neutral-region C1 event.  The trace identifies the missing branch: by step
  30 every sampled route has zero clearance, even though the hand center has
  not yet crossed the local body-inflation test, and the old finite fallback
  marks the direct illegal edge as non-detour.  Continuous reward strength
  cannot make that direction safe.  Training and its remaining watcher are
  stopped at `model_25` rather
  than treating unsafe early object motion as progress.  Artifacts:
  `outputs/teacher_eval/seed41_v44_relation_clearanceblend_matchedsoftc1_hardc1_model25_balanced128/eval_summary.json`,
  `outputs/teacher_eval/seed41_v44_model25_positive_endpoint_scene0127/eval_summary.json`,
  and
  `outputs/teacher_demos/seed41_v44_model25_positive_endpoint_scene0127/seed41_v44_model25_positive_endpoint_scene0127-step-0.mp4`.
- v45 first tried that recovery only after the current hand center itself
  entered the inflated semantic cloud.  Exact replay of the v44 failure shows
  this gate is too late: at step 30 every sampled route is illegal, but
  recovery remains false and the field is exactly aligned with the illegal
  direct edge; recovery activates only at step 60.  The corresponding
  1,024-environment start was therefore stopped after iteration 5 and is not a
  candidate teacher.  Probe artifact:
  `outputs/teacher_eval/seed41_v44_model25_scene0127_v45_recoveryfield_probe/eval_summary.json`.
- v46 removes the late proximity gate.  Whenever no sampled route is legal,
  its reward direction becomes the normalized outward clearance gradient.
  On the identical held-out +75.47-degree scene and old v44 policy, step 30
  reports `recovery_active=true` and changes field/direct alignment from
  `+1.000` to `-0.938`.  This fixes the illegal shortcut but creates a second
  attractor: learned `model_25` is safe at 0/128 C1, yet remains 0/128 strict;
  on scene 127 it retreats from 153.5 mm at step 30 to 399.0 mm at step 299,
  while every sampled route stays illegal.  Pure radial recovery therefore
  cannot restore lateral feasibility.  Training and its watcher are stopped
  at `model_25`.  Its one-environment configuration smoke is retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_06-04-18_seed41_t0_relation_clearancerecovery_allillegal_config_smoke_v46`.
  The rejected 1,024-environment run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_06-05-29_seed41_t0_relation_clearancerecovery_allillegal_matchedsoftc1_dir90_fullendpoint_fromscratch_v46`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/e3wrtdq8`.
  Artifacts:
  `outputs/teacher_eval/seed41_v46_relation_clearancerecovery_allillegal_matchedsoftc1_hardc1_model25_balanced128/eval_summary.json`,
  `outputs/teacher_eval/seed41_v46_model25_positive_endpoint_scene0127_trace/eval_summary.json`, and
  `outputs/teacher_eval/seed41_v44_model25_heldout_scene0127_v46_allillegal_recoveryfield_probe/eval_summary.json`.
- v47 retains lexicographic safety but adds an observation-derived boundary
  tangent whenever the legal route set is empty.  The tangent lies in the
  null space of the outward direction, so it cannot cancel clearance recovery;
  a stable `cross(world_up, direct_route)` side convention prevents the
  left/right switching of an unordered symmetric candidate set.  On the
  exact v46 scene-127 trajectory, recovery has outward alignment 0.447 and
  tangential fraction 0.894 instead of being purely radial.  A simulator-free
  open-wall regression repeatedly integrates the current-state field and
  restores a legal route in 18 steps while going around rather than through
  the wall.  The full suite passes 76 tests plus 7 subtests.  Exact field
  artifact:
  `outputs/teacher_eval/seed41_v46_model25_scene0127_v47_tangential_recoveryfield_probe/eval_summary.json`.
  A real one-environment/one-iteration IsaacLab smoke also completes and
  writes `model_0.pt` at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_06-32-19_seed41_t0_relation_tangentialrecovery_config_smoke_v47`.
  The 1,024-environment seed-41 from-scratch run is retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_06-35-08_seed41_t0_relation_tangentialrecovery_matchedsoftc1_dir90_fullendpoint_fromscratch_v47`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/svlzaa5n`.
  A separate hard-C1 held-out watcher evaluated iterations 25 and 50 before
  rejection.  The deliberately early
  `model_25` is safe but not yet competent: 0/128 constrained success and
  0/128 C1.  Its mean terminal planar/rotation errors are 60.8 mm/0.556 rad.
  On held-out positive-endpoint scene 127 it approaches to 88.6 mm without C1,
  then retreats while recovery remains active; the measured recovery field
  preserves outward alignment 0.447 and tangential fraction 0.894, so this
  checkpoint has not yet learned to track the new field rather than exposing a
  radial-only field regression.  Artifacts:
  `outputs/teacher_eval/seed41_v47_relation_tangentialrecovery_matchedsoftc1_hardc1_model25_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed41_v47_model25_positive_endpoint_scene0127_trace/eval_summary.json`.
  `model_50` remains 0/128 constrained and 0/128 C1 while mean terminal planar
  error regresses from 60.8 to 75.9 mm even as training-time field reward
  rises.  This is evidence of circulation reward hacking, not insufficient
  training, so the run and watcher are stopped.  Artifact:
  `outputs/teacher_eval/seed41_v47_relation_tangentialrecovery_matchedsoftc1_hardc1_model50_balanced128/eval_summary.json`.
- v48 makes the v47 direction field potential-consistent.  It maps the current
  non-negative semantic navigation potential through `p / (p + 1)`, takes the
  exact transition difference, attenuates positive descent when displacement
  disagrees with the local field, and never attenuates potential ascent.  A
  closed state-space loop therefore has non-positive total shaping reward,
  while the actor observation remains exactly 4,141 values and receives no
  route or phase variable.  Two unit regressions cover the closed-loop bound
  and aligned descent; the full suite passes 78 tests.  A real PPO smoke
  completes at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_07-00-05_seed41_t0_relation_potentialconsistent_config_smoke_v48`.
  Cumulative per-term replay provides simulator evidence at the selected scale:
  the known scene-127 circulation receives -14.56 field shaping, whereas the
  known legal scene-71 success receives +7.47, reaches the strict pose, and has
  zero C1.  Artifacts:
  `outputs/teacher_diagnostics/seed41_v47_model25_scene0127_v48_cycle_safe_reward_replay_v2/eval_summary.json`
  and
  `outputs/teacher_diagnostics/seed41_v19_model438_success_scene0071_v48_reward_scale_replay/eval_summary.json`.
  The first 1,024-environment seed-41 from-scratch run is retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_07-05-53_seed41_t0_relation_potentialconsistent_matchedsoftc1_dir90_fullendpoint_fromscratch_v48`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/6yyj94ya`.
  It is stopped before `model_25` after a matched successful-trajectory audit
  shows that its -43.02 absolute geodesic cost is about 36 times the accepted
  Euclidean -1.18 cost, while its +7.47 progress is only one quarter of the
  accepted +28.15 progress.  This is a scale mismatch, not a geometry result.
  The Euclidean reference artifact is
  `outputs/teacher_diagnostics/seed41_v19_model438_success_scene0071_euclidean_reward_scale_replay/eval_summary.json`.
- v49 uses rounded, auditable replay-matched weights: -0.03 absolute geodesic
  cost and +3000 potential-consistent progress.  On the known success these
  correspond to approximately -1.29/+28.02, while the known circulation stays
  strongly negative at approximately -59 total semantic shaping.  A real
  PPO smoke confirms the intended weights and unchanged 4,141/4,146 contract
  at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_07-12-57_seed41_t0_relation_potentialconsistent_calibrated_config_smoke_v49`.
  The 1,024-environment seed-41 from-scratch run is retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_07-14-05_seed41_t0_relation_potentialconsistent_calibrated_matchedsoftc1_dir90_fullendpoint_fromscratch_v49`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/97m8hqpx`.
  The independent hard-C1 watcher evaluated balanced checkpoints through
  `model_100`.  The early `model_25` is 0/128 constrained and
  0/128 C1, with 85.7 mm/0.397 rad mean terminal planar/rotation error; its
  positive endpoint remains at approximately reset error.  It is safe but not
  competent.  Models 50/75/100 also remain 0/128 constrained and 0/128 C1;
  mean planar error improves only to 71.9/63.4/70.8 mm and is non-monotonic.
  The run was stopped after `model_100`, rather than mistaking the rising
  training-time safe-contact rate for held-out competence.  Artifacts:
  `outputs/teacher_eval/seed41_v49_relation_potentialconsistent_calibrated_matchedsoftc1_hardc1_model25_balanced128/eval_summary.json`.
  `outputs/teacher_eval/seed41_v49_relation_potentialconsistent_calibrated_matchedsoftc1_hardc1_model100_balanced128/eval_summary.json`.
- v50 replaces the weighted length-plus-clearance objective with a
  lexicographic feasibility potential.  If any route satisfies the audited
  8 mm C1 clearance, only shortest safe-contact route length is optimized;
  if none does, maximum clearance is restored before length can matter.  Thus
  extra clearance can never buy movement away from the handle.  Feasible
  costs are strictly below one and infeasible costs are at least one.  The
  transition term uses the PPO discount (`gamma=0.99`), and the actor/critic
  observation contract remains 4,141/4,146.  The route sweep proxy is reduced
  from a duplicated 3 cm spherical inflation to 2 cm; hard C1 itself remains
  unchanged and still uses the complete hand semantic cloud plus PhysX
  whole-arm contact.  Two simulator-independent regressions cover
  lexicographic route selection and discount-consistent cycle equivalence;
  the full suite passes 80 tests and seven subtests.  A real PPO smoke is at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_07-48-23_seed41_t0_relation_lexicographicpotential_config_smoke_v50`.
  On the identical v49 scene-127 retreat, the semantic return changes from
  +3.87 to -41.84 as the potential worsens from 0.560 to 0.624.  On the known
  legal scene-71 success, every sampled route is now feasible, the potential
  improves from 0.560 to 0.055, semantic return is +46.33, strict success is
  retained, and C1 remains zero.  Artifacts:
  `outputs/teacher_diagnostics/seed41_v49_model50_scene0127_v50_body20_clear8_lexicographic_reward_replay/eval_summary.json`
  and
  `outputs/teacher_diagnostics/seed41_v19_model438_success_scene0071_v50_body20_clear8_lexicographic_reward_replay/eval_summary.json`.
  The 1,024-environment seed-41 from-scratch run is retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_07-58-17_seed41_t0_relation_lexicographicpotential_body20_clear8_softc1_dir90_fullendpoint_fromscratch_v50`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/zaoridx8`.
  The independent hard-C1 watcher rejects this run at `model_75`: models
  25/50/75 all obtain 0/128 constrained success.  Model 50 improves mean
  terminal planar/rotation error to 65.0 mm/0.146 rad with zero C1, but model
  75 regresses to 77.9 mm/0.162 rad and 28.1% C1.  Training-time legal contact
  rises while safe-contact rotation progress remains negative, so longer
  training cannot repair the missing goal-wrench relation.  The run is
  stopped at model 75.  Artifacts:
  `outputs/teacher_eval/seed41_v50_relation_lexicographicpotential_body20_clear8_softc1_hardc1_model50_balanced128/eval_summary.json`
  and
  `outputs/teacher_eval/seed41_v50_relation_lexicographicpotential_body20_clear8_softc1_hardc1_model75_balanced128/eval_summary.json`.
- v51 keeps v50's lexicographic C1 feasibility and discount-consistent
  potential, but replaces the XY-only trailing contact subset with a
  goal-wrench-aware safe contact manifold.  For safe point offset `r` and the
  desired planar unit push `f`, the support score is the old trailing term
  plus `tanh(delta_yaw / 0.1) * (r x f)_z`.  Positive and negative yaw goals
  therefore select opposite moment arms without admitting protected points,
  adding a world waypoint, or changing the actor observation.  The actor's
  seventh recoverable relation feature uses the identical score; its external
  actor/critic contracts remain 4,141/4,146.  Rotation progress is raised from
  8 to 20 to match the strict XY/SO(3) tolerances rather than leaving the yaw
  objective an order of magnitude weaker.  Regressions verify mirrored yaw
  selection and exact translation-only fallback; the full suite passes 82
  tests and seven subtests.  A real one-environment PPO smoke confirms the
  intended 20 rotation weight and unchanged contracts at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_08-35-49_seed41_t0_relation_wrench_lexicographicpotential_config_smoke_v51`.
  The first COM-corrected 1,024-environment seed-41 from-scratch run is retained at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_08-40-01_seed41_t0_relation_wrenchcom_lexicographicpotential_body20_clear8_softc1_dir90_fullendpoint_fromscratch_v51`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/eviib1i1`.
  It is stopped during early reaching exploration after a mesh-level audit
  discovers that the inherited v17 manifest contains only negative goal-yaw
  deltas (-0.229 to -0.071 rad).  It is not a bidirectional-yaw result.
- v52 fixes the data contract without changing scene physics.  The v18 train
  manifest preserves every initial pose, goal XYZ, support face, object,
  obstacle, mass, and material from v17, while balancing yaw signs inside
  every planar-direction bin: 256 positive and 256 negative.  Its disjoint
  held-out set contains 64/64 and 8/8 in every one of eight direction bins.
  The actual 50,000-point DOMINO hammer audit shows that wrench-aware
  selection improves signed moment arm in every one of 512 scenes while
  selecting zero protected points.  Artifacts:
  `data/manifests/teacher_direction_biyaw_v18/hammer_teacher_dir90_biyaw_full_endpoint_balanced512_seed35051.summary.json`,
  `data/manifests/teacher_direction_biyaw_v18/hammer_teacher_dir90_biyaw_eval128_seed12839.summary.json`, and
  `outputs/teacher_diagnostics/v51_wrench_contact_manifold_biyaw_v18_actual_hammer_audit.json`.
  Evaluation CSVs now include signed initial/terminal yaw error and yaw
  progress ratio; checkpoint selection separately gates positive and negative
  yaw so aggregate success cannot hide a one-sign policy.  The retained
  1,024-environment seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_08-50-26_seed41_t0_relation_wrenchcom_lexicographicpotential_biyaw_body20_clear8_softc1_dir90_fullendpoint_fromscratch_v52`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/0vjhki3e`.
  It was stopped after the balanced hard-C1 `model_50` audit: 0/128 success,
  zero C1, 88.9 mm mean terminal XY error, and 0.183 rad mean terminal
  rotation error.  More importantly, the 64 negative-yaw scenes have mean yaw
  progress ratio +1.31 (overshoot), while the 64 positive-yaw scenes have
  -0.21 (motion in the wrong direction).  Thus v52 learned a one-sign policy;
  more iterations cannot be treated as a fix for the observation/reward
  conflict.  Artifact:
  `outputs/teacher_eval/seed41_v52_relation_wrenchcom_lexicographicpotential_biyaw_body20_clear8_softc1_hardc1_model50_balanced128/eval_summary.json`.
- v53 removes the safe-attraction/protected-repulsion dead point structurally
  rather than by another scalar-weight sweep.  Before contact, the
  lexicographic geodesic potential still selects a C1-feasible route to the
  safe handle.  Once any observable legal safe contact is present, its
  navigation cost and distance are gated to zero, so the policy stops chasing
  a moving contact anchor and only pose progress can reward continued pushing.
  Pose progress is now gated on *legal* safe contact, not raw safe proximity.
  The independent C1 clearance hinge is narrowed from 20 mm to 12 mm around
  the unchanged 10 mm hard semantic boundary, preserving a 2 mm learning band
  without repelling valid handle contact from far away.
  The actor's internal safe-point relation channels are also separated into
  local XYZ, trailing support, and signed goal-yaw moment arm; translation and
  rotation evidence can no longer cancel inside one scalar.  The external
  deployable actor/critic contracts remain exactly 4,141/4,146 and no waypoint,
  hidden phase, or hard C1 termination is added to training.  A real
  one-environment PPO smoke and a four-environment hard-C1 checkpoint-load
  smoke pass at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_09-31-08_seed41_t0_relation_wrenchseparated_contactgate_config_smoke_v53`
  and
  `outputs/teacher_smoke/seed41_v53_model0_hardc1_architecture/eval_summary.json`.
  The retained 1,024-environment, balanced +/-yaw, seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_09-33-50_seed41_t0_relation_wrenchseparated_contactgate_biyaw_band10_w15_narrowc1_fromscratch_v53`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/ofxuoktk`.
  It was stopped after the balanced hard-C1 `model_50` audit.  Success remains
  0/128 with zero C1; legal-safe-contact episodes fall from 30.47% at
  `model_25` to 17.19%, and the positive-yaw half still rotates the wrong way
  (mean yaw-progress ratio -0.76 versus +0.31 for negative yaw).  The run is
  therefore a rejected one-sign solution, not an under-trained checkpoint.
  Artifact:
  `outputs/teacher_eval/seed41_v53_relation_wrenchseparated_contactgate_biyaw_band10_w15_narrowc1_hardc1_direction_summary.json`.
  Matching low-noise soft-transfer tasks now preserve both the separated-wrench
  network and the complete v53 reward contract while isolating C2, C3, or
  enabling both in combined scenes.  Matching hard diagnostic tasks reuse the
  same network for typed termination evaluation.  One-environment weights-only
  Isaac/PPO smokes confirm the intended reward schemas: C2 has 14 terms with
  only protected--clutter costs, C3 has 14 terms with only robot--clutter
  costs, and combined has all 16 terms.  Artifacts:
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_09-40-49_seed41_v53_c2soft_transfer_weightsload_smoke`,
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_09-42-44_seed41_v53_c3soft_transfer_weightsload_smoke`, and
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_09-43-59_seed41_v53_combinedsoft_transfer_weightsload_smoke`.
- v54 removes the remaining reward-side single-contact assumption.  A
  50,000-point audit of the real DOMINO hammer shows that even normalized
  maximin contact selection cannot guarantee both positive translation
  support and the requested signed yaw moment for every goal: for some joint
  XY/yaw goals no safe handle point realizes that one-push wrench.  The dense
  approach objective now routes to the complete semantic safe set and gates
  to zero on legal contact.  The policy, which still observes the recoverable
  relative goal plus separate trailing/moment point relations, must choose and
  change contacts using the simultaneous XY/Z/SO(3) pose-progress reward.
  This adds no waypoint, prescribed contact, hidden phase, COM, or other
  privileged actor input.  A real Isaac/PPO smoke and all 85 tests pass at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_10-05-36_seed41_v54_fullsafe_contactgate_config_smoke`.
  The retained 1,024-environment seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_10-06-52_seed41_t0_relation_fullsafe_contactgate_biyaw_narrowc1_fromscratch_v54`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/jei2fgqz`.
  Balanced hard-C1 evaluations are scheduled at
  `model_{25,50,75,100,125,149}` and must pass separately for each yaw sign.
  At `model_25`, hard-C1 remains zero and legal safe contact rises to 94/128
  (73.44%); one scene reaches 8.45 mm terminal XY error.  It is not yet a
  valid teacher: success is 0/128 and the policy rotates in the same positive
  direction for both goal-yaw signs (mean yaw-progress +4.52 on negative-yaw
  scenes and -2.87 on positive-yaw scenes).  A same-state counterfactual audit
  reflects only the recoverable relative-goal yaw and changes the deterministic
  action by mean L2 0.103/0.378 for negative/positive yaw, so the yaw input and
  relation path are live; the early policy has selected the wrong physical
  contact mode rather than losing the observation.  Artifacts:
  `outputs/teacher_eval/seed41_v54_relation_fullsafe_contactgate_biyaw_narrowc1_hardc1_direction_summary.json` and
  `outputs/teacher_eval/seed41_v54_model25_counterfactual_yaw_hardc1_balanced128/eval_summary.json`.
  Real one-environment hard C2, C3, and combined runtime smokes also load the
  same 4,141/4,146 architecture and instantiate exactly the intended typed
  termination sets; artifacts:
  `outputs/teacher_smoke/seed41_v53_model0_hardc2_architecture/eval_summary.json`,
  `outputs/teacher_smoke/seed41_v54_model0_hardc3_architecture/eval_summary.json`, and
  `outputs/teacher_smoke/seed41_v54_model0_hardcombined_architecture/eval_summary.json`.
  v54 is stopped after `model_50`: all 128 held-out episodes reach legal safe
  contact with zero hard C1 and mean terminal XY improves to 61.1 mm, but only
  1/128 reaches the complete pose.  The one-sign mode persists (+3.53 yaw
  progress for negative goals, -3.28 for positive goals), so later v54
  checkpoints are not treated as a solution.
- v55 preserves the successful full-safe/contact-gate geometry and adds one
  bounded current-state cost over the smooth maximum of normalized XY, Z, and
  full SO(3) errors.  It is zero only at the joint pose goal and approaches one
  when any component is poor; this prevents a high yaw-only or XY-only score
  and, unlike positive tracking reward, does not pay a living bonus for holding
  the reset pose.  Shaping scales are 8 cm / 1 cm / 0.2 rad, while the strict
  success contract remains 2 cm / 1 cm / 0.1 rad plus five-step dwell.  A real
  one-environment PPO smoke passes with the intended 13-term reward schema at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_10-34-41_seed41_v55_fullsafe_jointposecost_config_smoke`.
  The 1,024-environment seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_10-35-46_seed41_t0_relation_fullsafe_jointposecost_biyaw_narrowc1_fromscratch_v55`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/5hm6gqy1`.
  v55 is rejected and stopped at `model_50`.  At `model_25`, balanced hard-C1
  evaluation has 0/128 success, 98.44% legal-safe-contact episodes, and zero
  C1; the negative/positive yaw halves end at 0.408/0.817 rad mean rotation
  error.  By `model_50`, success remains 0/128 and legal safe contact collapses
  to 37.50%.  The always-active pose cost penalizes states before the robot can
  affect the hammer and degrades the previously solved approach instead of
  resolving contact choice.  Artifacts:
  `outputs/teacher_eval/seed41_v55_relation_fullsafe_jointposecost_biyaw_narrowc1_hardc1_model25_balanced128/eval_summary.json` and
  `outputs/teacher_eval/seed41_v55_relation_fullsafe_jointposecost_biyaw_narrowc1_hardc1_model50_balanced128/eval_summary.json`.
- A simulator-level lateral-push audit separates physical feasibility from RL
  reward learning.  The same fixed-seed hammer reset is replayed with 17
  Cartesian paths spanning lateral displacement while measuring the current
  10 mm C1 semantic boundary.  Automatic success reset is disabled for this
  audit and five-step dwell is accumulated locally, so a successful trajectory
  cannot be truncated before its full physical envelope is measured.  All
  17/17 paths make safe contact, 0/17 touches forbidden, 9/17 enter strict
  XY/Z/SO(3), and 6/17 satisfy dwell.  The fully legal signed-yaw envelope is
  `[-0.4185, +0.1446] rad`, rejecting the hypothesis that the missing
  positive-yaw mode is physically impossible.  Navigation retirement and
  post-contact pose-cost activation latch on the same step for all 17/17
  trajectories.  The auditable trajectory fields include first
  safe/forbidden contact, signed yaw at each event, both reward-latch steps,
  and the fully legal signed-yaw envelope.  Artifact:
  `outputs/teacher_audit/v56_single_scene_lateral_push_c1contract_deterministic_sweep17.json`.
- v56 changes reward timing rather than adding another waypoint or scalar
  trade-off.  Pre-contact behavior is exactly the v54 full-safe route.  The
  first legal safe-contact event retires the strong vector-field term for the
  rest of that training episode; only then does the bounded joint XY/Z/SO(3)
  state cost activate, including while the hand changes contact side.  The
  latch is training-reward state only: it is not an actor input, policy state,
  action label, termination shortcut, or change to the 4,141/4,146 contract.
  A real one-environment PPO smoke instantiates the intended 13 reward terms at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_11-04-07_seed41_v56_postcontact_latch_config_smoke`,
  and the real scripted contact smoke triggers the post-contact path without
  forbidden contact.  The targeted suite passes 88 tests plus 7 subtests.  The
  1,024-environment seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_11-07-09_seed41_t0_relation_fullsafe_postcontactposecost_biyaw_narrowc1_fromscratch_v56`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/1uhoq4za`.
  v56 is rejected and stopped after `model_50`.  Hard-C1 success remains
  0/128 with zero C1, while legal-safe-contact episodes fall from 79.69% at
  model 25 to 46.88% at model 50 and training-time legal contact subsequently
  collapses to about 1.7%.  The absolute pose cost therefore improves some
  terminal yaw errors but makes entering the post-contact phase itself
  avoidable.  At model 50 the negative/positive yaw halves have 43.75%/50.00%
  legal contact and mean yaw progress +0.78/-1.32, so the wrong-sign mode also
  remains.  Artifact:
  `outputs/teacher_eval/seed41_v56_relation_fullsafe_postcontactposecost_biyaw_narrowc1_hardc1_direction_summary.json`.
- v57 removes that contact-entry reward cliff.  At first legal safe contact it
  records the bounded joint-pose cost as a zero-reward reference; later
  XY/Z/SO(3) improvement relative to that reference is positive and regression
  is negative.  Thus refusing contact cannot avoid an absolute pose cost, while
  reaching, terminal thresholds, actor/critic observations, actions, and the
  no-waypoint contract remain unchanged.  A real one-environment PPO smoke
  instantiates the intended 13 terms at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_11-32-25_seed41_v57_postcontact_improvement_config_smoke`.
  A fixed-seed 17-path simulator audit obtains 17/17 legal safe contacts,
  0/17 forbidden contacts, and synchronized navigation/improvement latches on
  all 17 paths; every captured contact reference cost is finite.  Artifact:
  `outputs/teacher_audit/v57_single_scene_lateral_push_contactrelative_deterministic_sweep17.json`.
  The repository `tests/` suite passes 89 tests plus 7 subtests; two tests
  outside that suite still depend on the unrelated missing hard-coded legacy
  ICP checkpoint `/home/steve/corn/ckpts/512-32-balanced-SAM-wd-5e-05-920`.
  The 1,024-environment seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_11-33-38_seed41_t0_relation_fullsafe_postcontactimprovement_biyaw_narrowc1_fromscratch_v57`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/qidilkwj`.
  Balanced hard-C1 evaluation uses independent yaw-sign gates.  At
  `model_25`, success is still 0/128 with zero C1, but legal safe contact is
  82.03% overall and 79.69%/84.38% for negative/positive yaw, so the v56
  contact-avoidance collapse is absent.  Mean terminal rotation error is
  0.224/0.261 rad for negative/positive yaw.  Positive-yaw progress remains
  wrong-sign at -1.11, however, so this checkpoint is improved but not a
  selectable teacher.  Artifact:
  `outputs/teacher_eval/seed41_v57_relation_fullsafe_postcontactimprovement_biyaw_narrowc1_hardc1_direction_summary.json`.
  v57 is rejected and stopped after `model_50`: hard-C1 success remains 0/128
  with zero C1 and legal safe contact rises to 87.50%, but the mean terminal XY
  error is still 7.88 cm for both yaw signs.  Negative/positive yaw progress is
  +1.28/-0.38, respectively.  Removing the contact-entry cliff therefore
  preserves contact and improves one sign, but a full-safe pre-contact target
  still does not determine the force-producing contact side.
- v58 makes contact-side selection lexicographic instead of adding translation
  and yaw support into one cancellable score.  While the current recoverable
  yaw error exceeds 0.02 rad, it retains every safe point within 10 mm of the
  best signed yaw-compatible moment arm; below that threshold it falls back to
  the translation-support safe set.  The navigation field still retires at
  first legal contact, the actor observation remains 4,141-D, and strict
  success still jointly requires XY/Z/SO(3)+dwell.  A 50,000-point audit of the
  actual DOMINO hammer over all 512 balanced scenes selects no protected point,
  improves signed moment in 512/512 scenes, and keeps a positive worst-case
  yaw-compatible moment of 6.4 mm.  Artifact:
  `outputs/teacher_diagnostics/v58_yawfirst_contact_manifold_biyaw_v18_actual_hammer_band10mm_audit.json`.
  The pure metric/config suite passes 89 tests plus 7 subtests, and a real
  one-environment PPO smoke instantiates the intended 13 reward terms at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_11-55-36_seed41_v58_yawcompatible_postcontact_improvement_config_smoke`.
  The active 1,024-environment seed-41 from-scratch run is
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_11-57-55_seed41_t0_relation_yawcompatible_postcontactimprovement_biyaw_narrowc1_fromscratch_v58`;
  online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/qj2l88jn`.
  A GPU-4 watcher performs balanced 128-episode hard-C1 evaluation at
  `model_{25,50,75,100,125,149}` before any checkpoint can advance to clutter.
  At `model_25`, v58 has 0/128 success and zero C1 but only 9.38% legal
  safe-contact episodes.  Negative/positive yaw contact is 14.06%/4.69% and
  mean yaw progress is +1.48/-1.47.  The near-best set therefore preserves the
  same wrong-sign mode while making reaching substantially less balanced than
  v57's 82.03% model-25 contact rate.  Artifact:
  `outputs/teacher_eval/seed41_v58_relation_yawcompatible_postcontactimprovement_biyaw_narrowc1_hardc1_direction_summary.json`.
- v59 retains all safe points whose correct-sign yaw moment exceeds 2 mm,
  rather than only the 10 mm near-best band.  On the actual 50,000-point
  DOMINO hammer this selects 20.3--71.4% of the safe handle per scene (46.2%
  mean), requires zero fallback across all 512 balanced scenes, and selects
  zero protected points.  Mean selected signed moment is positive for every
  scene and for each yaw sign.  Artifact:
  `outputs/teacher_diagnostics/v59_yawpositive_contact_manifold_biyaw_v18_actual_hammer_floor2mm_audit.json`.
  The repository suite passes 90 tests plus 7 subtests, and a real
  one-environment PPO smoke instantiates all 13 terms at
  `logs/rsl_rl/franka_affordance_teacher_seed41/2026-08-25_12-13-09_seed41_v59_yawpositive_postcontact_improvement_config_smoke`.
- Relation-preserving low-noise soft-refine and hard runners are registered
  for C1, C2, C3, clutter, and combined transfer.  A real one-environment
  combined runtime smoke loads a relation checkpoint, instantiates all six
  termination terms (including typed C1/C2/C3), and completes evaluation.
  Artifact: `outputs/teacher_smoke/relation_combined_runtime_v41/eval_summary.json`.
- A deliberately short direct-hard probe v26 starts from v23 `model_650` and
  is rejected after deterministic dense screening.  After only two updates,
  `model_652` falls to 25/64 = 39.06% constrained success with 22/64 = 34.38%
  C1 on the hard held-out task.  Training and its remaining watcher were
  stopped rather than assuming that longer hard training would recover it.
  The next safety adaptation therefore follows the specified C1-soft-refine
  stage before any hard continuation.  Artifact:
  `outputs/teacher_eval/seed41_v26_dir90_hardc1_screen_model652_balanced64/eval_summary.json`;
  online run: `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/flht4854`.
- The other two independently initialized forward policies now continue the
  same reward-fixed curriculum rather than reporting only one seed.  Seed 17
  v24 and seed 23 v25 each use 1,024 environments and start at +/-45 degrees;
  both trace back to their own v7 from-scratch `model_300` rather than copying
  seed 41.  Their online runs are
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/a38r6fr5` and
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/r9crneop`.
  Independent balanced hard-C1 evaluation is scheduled at
  `model_{350,400,450,499}` for both seeds before either advances to +/-90.
  The first `model_350` checkpoints obtain 53.91% constrained / 13.28% C1
  for seed 17 and 47.66% / 15.62% for seed 23, so neither is selected yet.
  At `model_450`, seed 17 improves to 68.75% constrained / 9.38% C1, whereas
  seed 23 regresses to 46.09% / 28.12%.  The diverging trajectories require
  per-seed held-out checkpoint selection; a shared final iteration is not a
  valid three-seed result.  Their final `model_499` checkpoints obtain 66.41%
  constrained / 23.44% C1 and 67.97% / 18.75%, respectively, so seed 17 keeps
  `model_450` while seed 23 has no checkpoint that passes the +/-45 safety
  gate.
- The alternative full-margin v17 continuation is rejected at its final
  `model_449`: 84.38% pose, 83.59% constrained, and 1.56% C1.  Its +35 to +45
  degree bin falls to 37.5%, so longer soft training did not fix the kinematic
  direction bias.  Artifact:
  `outputs/teacher_eval/seed41_v17_dir45_model449_balanced128/eval_summary.json`.
- Zero-shot clutter audits of the selected v19 `model_438` expose C3 as the
  first transfer bottleneck.  C2 obtains 0/6 success with no typed collision;
  C3 obtains 0/18 success with 11/18 C3 events; and the three-scene combined
  demo set obtains 0/3 success with 2/3 C3 events.  These are diagnostic
  baselines, not final benchmark claims.  The new one-blocker `C2-Soft` and
  `C3-Soft` profiles therefore isolate the corresponding dense cost before the
  two-blocker soft and hard combined stages.  Both profiles pass a real
  four-environment, one-iteration Isaac/PPO runtime smoke.
- The original C3 diagnostic also changed goal distance from the T0 teacher's
  6--10 cm range to 22 cm, so its zero-shot failure cannot be attributed to
  routing alone.  A new 12 cm C3 transfer set isolates the first routing step:
  offline geometry filtering leaves 243 scenes, and the 30-step PhysX audit
  leaves 234 stable training scenes.  A disjoint-seed evaluation pipeline
  leaves 157 stable scenes from 158 geometry-valid candidates.  Artifacts:
  `outputs/teacher_diagnostics/c3_short12cm_geometry_audit_v11.json`,
  `outputs/teacher_diagnostics/c3_short12cm_settling_v11.json`,
  `outputs/teacher_diagnostics/c3_short12cm_eval_geometry_audit_v11.json`, and
  `outputs/teacher_diagnostics/c3_short12cm_eval_settling_v11.json`.
- A stricter v12 audit now checks the actual whole-Franka C3 contact predicate
  throughout the 30 zero-action steps, not only rigid-body drift.  It observes
  zero reset/settling C3 contacts, leaves 232/234 training scenes after two
  additional drift rejections, and accepts all 157 evaluation scenes.  In
  contrast, deterministic `model_438` produces C3 on its first policy action
  in 107/128 held-out scenes and obtains 0/128 success.  The blocker therefore
  does not create an unrecoverable initial collision; it exposes the intended
  missing obstacle-conditioned routing behavior.  Artifacts:
  `outputs/teacher_diagnostics/c3_short12cm_zeroaction_c3_audit_v12.json`,
  `outputs/teacher_diagnostics/c3_short12cm_eval_zeroaction_c3_audit_v12.json`,
  and
  `outputs/teacher_eval/seed41_model438_c3short_baseline_balanced128/eval_summary.json`.
- The analogous 12 cm C2 proposal is deliberately rejected rather than used
  as an easy-looking but invalid curriculum.  With the current phone blocker,
  all 1,024 train and 768 disjoint eval candidates place the hammer/phone
  surfaces within 0.83 mm at both endpoints; therefore 0 scenes satisfy the
  required >5 mm start/goal clearance even though every straight midpoint is
  blocked.  C2 consequently retains the 22 cm geometry unless a genuinely
  smaller blocker asset is introduced.  Negative audit artifacts:
  `outputs/teacher_diagnostics/c2_short12cm_geometry_diagnostic_v13.json` and
  `outputs/teacher_diagnostics/c2_short12cm_eval_geometry_diagnostic_v13.json`.
- The replacement short-horizon C2 curriculum uses DOMINO
  `117_whiteboard-eraser:0` at scale `(0.04, 0.04, 0.04)` and mass `0.04 kg`.
  Unlike the phone, its approximately 7.7 x 3.0 x 4.1 cm support footprint
  leaves the hammer clear at both endpoints while every accepted scene blocks
  the protected-region straight-line sweep.  Geometry filtering retains 256
  training and 192 disjoint-seed evaluation candidates; a 30-step PhysX audit
  with the actual C2 contact predicate retains 250 and 185 respectively, with
  zero reset/settling C2 events in both sets.  The rejected 6/7 scenes exceed
  only the 0.10 rad/s terminal angular-speed gate.  Artifacts:
  `outputs/teacher_diagnostics/c2_short12cm_eraser_geometry_audit_v14.json`,
  `outputs/teacher_diagnostics/c2_short12cm_eraser_eval_geometry_audit_v14.json`,
  `outputs/teacher_diagnostics/c2_short12cm_eraser_train_zeroaction_c2_audit_v15.json`,
  and
  `outputs/teacher_diagnostics/c2_short12cm_eraser_eval_zeroaction_c2_audit_v15.json`.
  The direction-limited C1 `model_438` baseline obtains 0/128 success on the
  audited C2 set and never reduces planar goal error below the initial 12 cm;
  it records zero C2 events but 14 C1 episodes.  This is a genuine
  obstacle-conditioned transfer problem rather than an initial-collision
  artifact.  Baseline artifact:
  `outputs/teacher_eval/seed41_model438_c2short_eraser_baseline_balanced128/eval_summary.json`.
- The v22 C3-soft transfer started from selected v19 `model_438` on the 232
  reset-clear scenes with 1,024 environments.  Its first `model_450` obtains
  0/128 success and 102/128 C3 episodes under the disjoint hard C1+C3 audit.
  It was stopped at iteration 474 because clutter transfer had been started
  from the direction-limited +/-45-degree base before the prerequisite v23
  direction gate was complete.  It is diagnostic-only and must not be
  promoted as a trained C3 result.  Online run:
  `https://wandb.ai/simonlsx/non-prehensile-affordance/runs/g0yhojzk`.
- An earlier result that stopped after 129 aggregate vector terminals is
  invalid because fast successful environments were counted repeatedly.  It
  must not be cited as a 100% held-out result.  Only summaries declaring
  `episode_allocation=balanced_per_environment` are accepted hereafter.

## Required demos

The final handoff contains at least four slow-motion videos: C1 legal-handle
contact, C2 protected head passing a blocker safely, C3 whole-arm obstacle
routing, and a combined two-blocker push.  Every video must show the target
pose as a transparent hammer plus safe/protected semantic overlays, and its
sidecar JSON must record strict success and typed C1/C2/C3 outcomes.
Use `scripts/evaluate_affordance_teacher.sh` for both quantitative evaluation
and videos. Set
`PROFILE=t0_forward|t0_dir45|t0_dir90|t0_goal_side_dir90|t0_dir360_fixedyaw|c1_dir45|c1_dir90|c1_dir360_fixedyaw|t0|c1_forward|c1_matched|c1|c2|c2_short|c3|c3_short|combined`,
`CHECKPOINT=...`, and `VIDEO=1`
for a rendered one-environment demo; deterministic inference is always used.
Video mode overrides the environment's debug-wide `(8, 0, 5)` viewer with a
close-up camera by default.  `CAMERA_EYE="x y z"` and
`CAMERA_LOOKAT="x y z"` remain available for scene-specific framing.
For long runs, `scripts/watch_affordance_teacher_checkpoints.sh` waits for an
explicit checkpoint grid and evaluates each checkpoint exactly once, with a
bounded timeout and existing-summary skip.  It never promotes online rollout
metrics to held-out results.
Use `scripts/summarize_affordance_teacher_direction_eval.py` to aggregate those
balanced per-scene CSVs into six direction bins.  Checkpoint selection gates on
the `[70, 90]` positive endpoint with zero C1 before comparing overall success;
this prevents the easier negative directions from hiding the audited local
minimum.

The first accepted interim demo uses v19 `model_438` on held-out scene 71
(approximately +39 degrees).  It is a strict/constrained success with zero
C1/C2/C3 events and is stored at
`outputs/teacher_demos/seed41_model438_c1_dir45_scene0071/seed41_model438_c1_legal_handle_scene0071_slowmo4x.mp4`;
the auditable sidecar is in the adjacent `eval/eval_summary.json`.  This is the
C1 member only, not a substitute for the pending C2, C3, and combined demos.

## Audited diagnostic artifacts

The checked-in candidate generator is
`scripts/generate_affordance_teacher_diagnostics.py`.  Candidates are first
filtered by `scripts/audit_affordance_teacher_diagnostics.py`, then must pass
30 zero-action simulator steps under `scripts/audit_domino_manifest_settling.py`.
The current held-out demo/evaluation manifests are:

- C1: `data/manifests/teacher_diagnostics_v4/hammer_teacher_t0_c1_32_seed1817.jsonl`
  (32/32 settled);
- C2: `data/manifests/teacher_diagnostics_v3/hammer_teacher_c2_final_seed1817.jsonl`
  (6/6 settled, start/goal protected clearance above 5 mm, straight protected
  sweep midpoint below 5 mm);
- C3: `data/manifests/teacher_diagnostics_v3/hammer_teacher_c3_final_seed1817.jsonl`
  (18/18 settled, upright blocker centered on the initial TCP-to-safe-root
  corridor);
- combined: `data/manifests/teacher_diagnostics_v4/hammer_teacher_combined_final_seed1817.jsonl`
  (3/3 settled after satisfying both C2 and C3 geometry gates).

The small combined set is for qualitative counterfactual demos.  Quantitative
multi-seed reporting uses the larger direction-balanced manifest and must not
claim statistical significance from these three hand-audited scenes.

## Audited training manifests

The larger v6 curriculum set uses the same support pose and applies both an
offline geometry rejection test and a 30-step zero-action PhysX settling test:

- C2: `data/manifests/teacher_training_v6/hammer_teacher_c2_stable_seed3811.jsonl`
  (256/256 settled);
- C3: `data/manifests/teacher_training_v6/hammer_teacher_c3_stable_seed3811.jsonl`
  (248/256 settled);
- combined: `data/manifests/teacher_training_v6/hammer_teacher_combined_stable_seed3811.jsonl`
  (140/145 settled).

The final transfer manifest is
`data/manifests/teacher_training_v6/hammer_teacher_pose256_combined140_balanced536_seed4811.jsonl`.
It mixes 256 direction-balanced, randomized-yaw pose scenes with two shuffled
copies of the 140 audited combined safety scenes.  This prevents clutter
transfer from silently forgetting full-pose competence while giving C1/C2/C3
challenges approximately half of the training distribution.  Recreate the
mixture with `scripts/compose_affordance_teacher_manifest.py`.

Final combined validation uses the disjoint-seed manifest
`data/manifests/teacher_heldout_v7/hammer_teacher_combined_stable_seed5821.jsonl`
(64/64 settled).  No scene from this manifest is included in the v6 training
mixture.

Before the 22 cm C3 set, the first routing transfer uses the shorter audited
manifests:

- C2 train: `data/manifests/teacher_c2_short_eraser_v15/hammer_teacher_c2_short12cm_eraser_train_stable_resetclear_seed19847.jsonl`
  (250 stable scenes with zero zero-action C2 contacts);
- C2 eval: `data/manifests/teacher_c2_short_eraser_eval_v15/hammer_teacher_c2_short12cm_eraser_eval_stable_resetclear_seed20851.jsonl`
  (185 stable, disjoint-seed scenes with zero zero-action C2 contacts);
- C3 train: `data/manifests/teacher_c3_short_v12/hammer_teacher_c3_short12cm_stable_resetclear_seed14847.jsonl`
  (232 stable scenes with zero zero-action C3 contacts);
- C3 eval: `data/manifests/teacher_c3_short_eval_v12/hammer_teacher_c3_short12cm_eval_stable_resetclear_seed15851.jsonl`
  (157 stable, disjoint-seed scenes with zero zero-action C3 contacts).

Use `PROFILE=c3_short` for hard C1+C3 evaluation.  The geometry filter places
one upright mug on the initial TCP-to-safe-root corridor while preserving
positive reset clearance; C2 is intentionally disabled in this isolated
profile.

Matched-distribution T0 validation uses
`data/manifests/teacher_heldout_pose_v8/hammer_teacher_pose_heldout256_stable_seed6827.jsonl`:
256 disjoint-seed, direction-balanced, randomized-yaw scenes with 6--10 cm
goals. All 256 pass the same 30-step zero-action settling audit. `PROFILE=t0`
measures pose competence with soft costs; `PROFILE=c1_matched` evaluates the
same scenes with hard C1 termination. The longer 22 cm `PROFILE=c1` manifest
remains a transfer/qualitative diagnostic rather than the T0 pose gate.

The initial direction-expansion curriculum has its own disjoint forward-cone
validation set at
`data/manifests/teacher_heldout_forward_v9/hammer_teacher_forward_heldout128_seed7829.jsonl`.
It contains 128 scenes on the same support face, uses a different random seed,
and passes the 30-step settling audit in
`outputs/teacher_diagnostics/forward_heldout_settling_v9.json`.  Evaluate this
warm-up only with `PROFILE=t0_forward`; the final T0 gate remains the
direction-balanced `PROFILE=t0` set above.

Direction expansion deliberately changes one distribution axis at a time.
The disjoint train/eval pairs under
`data/manifests/teacher_direction_curriculum_v10/` cover ±45 degrees, ±90
degrees, and 360 degrees while retaining the forward warm-up's support face,
initial-pose/yaw range, goal distance, yaw delta, and friction range.  Advance
only after balanced held-out evaluation passes the preceding profile; broad
initial yaw, workspace, and dynamics randomization are later curriculum axes,
not bundled into the first direction transfer.
