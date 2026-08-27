# Affordance teacher task settings

This file is the source-of-truth for teacher experiments. The main setting is
an affordance-constrained DAPL task. Values must not be changed without naming
the resulting run as an ablation.

## Scene and task distribution

| Setting | DAPL | DyWA | Main teacher |
| --- | --- | --- | --- |
| Physics step | not separately reported | 0.0125 s | 0.0125 s |
| Policy period | world model predicts 0.1 s | 8 physics steps = 0.1 s | 8 physics steps = 0.1 s |
| Episode horizon | 300 control steps | 128 in the released `arm_div_base.yaml` teacher (`icra_base.yaml` is a separate 300-step setting) | 300 control steps |
| Parallel environments | 2048 over 8 L40 GPUs | 4096 for the teacher | 1024 per seed (user-requested compute setting) |
| Target initial XY | table-centred x `[-0.15, 0.15]`, y `[-0.30, 0.30]` | sampled on the `0.4 x 0.5 m` table with scene margin scale `0.95` and footprint fall prevention | DAPL range |
| Initial orientation | random precomputed stable pose, up to 64 candidates | random stable pose plus random yaw | random DAPL stable support face plus yaw |
| Goal orientation | independently random precomputed stable pose | random stable pose plus random yaw | same support face as initial pose plus independent full-range yaw |
| Goal XY | no separate numerical box is reported; minimum displacement is specified | released `arm_div_base` uses task `margin_scale=0`: sample along the current-object-to-table-centre ray, from `1.1 x 0.05 = 0.055 m` separation toward the centre | same central target region; explicit compatibility choice |
| Robot initial joints | not numerically reported | uniform in the published seven-joint box | DyWA joint box |
| Minimum initial-goal XY separation | 0.15 m | 0.10 m in the paper | 0.15 m |
| Tasks per scene | 16 | resampled per episode | 16 |
| Train/eval scenes | 1024 sparse train; 128 held-out per track | 323 train objects; 50 eval variants | 1024/128 |
| Clutter density | sparse 4, moderate 8, dense 12 total objects | single object | DAPL density tracks |

DAPL sources: [paper appendix](https://arxiv.org/html/2603.09882),
`dapl/generation.py`, and `dapl/scene.py`. DyWA sources:
[paper](https://arxiv.org/html/2503.16806),
`/data1/linsixu/DyWA/dywa/src/data/cfg/env/arm_div_base.yaml`, and
`/data1/linsixu/DyWA/dywa/src/data/cfg/env/icra_base.yaml`.

The DyWA Franka reset box used by the main teacher is, in joint order,
`q_lower=(-0.3,-0.4636,-0.2,-2.7432,-0.3335,1.5269,-1.5708)` and
`q_upper=(0.3,0.5432,0.2,-1.5237,0.3335,2.5744,1.5708)` radians. Joint
velocities are reset to zero.

### No silent assumptions

The following settings cannot honestly be called exact DAPL values because
the paper does not publish them or because this benchmark intentionally
changes them:

- DAPL does not report a separate numerical XY box for the goal. The current
  generator samples both ends of a task pair in the published central target
  region. This keeps both poses reachable and is recorded as an implementation
  choice, not a quoted paper parameter.
- The tabletop placement bounds and DOMINO object masses/scales/frictions come
  from this repository's workspace and DOMINO metadata. They are not silently
  replaced by DyWA's single-object domain-randomization ranges.
- The main target set is restricted to the DOMINO hammer because the first
  scientific question is affordance-safe tool manipulation. DAPL uses a broad
  Objaverse library and DyWA uses DexGraspNet objects.
- T0 has no active clutter so C1 can be identified cleanly. C2/C3 will activate
  the DAPL sparse composition of one large and two small obstacles.
- Exact DAPL generation independently samples the initial and goal stable
  poses.  That is retained as the named `dapl-paper` reproduction setting, but
  is not the main planar-pushing teacher distribution: in the audited hammer
  manifests only 11.8% of relative rotations were about the table normal and
  61.3% changed support height by more than the strict 1 cm tolerance.  The
  main teacher therefore uses the explicitly named `dapl-planar-push` setting.
  It changes only `preserve_target_support_pose=True`: DAPL XY ranges, minimum
  15 cm displacement, 16 tasks per scene, obstacle composition, randomized
  initial stable face, and independent full-range yaw remain unchanged.  Full
  Z/roll/pitch/SO(3) success is still evaluated, so tipping the tool is a
  failure rather than an unobserved degree of freedom.
- The action is DAPL-style seven-joint residual control. The simulator action
  magnitude is `0.1`; DAPL does not report its simulation bound, although its
  real-robot appendix reports a `0.1` to `0.01` action-magnitude curriculum.
- C1 penalties and strict five-step success are our benchmark additions. They
  must never be presented as DAPL or DyWA settings.
- `dywa-arm-div-planar-push` is a root-cause diagnostic, not the main DAPL
  distribution. It reproduces the released `arm_div_base` centre-ray XY
  geometry with root offsets `[-0.19,0.19] x [-0.2375,0.2375]` m and minimum
  separation `0.055` m, while deliberately retaining the benchmark's
  same-support-face requirement, strict success, 300-step horizon, DOMINO
  hammer, and inactive clutter. Train/eval manifests are
  `data/manifests/domino_hammer_dywa_armdiv_planarpush_train1024_seed1701.jsonl`
  and
  `data/manifests/domino_hammer_dywa_armdiv_planarpush_eval128_seed1801.jsonl`.
  They contain 16,384/2,048 unique tasks with zero overlap; mean goal
  displacement is 9.74/9.78 cm.

## Reward contract

The task keeps DAPL's four terms and weights:

| Term | Definition | Weight |
| --- | --- | ---: |
| Contact proximity | `1 - tanh(d_safe / 0.1)` | 1 |
| Coarse goal | safe-distance gate `< 0.1 m`, `1 - tanh((d_p+d_r/5)/0.6)` | 5 |
| Fine goal | same gate, `1 - tanh((d_p+d_r/5)/0.3)` | 16 |
| Success | sparse success reward | 2000 |

The only semantic substitution is `d_safe`, the minimum end-effector distance
to a safe affordance point, in place of DAPL's target-centroid distance. C1 is
then added as a local soft penalty and as a validity condition on success.
There is no waypoint, prescribed push side, yaw half-space, wrench target,
first-contact bonus, component-wise pose progress, or action penalty.

The benchmark's strict success condition (XY < 0.02 m, Z < 0.01 m, full
SO(3) < 0.1 rad for five consecutive steps) is intentionally stricter than
DAPL's reported planar 0.05 m / rotation 0.1 rad condition. This is a stated
task contribution requested for safe tool placement, not an undocumented
randomization choice. Both DAPL-compatible and strict success should be
reported during evaluation.

## PPO and observation contract

PPO follows DAPL: value loss coefficient 0.5, clipped value loss, clip 0.3,
entropy 0.006, 8 epochs, 8 mini-batches, learning rate `5e-5`, adaptive
schedule, gamma 0.99, GAE lambda 0.95, desired KL 0.016, and max gradient norm
1.0.

The actor receives the recoverable affordance observation: 512 target points
with `[x,y,z,safe,protected]`, 512 obstacle XYZ points, hand state, joint
state, previous action, relative goal, and noisy target twist. Exact mass and
friction remain critic-only. This differs deliberately from DAPL's privileged
per-point mass/velocity representation because the current phase validates an
oracle-affordance teacher that can later be distilled.

## Experimental progression

1. T0/C1: hammer target, DAPL pose distribution, inactive clutter, soft C1.
2. C1-hard: initialize from the selected T0 checkpoint and enable illegal
   robot-to-protected contact termination.
3. C2/C3: activate DAPL sparse clutter and add each typed constraint in
   isolation before the combined task.
4. Moderate and dense tracks are evaluation/generalization stages, not a
   replacement for establishing the sparse teacher.
