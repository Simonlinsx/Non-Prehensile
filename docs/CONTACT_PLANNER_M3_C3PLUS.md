# Contact Planner M3: Semantic C3+ Integration

For current stock-closed-gripper acceptance and the 2026-09-08 bridge
comparisons, see [Simulation gates](SIMULATION_GATE_PROGRESS.md). Historical
single-scene successes below include other end effectors and executors;
they do not establish acceptance of the current closed-gripper route.

## Decision

M3 was started to replace the then-unsuccessful fixed straight-push action
family from M2 with the open-source Push Anything sampling + C3+ controller.
A repository-owned
monitor and semantic trajectory auditor independently evaluate strict target
pose and C1.  Isaac Lab is the independent evaluator for target physics,
clutter, whole-arm C3, and cross-simulator robustness.  The native C3+
measured-state/task-space relay and repository-owned Isaac executor now provide
a strict single-scene online closed-loop C1 pass and same-run video.  The
randomized S2 gate remains open.  M2 remains a documented ablation and is not
extended with more scalar rewards or stage-specific task waypoints.

### Current physical end-effector contract

The canonical Isaac/real-robot route now uses the stock Franka gripper with
both finger joints commanded closed.  No sphere, peg, or other pushing tool is
attached to the simulated robot.  C3+ still needs a translation-only contact
body, so the optimizer uses an 8 mm sphere inscribed in the closed fingertip
union; its front tangent matches the real distal finger face.  A separate
27 mm cylinder is used only as the conservative closed-finger collision
envelope during repositioning.  Cartesian execution holds the initial wrist
orientation so that this proxy-to-finger transform stays valid.  This keeps
the upstream Sampling + C3+ formulation while making the physical contact
geometry directly reproducible on a stock Franka.

The historical experiments below that explicitly say “spherical pusher” are
retained as ablations; they are no longer the evaluator default.

The upstream versions are pinned in
`third_party/push_anything/UPSTREAM.json`.  The audited checkout is DAIRLab
`dairlib` branch `push_anything_dev` at commit `9d988c835d6e99330397701487fce5ce4ceafa3c`;
its Bazel module pins C3 commit `5c08cb2e14b1ab10e024cb46e8504970cffcd5ea`
and Drake `v1.51.1`.

Apply the repository-owned integration patches to that exact checkout with:

```bash
bash scripts/apply_push_anything_patches.sh \
  /data1/linsixu/dairlib-push-anything
bash scripts/apply_c3_patches.sh \
  /data1/linsixu/c3-push-anything
```

The first patch contains the semantic C1 bridge: optional per-object sampling
and unsafe meshes, reproducible sampling, predicted-trajectory rejection, a
high-rate OSC execution shield, and the independent pose monitor.  The
complete physical/contact object model is retained.  It also invalidates a
cached repositioning point if object motion makes that point unsafe.  The
second patch is a one-line build compatibility fix for C3's no-Gurobi stub: it
matches the Gurobi implementation's existing
`options.M.value_or(1000)` conversion.  C3+'s optimizer is otherwise unchanged.
The small third upstream patch registers the online relay Bazel target; the
relay source itself is repository-owned and synchronized by
`apply_push_anything_patches.sh`.

## Architecture

```text
RGB-D / oracle geometry
        |
        v
object mesh + pose + safe/protected face semantics
        |
        v
safe-only global EE surface sampling
        |
        v
semantic C3+ local contact-implicit MPC
        |
        v
EE trajectory + contact-force plan
        |
        +--> Drake OSC / later real robot
        |
        `--> independent pose + semantic trajectory acceptance
```

Push Anything's public controller samples mesh-normal EE locations.  The
staged one-object configuration solves a five-step local C3+ problem for each
candidate and switches between a
collision-free repositioning phase and the contact-rich MPC phase.  This adds
the continuous contact trajectory freedom missing from M2.

## Semantic mesh contract

DOMINO supplies sparse affordance anchors and the current repository exposes
aligned point scores.  C3+ requires contact geometry, so M3 partitions every
target triangle into exactly one class:

- `protected`: any of the triangle's vertices, edge midpoints, or centroid is
  protected;
- `safe`: every sampled location is safe and none is protected;
- `neutral`: every mixed or uncertain triangle.

This is intentionally conservative at part boundaries.  The safe sampling
mesh is then eroded away from the unsafe boundary; this accounts for the
finite 19.5 mm EE sphere instead of treating a sampled surface point as a
zero-radius contact.  Generate the stable-support, object-local metre-scale
OBJ files and their provenance manifest with:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/export_domino_semantic_mesh.py \
  --domino-root /data1/linsixu/DOMINO \
  --asset-id 020_hammer:0 \
  --output-dir data/push_anything_semantics/020_hammer_0
```

The manifest records the source SHA-256, scale, stable support transform,
thresholds, class counts, erosion margins, and original face indices.
`full.obj` remains the physical model, `unsafe.obj` is protected plus neutral,
and `safe_guarded.obj` is used only by the sampler.  The accepted hammer retains
1,002 of 2,233 safe faces after a 40 mm surface-boundary and 65 mm offset-center
clearance check.  A target export fails closed if a required partition is
empty.

## Constraint mapping

| Contract | C3+ change | Acceptance |
| --- | --- | --- |
| C1 robot-target | guarded-safe sampling, C3 horizon rejection, cached-target invalidation, and high-rate OSC shield | offline finite-radius EE audit: legal safe contact exists and protected/neutral contact is zero |
| C2 clutter-target | reject target-protected/clutter events in every restored rollout and formal step | Isaac Lab protected-point clearance plus filtered PhysX target-obstacle contact |
| C3 robot-clutter | reject whole-robot/clutter events in every restored rollout and formal step | Isaac Lab arm proxy clearance plus per-link filtered PhysX contacts |

Filtering the global sampler alone is not sufficient for C1: the continuous
trajectory must also keep neutral/protected gaps positive.  Likewise, the
original Push Anything object-object contacts are deliberately enabled and
cannot be reported as satisfying C2 without the protected-part constraints.

The native restored-rollout executor deliberately separates the **actual
failure predicate** from the **planning margin**.  Actual C2/C3 failure remains
a localized, filtered PhysX contact.  Candidate rollouts are more conservative:
they are rejected when protected--clutter or whole-robot--clutter clearance
falls below 10 mm by default, even if that particular shadow simulation does
not yet register contact.  The margins are exposed as
`--rollout-protected-clearance-m` and
`--rollout-robot-obstacle-clearance-m`; the result JSON records the minimum
clearance and whether collision or margin rejection fired for every candidate.
This buffer is needed because restored PhysX rollouts and their subsequent
formal execution are close but not bit-identical.

The two simulators must also share the same physical tool center point.  Push
Anything does not use the stock Franka fingertips: its URDF mounts a 12.7 mm
radius, 101.6 mm long peg and a 19.5 mm radius spherical tip to `panda_hand`.
The Isaac executor generates the corresponding Franka URDF at startup and
uses the spherical surface for C1 classification.  It also raises the Franka
base by 29 mm because Push Anything places its support plane 29 mm below the
base frame.  Using a stock gripper or omitting this offset can produce a
plausible-looking trajectory whose physical contact and semantic audit do not
correspond to the planner model.

## Acceptance gates

The first controller gate is one DOMINO hammer, no clutter:

```text
3D position error     < 0.020 m
full SO(3) error      < 0.100 rad
dwell                 >= 5 monitor messages
legal safe contact    > 0
C1 violations         = 0
```

Pose success and semantic safety are reported separately, then combined by
`verify_push_anything_c1_acceptance.py`.  The deterministic seed-17 gate now
passes.  Randomized single-hammer robustness must pass before C2/C3 becomes an
end-to-end task claim.

## Accepted deterministic C1 run

Run `domino_hammer_strict_c1_resample_q5_yaw10_seed17_v47` uses an 80 mm
translation and 10 degree yaw target from the initial pose.  The task starts
with a joint translation-and-rotation objective; there is no pose-stage switch
or hand-authored waypoint sequence.  It reached the joint gate in 124.64 s:

| Metric | Result |
| --- | ---: |
| final position error | 0.00412 m |
| final SO(3) error | 0.03461 rad (1.98 deg) |
| consecutive in-gate messages | 5 |
| legal safe-contact rows | 397 |
| protected-contact rows | 0 |
| neutral-contact rows | 0 |
| C1 violation rows | 0 |

The audit covers all 2,518 CSV rows with complete EE/object poses, not only the
terminal state.  Its finite-radius contact threshold is 21.5 mm (19.5 mm sphere
plus 2 mm tolerance).  The compact checked-in evidence is
[`contact_planner_m3_c1_seed17_v47.json`](evidence/contact_planner_m3_c1_seed17_v47.json);
the full CSV remains an ignored generated artifact.

## Randomized 50-scene protocol

The nominal robustness gate keeps the same hammer, support face, and no-clutter
C1 contract while varying the task geometry.  The checked-in manifest
`hammer_c1_outward120_eval50_seed20260902.jsonl` contains exactly 50
deterministic scenes:

- initial X: 0.39, 0.40, or 0.41 m;
- initial Y: 0.18 through 0.22 m in 1 cm increments;
- goal distance: 0.06 through 0.10 m in 1 cm increments, ten scenes each;
- goal direction: stratified from -58.8 to +58.8 degrees relative to the
  ray from the robot base to the initial target position;
- relative goal yaw: -10, -5, 0, +5, or +10 degrees, ten scenes each;
- independent deterministic contact-sampling seed per scene.

Run the resumable evaluator with:

```bash
/usr/bin/python3 scripts/evaluate_push_anything_c1_randomized.py \
  --manifest data/manifests/contact_planner_m3/hammer_c1_outward120_eval50_seed20260902.jsonl \
  --output-root outputs/contact_planner_m3/hammer_c1_outward120_eval50 \
  --controller-position-success-threshold 0.015 \
  --controller-orientation-success-threshold 0.075 \
  --tcpq-port 7730 --timeout-s 180
```

`summary.json` is rewritten after every scene.  It reports geometry, C1, and
their conjunction separately, including four goal-direction bins.  Existing
completed scene artifacts are reused on restart unless `--rerun` is supplied.
Infrastructure failures, including an early native child-process exit, are
recorded separately and are not counted as algorithm failures.

The controller thresholds above are deliberately tighter than the unchanged
external 0.020 m / 0.100 rad / five-message acceptance gate.  They give the
object settling margin before C3 moves the EE away; they do not relax reported
success.  The wider
`hammer_c1_outward180_eval50_seed20260902.jsonl` remains a side-push stress
manifest.  It is not mixed into the nominal success rate because the guarded
safe-handle mesh and 0.3 friction coefficient make some extreme direction/yaw
pairs infeasible for a legal single contact.  The generator exposes
`--relative-direction-limit-deg` so both distributions are reproducible.

The pre-registered eight-direction screen for the nominal distribution passed
8/8 joint pose, 8/8 C1, with legal safe-handle contact in every scene and zero
infrastructure failures.  Each of the four relative-direction bins passed 2/2.
The compact auditable record is
[`contact_planner_m3_outward120_screen8.json`](evidence/contact_planner_m3_outward120_screen8.json).
The completed 50-scene nominal run passed the predeclared S1 gate: joint
pose+C1 passed 41/50 (82%), C1 passed 50/50, every scene made legal safe
contact, and there were zero infrastructure failures.  The four relative
direction bins passed 13/13, 9/12, 11/13, and 8/12 respectively, so the lowest
bin is 66.7%, above the 60% per-bin floor.  The nine task failures remain
geometry failures rather than safety or infrastructure failures.  The compact
record and source/result hashes are
[`contact_planner_m3_outward120_eval50.json`](evidence/contact_planner_m3_outward120_eval50.json).

The earlier `hammer_c1_front180_eval50_seed20260901.jsonl` sampled angles
around world +X.  Since the initial object has positive Y, its most-negative
directions have a negative dot product with the base-to-object ray and ask the
robot to move the object back toward its base.  Those are pulling/around-the-
back stress cases, not the intended outward pushing hemisphere.  That manifest
and its results are retained for provenance but are not used as the S1
front-hemisphere acceptance distribution.

### 2026-09-02 controlled objective ablation

The original local variant optimized XY and yaw simultaneously from the first
controller step.  The paper-aligned `push-anything` mode instead uses the
upstream schedule: position outside 5 cm, then full pose inside 5 cm.  Both
modes retain exactly the same terminal XY+SO(3)+dwell+C1 acceptance gate.

On the fixed eight-scene diagnostic subset (`000, 002, 006, 018, 029, 037,
042, 046`), the simultaneous variant passed 5/8 and the upstream schedule
passed 7/8.  C1 passed 8/8 under the upstream schedule.  It recovered one
old no-contact failure (`scene037`) and one yaw failure (`scene046`); the only
remaining failure was `scene002`.  The full fixed 50-scene run is stored under
`outputs/contact_planner_m3/hammer_c1_front180_eval50_upstream_objective_v1`
and is resumable.

The completed legacy world-frame 50-scene run exposed a directional failure
that the eight-scene subset did not represent well.  Joint pose passed 28/50
(56%), while C1 passed
50/50 with legal safe contact in every scene and zero protected/neutral
contacts.  It therefore **failed** the predeclared 80% S1 geometry gate:

| goal-direction bin | joint pose | C1 |
| --- | ---: | ---: |
| `[-90,-45)` | 0/12 | 12/12 |
| `[-45,0)` | 5/13 | 13/13 |
| `[0,45)` | 11/12 | 12/12 |
| `[45,90]` | 12/13 | 13/13 |

The result is stored in
`outputs/contact_planner_m3/hammer_c1_front180_eval50_upstream_objective_v1/summary.json`.
The compact checked-in summary is
[`contact_planner_m3_s1_front180_summary.json`](evidence/contact_planner_m3_s1_front180_summary.json).
Three fixed negative-direction scenes (`002`, `003`, `004`) were used for
bounded diagnostics:

| diagnostic | joint pose | C1 | conclusion |
| --- | ---: | ---: | --- |
| reduce the execution-shield stop band from 55 mm to 30 mm | 0/3 | 3/3 | safety clearance is not the blocker; do not relax C1 |
| add pre-switch quaternion weight 5 | 0/3 | 3/3 | one near miss improves, but the direction bin remains unsolved |
| add pre-switch quaternion weight 20 | 0/3 | 3/3 | stronger yaw cost is worse |

These negative controls freeze the safety margin and objective schedule.  They
show that the legacy pulling stress cases are not repaired by weakening C1 or
adding another scalar yaw cost.  A corrected outward-hemisphere screen must be
reported separately before deciding whether multi-contact lookahead is needed.

### Drake-to-Isaac support-frame and replay gate

The bridge now carries the semantic export's exact
`support_quaternion_wxyz`.  Previously it copied a template pose containing an
extra -90 degree planar yaw while leaving the TCP path unchanged.  That made a
Drake-safe contact land on the protected hammer head in Isaac Lab.  Unit tests
now reject a stage manifest without an explicit support frame, and older
artifacts are upgraded from the canonical semantic manifest.

After this fix, all 7/7 native-success trajectories in the controlled subset
made legal safe contact in Isaac Lab and none touched the forbidden region.
Two of seven also met the unchanged strict pose gate in PhysX (`scene006` and
`scene042`).  This is deliberately reported as an **open-loop cross-simulator
baseline**, not a deployable controller: the other five failures show that an
entire Drake trajectory cannot be replayed open loop across different contact
dynamics.  Formal execution stops only after the strict joint gate and C1 have
held for five steps; it no longer continues pushing after success.

A bounded measured-pose correction was also tested on the failed `scene018` as
a diagnostic and was rejected: it changed 31.96 mm / 0.142 rad into 33.17 mm /
0.364 rad while preserving C1.  Re-anchoring or biasing a recorded TCP path
cannot choose a new contact or recompute contact force, so this heuristic is
not part of the implementation.  The required next step is the actual online
interface already used conceptually by Push Anything on hardware: publish the
measured Franka and object state to the C3+ controller at every cycle, consume
its newly optimized command, and enforce the semantic shield during execution.

### Simulation-first acceptance ladder

No real-robot claim is made until the same controller passes the following
ordered gates.  Infrastructure failures are always reported separately.

1. **S1 — native single hammer:** fixed 50-scene manifest, strict XY/Z/SO(3)
   dwell gate, at least 80% joint success, at least 60% in every direction
   bin, and zero C1 violations.
2. **S2 — Isaac Lab online controller:** replace `franka_sim` with an Isaac
   state/command bridge so C3+ replans from measured state; at least 70% joint
   success on the same scenes and zero C1 violations.  Recorded-trajectory
   replay does not count toward this gate.
3. **S3 — typed clutter:** stable scenes containing both path blockers and
   target-touching clutter; jointly report task, C1, C2, and C3, with zero
   safety violations in the accepted 50-scene set.
4. **S4 — deployability stress:** repeat S2/S3 with bounded pose noise,
   observation delay, mass/friction variation, and command delay.  Only then
   connect the identical state/command contract to the Franka hardware bridge.

S1 passes on the outward 120-degree nominal distribution (41/50 joint and
50/50 C1).  The older world-frame front-180 stress distribution remains a
separate failed result (28/50 joint and 50/50 C1).  The S2 online bridge passes
one strict Isaac Lab scene, but the same-manifest randomized S2 rate is still
pending.  Perception is deliberately kept out of these oracle-state controller
tests so planning and perception failures remain identifiable.

The S2/native relay boundary is versioned in
`dapl.contact_planner.c3_online_protocol`.  Its state packet is restricted to
quantities that can be produced on hardware: timestamp/sequence, Franka `q`,
`dq`, measured effort, measured rigid-body pose/twist, and explicit safety
state.  Version 2 carries the target only; version 3 carries the target and up
to three active clutter objects in the target-first order recorded by the
shared scene spec.
It supports either seven joint efforts or the current C3+ task-space position,
velocity, and feedforward-force sample, plus explicit `READY`, `STALE_STATE`,
`SEMANTIC_HOLD`, and `PLANNER_FAILURE` bits.  Packets are fixed-size
network-endian records;
wrong versions, malformed sizes, non-finite data, unknown status bits, and
non-normalized object quaternions fail closed.  Inactive v2 slots have a
canonical zero padding, so an object-count mismatch cannot silently turn an
unstaged object into free space.  No future state, simulator handle, or PhysX
contact tensor may be a planner input.  The protocol does carry
`LEGAL_SAFE_CONTACT` and optional `FORCE_C3_MODE` bits; a hardware provider must
derive the former from force/torque sensing plus estimated semantic contact
location, not from a simulator contact label.  Isaac removes each DOMINO
object's own support rotation and the table offset before publishing, so the
native controller sees the same object frame as its baked mesh.

### Hardware-observation gate

The protocol being hardware-shaped is not by itself a real-robot result.  The
current Isaac provider still reads exact target/clutter root pose and twist,
and its semantic contact audit uses simulator geometry and filtered PhysX
reporters.  These are allowed only while isolating planner behavior and for
evaluation; the fixed-scene S2 pass is therefore controller-deployable but not
yet perception-complete.

Before claiming deployability, the same executable planner must pass without
changing its inputs or thresholds when the Isaac providers are replaced by:

| Runtime field | Hardware source | Current status |
| --- | --- | --- |
| Franka `q`, `dq`, effort | robot state interface | represented by the relay |
| target and clutter pose/twist | calibrated RGB-D tracking with uncertainty and stale-state rejection | provider pending |
| `safe` / `protected` surface | RGB-D semantic affordance prediction, optionally registered to a known CAD mesh | provider pending |
| legal safe contact | external-joint-torque or wrist F/T contact estimate intersected with the semantic surface | provider pending |
| goal pose | user/task command in the calibrated world frame | represented by the relay |
| task-space command | short C3+ segment followed by robot-local IK/impedance control | simulated executor exists; hardware bridge pending |

Simulator truth remains available to an independent evaluator, but it must be
blocked from the planner process during this gate.  The pre-hardware test adds
RGB-D occlusion/dropout, pose covariance, calibration error, force noise,
latency, and an emergency stop/retract path.  A planner rollout does not count
as real-deployable if it succeeds only with exact Isaac root state or a PhysX
contact flag.

Directly applying Drake OSC torques to Isaac was rejected as the deployment
interface.  In a 30 s diagnostic it preserved C1 but never contacted the
target, because the Drake and Isaac end-effector/dynamics models are not
torque-equivalent.  The accepted interface consumes C3+'s newly optimized
task-space segment and uses robot-specific IK plus a 100 Hz local joint servo.
C3+ receives the latest measured robot/target state and replans at 20 Hz.
Between replans the servo follows only the velocity of that current short
segment; no previously recorded global trajectory is replayed.

The relay waits for a trajectory stamped from the just-published state and
fails closed on timeout.  A 100 microsecond comparison tolerance accounts only
for Drake's floating-second to integer-microsecond truncation (observed as
`200000 -> 199999`); it is 500 times smaller than the 50 ms planner period.
The first planner cycle is allowed to hold during startup.  The TCP idle
timeout is separately configurable and defaults to 60 wall-clock seconds:
multi-object optimization and rendering can take more than two wall-clock
seconds between valid simulator packets, which is not evidence of stale
simulation state.

One reproducible online Isaac Lab invocation is:

```bash
PUSH_ANYTHING_ROOT=/data1/linsixu/dairlib-push-anything \
PUSH_ANYTHING_TCPQ_PORT=7804 \
PUSH_ANYTHING_RELAY_PORT=7805 \
PUSH_ANYTHING_FRESH_TASK_TIMEOUT_MS=1000 \
PUSH_ANYTHING_FRESH_TIMESTAMP_TOLERANCE_US=100 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_push_anything_isaaclab_online.sh \
  outputs/contact_planner_m3/ablation_upstream_objective_v2_valid/scene006/isaaclab_manifest_support_fixed.jsonl \
  outputs/contact_planner_m3/online_isaac_scene006_v1/final_synchronized20hz_video/result_task_video.json \
  --planner-frequency-hz 20 --max-sim-time-s 25 \
  --video \
  --video-folder outputs/contact_planner_m3/online_isaac_scene006_v1/final_synchronized20hz_video \
  --video-name-prefix scene006_c3_online_final \
  --video-fps 20 --headless --device cuda:0
```

The wrapper starts and cleans up the TCPQ hub, native C3+ controller, and relay
as one run.  The selected upstream checkout must first be staged to match the
Isaac manifest exactly.  In scene mode the wrapper reads `runtime_root` from
the shared scene spec unless `PUSH_ANYTHING_ROOT` is explicitly set, exports
the resolved path to the executor, and records it in `result.json`.  This
prevents Isaac from loading clutter while C3 silently reads a stale
single-object configuration.  The default joint gate is 20 mm XY, 10 mm Z,
100 mrad SO(3) (about 5.7 degrees), held for 0.5 s.  This matches the stated
approximately-six-degree task setting and Push Anything's staged goal file.
A timestamp-tolerant non-rendered run also passed the stronger historical
75 mrad diagnostic at
17.54 mm XY, 0.99 mm Z, and 73.91 mrad, with 348/349 fresh planner cycles and
no C1 violation.  A separately rendered replicate passed the same pose/safety
gate at 17.65 mm XY, 0.99 mm Z, and 64.85 mrad; it contains more intermittent
safe holds (241/353 fresh cycles), so it is visual evidence rather than the
freshness reference.  The 1280x720/20 FPS video is
`outputs/contact_planner_m3/online_isaac_scene006_v1/synchronized20hz_video/scene006_c3_online_synchronized20hz.mp4`.
The older checked-in evidence record retains a historical 100 mrad result for
provenance and must not be confused with the stronger 75 mrad diagnostic above.

Video frames are streamed directly to FFmpeg and rendering is limited to 20
Hz, so recording does not retain several gigabytes of raw 100 Hz RGB frames or
alter the 100 Hz servo clock.  The translucent cyan hammer is the goal pose;
green and red point overlays are the moving safe and protected regions.

### S2 executor reset: keep planning and contact execution separate

The single-hammer bridge is treated as two components. C3+ owns global contact
selection, repositioning, and object-effect prediction. The robot-local
controller owns tracking, contact acquisition, and the independent C1 stop.
The default path does not add a waypoint policy, learned residual, forced-C3
mode, or yaw-specific lateral contact offset.

A controlled audit ruled out three tempting but incorrect explanations for the
remaining Isaac gap:

| change | best/final XY | final SO(3) | conclusion |
| --- | ---: | ---: | --- |
| clean 20 Hz position-IK baseline | 31.4 / 31.6 mm | 34 mrad | safe, but leaves a lateral residual |
| only raise replanning to 30 Hz | 35.0 / 38.2 mm | 101 mrad | planner frequency is not the cause |
| official low-PD Franka gains | 33.1 / 43.3 mm | 170 mrad | lowering joint stiffness is worse |
| replay a native-success C3 path in Isaac | 72.5 / 77.9 mm | 422 mrad | a native plan is not dynamically portable |

The online rigid-body state uses the object link origin (not the COM) and the
same support-frame, table-height, goal, mass, inertia, and friction contract as
the staged Drake model. The residual therefore belongs to the local
trajectory/contact-effect boundary, not to a fixed coordinate offset.

The Isaac OSC path also no longer requests a Jacobian at the 126.5 mm spherical
tip through `body_offset`. IsaacLab 2.2 applies that body-frame offset directly
to a root-frame Jacobian, which made the wrist drift to its limits. OSC now
controls `panda_hand` and analytically transforms the spherical-tip target into
the hand frame. This keeps the wrist stable, but a full 25 s pure-C3 test still
finished at 63.5 mm / 185 mrad. Consequently, further OSC gain sweeps are not
part of the plan.

The later fixed-seed contact-executor audit produced the following results.
Every row passed C1/C2/C3; none passed the joint pose+dwell gate:

| run | single change | best/final XY | final SO(3) | decision |
| --- | --- | ---: | ---: | --- |
| v172 | C3 approach + bounded axial contact pulse | 21.29 / 21.29 mm | 91.3 mrad | retained as the simple near-miss baseline |
| v173 | add measured-yaw lateral offset | 21.02 / 21.02 mm | 89.2 mrad | no material benefit |
| v174 | stop a yaw pulse when angular velocity reverses | 21.15 / 30.83 mm | 12.0 mrad | fixes yaw by sacrificing XY |
| v175 | redirect the active push toward the live XY goal | 20.48 / 20.48 mm | 94.8 mrad | near miss; violates the C3-selected contact geometry |
| v177 | increase all horizon weights with `gamma=1.5` | 76.63 / 76.63 mm | 60.9 mrad | rejected; planner avoids productive contact |
| v178 | retain only contact lines within 5 mm of COM | 51.75 / 51.75 mm | 56.8 mrad | rejected; moment arm alone misses support-friction effects |
| v179 | execute the 50 ms first knot instead of 100 ms lookahead | 89.64 / 89.64 mm | 18.1 mrad | rejected; position servo does not realize enough contact impulse |
| v181 | retain the 0.5 N contact-confirmation threshold for 45 s | 20.83 / 20.83 mm | 95.7 mrad | rejected; most productive 0.10--0.22 N contacts are misclassified as no contact |
| v182 | only lower contact confirmation to 0.1 N | 19.66 / 19.66 mm | 28.3 mrad | **accepted**; 0.5 s strict dwell and C1/C2/C3 all pass |

The v175 direction override, v177 gamma change, and v178 COM filter were
removed after their controlled runs. The v179 lookahead was not made the
default. Defaults use the upstream `gamma=1.0`, 100 ms task lookahead, 100 mrad
pose threshold, no forced-C3 contact mode, and zero lateral yaw gain.  The
50 g DOMINO hammer uses a 0.1 N contact-confirmation threshold: the previous
0.5 N value exceeded almost every sustained productive contact in the PhysX
audit and caused the executor to repeat contact acquisition instead of handing
off to its bounded contact servo.  This threshold is an observable sensor
calibration parameter, not privileged simulator state; hardware deployment
must estimate it from the real force/torque noise floor and the mounted
pusher's contact sensor.

An audit bug had also compared measured motion with
`DYNAMICALLY_FEASIBLE_BEST_PLAN`, which is the prospective repositioning
candidate rather than the plan being executed. The relay now records
`DYNAMICALLY_FEASIBLE_CURR_PLAN`. With that correction, the 15--17 s contact
window shows that C3's immediate yaw prediction and PhysX yaw have the same
sign. The longer five-step prediction is non-monotonic and places much of the
orientation correction late in the horizon. This rules out a fixed frame-sign
patch and identifies a receding-horizon first-action issue instead.

The single-hammer gate no longer justifies a new planner cost: v182 passes it
after correcting the measured-contact threshold alone. Terminal-wide gamma
sweeps, COM heuristics, and additional Isaac-side yaw state machines remain out
of scope. The next step is a rendered repeat of v182, followed by target-only
randomized evaluation; clutter is enabled only after that repeat confirms the
same physical execution and safety outcome.

The rendered v183 repeat reproduced the same result at 19.66 mm XY and
28.3 mrad SO(3), held for all 50 required servo steps with C1/C2/C3 passing.
Its 1280x720, 20 FPS Isaac Lab video is
`outputs/contact_planner_m3/acceptance_v183_contact_threshold010_video/videos/hammer_safe_c3_v183.mp4`.
The translucent cyan hammer is the goal pose; green points mark the legal safe
handle and red points mark the protected tool head.

The current target-plus-clutter online diagnostic uses a stable DOMINO mug
immediately beside the hammer.  Both rigid-body states are published to C3+;
Isaac independently audits C1/C2/C3 at 100 Hz and stops on the first violation.
With the physical spherical pusher, matched support frame, and resolved scene
runtime, a 21 s rendered probe made legal safe-handle contact and passed all
three safety audits, but moved the target by less than 1 mm and therefore did
not pass the strict pose gate.  Its actual Isaac Lab video is
`outputs/contact_planner_m3/online_clutter_c3_scene_v5_targetswitch050/scene034_physical_pusher_contactservo_c006_video21s_v1/videos/scene034_hammer_mug_c3_contactservo.mp4`.

A second diagnostic that continuously advanced the same contact reduced XY
error by about 4.6 mm and kept C1/C2, but correctly terminated when the
spherical pusher touched the mug (C3).  This exposes the present algorithmic
bottleneck: yaw correction at a fixed handle contact moves the pusher toward
the nearby obstacle.  The continuous contact servo remains disabled by
default and is only an ablation.  The formal next experiment uses the native
C3+ mode switch to push, retract, resample a safe contact, and replan from the
new measured state.  Until that run reaches the strict pose+dwell gate without
C1/C2/C3 violations, this scene is a diagnostic—not an accepted clutter demo.

The state protocol now separates `LEGAL_SAFE_CONTACT` (an observed fact) from
`FORCE_C3_MODE` (an explicit planner request). This prevents a contact audit
from silently changing planner mode. Both are obtainable from a real-robot
contact/semantic monitor; neither requires PhysX-only state. Physical contact
sensors remain evaluation-only for C1/C2/C3 unless the executor explicitly
enables force-C3.

On fixed target-plus-mug `scene034`, the historical 5-candidate 90 s run
reduced XY error from 120.0 to 80.5 mm and ended at 71.9 mrad yaw. All C1/C2/C3
audits passed, but this remains a clutter diagnostic rather than an accepted
demo. Candidate-count expansion is postponed until the simpler single-hammer
first-action gate above passes.

The old top-down trajectory renderer is not an Isaac Lab demo and is retained
only as an offline diagnostic.  Produce an auditable simulator video by
recording the formal execution from the same process that evaluates restored
physics rollouts:

```bash
DAPL_CLUTTER_MANIFEST=data/manifests/contact_planner_m3/push_anything_scene000_isaaclab_smoke.jsonl \
DOMINO_ROOT=/absolute/path/to/DOMINO \
DOMINO_USD_ROOT=$PWD/data/domino_usd \
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/run_contact_planner_m1.py \
  --task Isaac-AffordanceTeacher-FrozenV7-GoalWrench-C1-Franka-v0 \
  --num-envs 1 --max-replans 12 \
  --output-candidates 32 --physics-rollout-candidates 16 \
  --rollout-lookahead-steps 2 --rollout-plateau-escape-actions 2 \
  --rollout-maximum-cost-increase 0.1 \
  --rollout-transient-rotation-cap-rad 0.15 \
  --push-direction-samples 7 --push-direction-span-deg 60 \
  --hand-yaw-samples 13 --hand-yaw-span-deg 180 \
  --minimum-push-distance-m 0.003 --maximum-push-distance-m 0.015 \
  --push-distance-samples 5 --physical-contact-force-threshold-n 0 \
  --retreat-lift-m 0.05 \
  --video --video-folder outputs/contact_planner_m3/videos/native_c1 \
  --output outputs/contact_planner_m3/native_c1_result.json --headless
```

The selective recorder skips shadow rollouts and writes only the actual robot
execution.  The translucent cyan goal hammer and the green safe/red protected
surface markers are drawn by the native Isaac Lab visualization wrapper.

## Native build (no Docker and no Gurobi)

Docker is not required.  C3 provides a no-Gurobi implementation of the MIQP
class, and the `anything` example has a separate `C3+` configuration.  M3
therefore compiles with `--define=WITH_GUROBI=OFF` and runs C3+; it does not use
the unavailable MIQP projection at runtime.

The current server account has no sudo access, so M3 uses a fully user-local
toolchain.  Bootstrap the pinned Bazelisk binary (including SHA-256
verification) with:

```bash
cd /data1/linsixu/IsaacLab-nonPrehensile
bash scripts/bootstrap_push_anything_user.sh
```

The build wrapper puts Bazel/Bazelisk caches on `/data1` and reuses the
OpenBLAS shared library already present in the `anydex-torch` Conda
environment.  LCM is a Bazel module dependency and is compiled in the Bazel
workspace; a system `liblcm-dev` package is not required.  Bazel uses its
embedded JDK, so the host OpenJDK 11 is not part of this toolchain.  No Gurobi
installation or license is needed for our C3+ path.

The wrapper uses Bazel batch mode because the restricted execution environment
does not permit the local gRPC socket used by Bazel's persistent server.  This
is slower to start but does not change the compiled controller.

The build also overrides the C3 module with the pinned local checkout so the
audited no-Gurobi compatibility patch is used.  The original commit remains
unchanged and the local diff is fully represented by the repository patch.

Afterwards, check and build only the five targets needed for the simulation and
monitor:

```bash
cd /data1/linsixu/IsaacLab-nonPrehensile
bash scripts/build_push_anything_native.sh --check
bash scripts/build_push_anything_native.sh --build
```

The wrapper pins the audited upstream commit, keeps Bazel output on `/data1`,
adds the user-local OpenBLAS library and runtime path, explicitly disables
Gurobi, and avoids the much larger `bazel build ...` target.  Keep at least
roughly 30 GiB free for the pinned Drake source build.

### Reproduce the accepted gate

The end-to-end wrapper stages the exact hammer/configuration, incrementally
builds the pinned sources, launches the four local processes, audits every
trajectory row, and writes `joint_acceptance.json`:

```bash
cd /data1/linsixu/IsaacLab-nonPrehensile
PUSH_ANYTHING_TCPQ_PORT=7727 \
  bash scripts/run_push_anything_c1_acceptance.sh
```

Set `PUSH_ANYTHING_BUILD=0` only after the exact staged source has already been
built.  A run exits zero only when both geometry and semantic C1 pass.

### Verified on this host

On 2026-09-01, the user-local build completed all 12,587 actions for the first
four targets, and the online relay target was added and rebuilt on 2026-09-02:

- `//examples/sampling_c3:franka_sim`
- `//examples/sampling_c3:franka_osc_controller`
- `//examples/sampling_c3:franka_sampling_c3_controller`
- `//examples/sampling_c3:monitor_push_anything_baseline`
- `//examples/sampling_c3:online_bridge_relay`

`ldd` resolves Drake, the Bazel-built LCM library, and the user-local OpenBLAS
library.  The deterministic single-hammer pose+C1 gate above also completed in
the native, sudo-free runtime.  This is one accepted controller instance, not
yet a claim of randomized-scene robustness, C2, C3, or real-robot success.
