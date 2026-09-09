# FR3 simulation development baseline — 2026-09-09

This snapshot packages the current implementation for continued optimization.
**It has not passed randomized simulation acceptance and is not a hardware
release.** Publishing it before acceptance was explicitly requested by the user.

## Current method

IsaacLab supplies measured FR3/hammer state to the original Push Anything/C3+
planner at 20 Hz. The original native OSC computes fresh joint torques at 1 kHz;
IsaacLab executes them with 1 ms physics steps. Semantic safe-handle sampling,
predicted-path checks, and measured C1 contact auditing protect the hammer head
and claw. The physical end effector is the original closed stock gripper.

The current entry point is `scripts/run_push_anything_isaaclab_online.sh` with
`PUSH_ANYTHING_EXECUTOR_MODE=effort`. The task-space servo executor, attached
sphere, alternate dynamics rollouts, and older teacher results remain available
as historical experiments; they are not the current FR3 acceptance setup.

## Verified result and limits

Debug scene007 succeeds in **14.182 simulation seconds**: XY error 18.754 mm,
height error 0.988 mm, full SO(3) error 0.08482 rad. All three thresholds hold
continuously for 500 measured 1 ms steps. Dense finger closure, legal physical
contact, C1, and actual FR3 backend alignment pass their independent audits.
The rendered rerun has the same 1,899 trace rows as the prior successful run.
Repeated runs of this one debug scene do not count as independent test scenes.

[Actual IsaacLab success video](media/fr3_isaaclab_success_scene007_goal.mp4)
shows the full robot and a cyan goal object at 35% opacity. The goal has no
enabled collision or rigid-body physics. Video SHA256:
`2533771b72b9bcb729565fce4e1e3b94b18e3c30892102c412c6513695a7786a`.
The failure rerun is still running at this publication snapshot.

The formal 50-scene batch uses an earlier frozen executor. At the archived
audit, scenes000/002/003 finished without success; scenes001/007 have invalid
exception evidence, and 45 scenes have no final auditable result yet. No formal
success is certified at this snapshot. All 50 remain in the denominator.
The batch cannot satisfy the complete evidence contract, regardless of its
remaining results. See the [partial audit](evidence/fr3_baseline_20260909/formal50_partial_audit.json).
Formal scene007 is a different scene from the successful debug scene007.

Known issues:

- Generalization: some initial poses stall or leave excessive XY/orientation
  error after 180 seconds.
- Workspace: formal scene001 exceeded the native 0.75 m TCP radius bound.
  Candidate trajectories can remain inside the planner's XYZ box while crossing
  the radial bound. Candidate observations do not prove every knot was executed;
  selected-reference feasibility and tracking margins require further work.
- Failure evidence: the old frozen executor lost in-memory records on exceptions.
  The current executor preserves partial CPU-side evidence, closes sidecars/video,
  and distinguishes completed, contact-audited, and partially executed steps.
  CPU fault injection passes; a real exception regression is still pending.

## Repository map

| Purpose | Entry |
| --- | --- |
| Acceptance criteria and frozen batch | [SIMULATION_ACCEPTANCE_GOAL.md](SIMULATION_ACCEPTANCE_GOAL.md) |
| FR3 physics, friction, closed-finger mimic | [FR3_MODEL_ALIGNMENT.md](FR3_MODEL_ALIGNMENT.md) |
| Pinned upstream and native build | `third_party/push_anything/UPSTREAM.json`, `scripts/build_push_anything_native.sh` |
| Native integration patches | `third_party/push_anything/patches/0001` through `0026` |
| Matched official robot assets | `data/robot_models/fr3_stock_20260909/` |
| Captured successful runtime | `data/baselines/fr3_native_osc_20260909/` |
| Online torque executor and recording | `scripts/run_c3_online_isaaclab.py`, `scripts/isaac_online_video.py` |
| Scene preparation, freeze, workers | `scripts/prepare_fr3_osc_scenes.py`, `scripts/fr3_frozen_batch.py`, `scripts/run_fr3_osc_batch_worker.py` |
| Independent result audit | `scripts/audit_fr3_osc_batch.py` |
| Compact evidence and snapshot time | [evidence/fr3_baseline_20260909/](evidence/fr3_baseline_20260909/) |

The captured runtime contains all 60 enumerated native parameters/geometry inputs
and their checksums, plus the Isaac scene and exact recorded video launch plan.
No planner gains were changed during publication. Large raw traces, native
binaries/runfiles, simulator caches, and active outputs remain local under
`outputs/contact_planner_m3/workspace_height_20260908/`. The committed summaries
record their hashes and provenance; they do not replace the complete raw evidence.

## Run and verify

Use the existing Isaac Sim 5.0 / IsaacLab 2.2.0 environment and built native
Push Anything checkout. The simulator, DOMINO assets, semantic assets, and native
runfiles are external prerequisites; this is not a self-contained container.
On another machine, regenerate model paths and stage assets before execution.
The captured model manifests and Isaac scene contain this workstation's absolute
paths so their original evidence hashes remain intact.

To regenerate model files in a **new** directory for a different checkout:

```bash
python scripts/prepare_fr3_robot_models.py \
  --official-urdf data/robot_models/fr3_stock_20260909/fr3_official.urdf \
  --package-root data/robot_models/fr3_stock_20260909/vendor/franka_description \
  --output outputs/fr3_local_model
```

For a new debug replay on the existing workstation, select free ports/GPU and
an unused output directory. Use a clean shell without inherited experimental
`PUSH_ANYTHING_*` or `DAPL_*` overrides:

```bash
export ISAACLAB_PYTHON=/data1/linsixu/miniconda3/envs/dapl-isaaclab/bin/python
export PUSH_ANYTHING_ROOT="$PWD/data/baselines/fr3_native_osc_20260909/runtime"
export PUSH_ANYTHING_BINARY_ROOT=/data1/linsixu/dairlib-push-anything-canonical
export PUSH_ANYTHING_ROBOT_MODEL_MANIFEST="$PWD/data/robot_models/fr3_stock_20260909/robot_model_manifest.json"
export PUSH_ANYTHING_EXECUTOR_MODE=effort PUSH_ANYTHING_TCPQ_QUICK_ACK=1
export PUSH_ANYTHING_PLANNER_PERIOD_MS=50 PUSH_ANYTHING_FRESH_TASK_TIMEOUT_MS=10000
export PUSH_ANYTHING_FRESH_EFFORT_TIMEOUT_MS=1000 PUSH_ANYTHING_DIAGNOSTIC_OSC_HOLD=0
export PUSH_ANYTHING_TCPQ_PORT=22184 PUSH_ANYTHING_RELAY_PORT=22185
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_push_anything_isaaclab_online.sh \
  data/baselines/fr3_native_osc_20260909/isaac_manifest.jsonl \
  outputs/fr3_debug_replay/result.json \
  --stock-closed-gripper --headless --device cuda:0 --max-sim-time-s 180 \
  --physics-dt-s 0.001 --control-decimation 1 --trace-stride 10 \
  --socket-timeout-s 10 --video --video-fps 25 --goal-ghost-opacity 0.35
```

The command points at the byte-verified copied runtime. The observed success
comes from the original launch recorded in `recorded_video_launch.json`; another
run must be measured and audited independently. Debug replays are excluded from
the formal 50-scene batch.

CPU regression command, run with the project Python environment:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile python -m pytest tests -q
```

At publication: **295 passed, 1 skipped, 7 subtests passed**; three dependency
deprecation warnings. The successful physical video provides normal-path
regression evidence, but passing CPU tests does not certify manipulation success.

## Next optimization

Finish the failure video and retain every prescribed batch outcome. Diagnose
selected-reference workspace feasibility and pose convergence, preserve original
OSC and C1 guards, then validate a revised controller on debug scenes before
freezing another full acceptance batch. Current live workers and their frozen
inputs must not be changed by an optimization or native rebuild.
