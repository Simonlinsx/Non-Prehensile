# FR3 simulation acceptance goal

The active goal requires **FR3 + original stock closed gripper**, the original
Push Anything/C3+ planner and original OSC torque execution, semantic C1 safety,
and **strictly more than40% success across50 fixed random scenes**. The operative
minimum is21/50:20/50 equals40%. On 2026-09-09 the user requested publishing the
current development baseline before further optimization. That changes the
delivery order, not the simulation acceptance criteria. See the
[development baseline](FR3_SIMULATION_BASELINE.md); a code push is not acceptance.

## Required evidence

All50 prescribed scenes remain in the denominator. A success must finish within
180 simulation seconds, have genuine legal gripper contact, and satisfy all of:

- XY position error <0.02m;
- height error <0.01m;
- full normalized SO(3) orientation error <0.105rad;
- all three conditions continuously for0.5s, independently reconstructed from
  500 consecutive measured1ms physics steps.

Every scene must have zero C1 violations. Missing or invalid evidence prevents
acceptance. The physical robot must remain the matched FR3 with its stock
nominally closed gripper. The frozen closure audit requires zero commanded
finger opening and measured per-finger opening/mismatch within1mm at every
physics step. No manual pushing controller, forced-C3 contact trigger,
diagnostic reference replay, or stationary hold can substitute for execution.

The original torque executor uses schema `nonprehensile.c3_online_isaaclab.v1`.
Formal attribution and acceptance are determined by the separate frozen-batch
execution record and independent audit; legacy diagnostic labels in the raw
executor output are not a certificate of acceptance. The auditor never relabels
these results as the older task-executor schema.

## Fixed scenes and controller

Protocol: `data/manifests/contact_planner_m3/hammer_c1_acceptance50_protocol.json`.
Manifest: `hammer_c1_acceptance50_randomyaw_seed20260909.jsonl`, SHA256
`28154e632c6b48171fff832e4b8f8d4514078e926116322ad77df2658c97bf34`.
Initial yaw spans the full circle; displacement directions are within±90deg
relative to the initial object direction from the robot base. Distances are
balanced over6–10cm, and relative goal-yaw changes over−10,−5,0,5,10deg.

The active batch directory is:

`outputs/contact_planner_m3/workspace_height_20260908/fr3_osc_acceptance50_prepared`

Its `controller_freeze.json` has SHA256
`70b5474bc3dfd373eb83dc9a8fe034935c03bb65e1e5e4b6100bd3a47a94b34d`.
It records196 execution-source files, both native binaries,54 external model,
asset and native-library inputs with archived bytes, and60 runtime dependencies
per scene. Native controller/geometry signatures match across all50 scenes,
excluding only prescribed initial poses, fixed goals and sampling seeds.
Runtime dependency files have been materialized in the scene directories.
The local native runfiles and Isaac installation remain required; this is not
a standalone portable container archive.

## Current execution

Formal execution started **2026-09-09 07:59:42 UTC**. Four workers use manifest
index modulo4 on GPUs2,3,5,6, with tmux sessions `c1_fr3_formal50_0` through`_3`.
Their exact launch commands are recorded in `workers_launched.json`; per-scene
`execution.json` records process attribution, command, selected environment,
freeze digest, completion status and result hash. Existing completed or
interrupted attempts are never automatically rerun or selected best-of.

On 2026-09-09 at08:27 UTC, scene001 terminated after88.150s of recorded physics
steps. The native planner asserted when the measured TCP horizontal radius
reached0.750387m, beyond its frozen0.75m workspace bound. Isaac subsequently
timed out receiving torque. The exception result omitted its in-memory trace
and safety counters, so the independent auditor marks this attempt invalid;
zero C1 violations cannot be certified for it. See
`scene001/native_abort_diagnostic.json` for hashed evidence. This batch therefore
cannot satisfy the complete acceptance contract. All50 attempts remain in the
denominator, and the fixed workers continue to collect their outcomes; shard1
has advanced toscene005. The bound and frozen controller remain unchanged.

The next executor revision must preserve partial failure evidence and accurately
count completed physics steps. It cannot repair or relabel this existing result.
Workspace/convergence failures also require diagnosis before another frozen
acceptance batch can be qualified.

The working-tree executor now preserves CPU-side partial records on exceptions,
including contact decisions and the closed, hashed finger sidecar. It counts
only successfully returned physics steps and separately records incomplete-step,
contact-audit and finger-record coverage. Fourteen relevant CPU tests pass,
including fault injection through the actual exception/finalization blocks.
The normal execution path now has a physical regression: the actual IsaacLab
video rerun of debug scene007 succeeds at14.182s, with all1899 trace rows
identical to the prior successful run. Its exception path has CPU fault-injection
coverage; the failure video rerun is still pending at this snapshot. This revision
is not used by the current frozen workers. Initial evidence:
`exception_evidence_fix_20260909/validation.json` in the parent experiment directory.

The known debugscene007 succeeded at14.182s and passed a complete repeated
closure/pose/contact/model audit with identical recorded trajectories.
That result is **excluded from the formal50**. Prior Panda and Isaac replacement
OSC results are also excluded. See [FR3 alignment](FR3_MODEL_ALIGNMENT.md) for
model, friction and mimic-joint evidence; earlier investigation documents are
historical diagnostics, not current acceptance results.

## Independent audit

The frozen batch tools are in `verification_source`, with their own hash manifest.
`fr3_frozen_batch.py` verifies common and scene-specific inputs before execution.
`audit_fr3_osc_batch.py` verifies the original-OSC schema, torque freshness,
actual FR3 backend/model alignment, actual initial/goal poses and native fixed
goal, dense finger closure, positive legal hand-target force, strict pose dwell
and recorded C1 events. Native goal mode must remain2. Orientation and frame
checks use the pinned semantic support pose and29mm robot-base height offset.

Run the archived auditor with the recorded Python environment, batch root and
freeze digest; it writes a report containing `simulation_acceptance_pass`.
Incomplete ongoing batches return false, retaining all50 in the denominator.
Publishing a development snapshot does not make this simulation report pass.
The goal still requires a qualifying full batch and publication of the final
accepted revision. No true-robot actuation has been authorized or performed.
