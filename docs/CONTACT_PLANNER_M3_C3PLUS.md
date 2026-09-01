# Contact Planner M3: Semantic C3+ Integration

## Decision

M3 replaces the unsuccessful fixed straight-push action family from M2 with
the open-source Push Anything sampling + C3+ controller.  A repository-owned
monitor and semantic trajectory auditor independently evaluate strict target
pose and C1.  Isaac Lab remains the later evaluator for clutter, whole-arm C3,
and cross-simulator robustness.  M2 remains a documented ablation and is not
extended with more scalar rewards or waypoints.

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

Push Anything's public controller samples mesh-normal EE locations, solves a
seven-step local C3+ problem for each candidate, and switches between a
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
| C2 clutter-target | planned: positive clearance for clutter-protected pairs | planned Isaac Lab contact audit |
| C3 robot-clutter | planned: prohibit EE-clutter contact and check every arm link | planned Isaac Lab contact audit |

Filtering the global sampler alone is not sufficient for C1: the continuous
trajectory must also keep neutral/protected gaps positive.  Likewise, the
original Push Anything object-object contacts are deliberately enabled and
cannot be reported as satisfying C2 without the protected-part constraints.

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
passes; randomized single-hammer robustness is the next gate before C2/C3.

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

The next robustness gate keeps the same hammer, support face, and no-clutter C1
contract while varying the task geometry.  The checked-in manifest
`hammer_c1_front180_eval50_seed20260901.jsonl` contains exactly 50 deterministic
scenes:

- initial X: 0.39, 0.40, or 0.41 m;
- initial Y: 0.18 through 0.22 m in 1 cm increments;
- goal distance: 0.06 through 0.10 m in 1 cm increments, ten scenes each;
- goal direction: stratified across the robot-facing feasible hemisphere,
  from -88.2 to +88.2 degrees;
- relative goal yaw: -10, -5, 0, +5, or +10 degrees, ten scenes each;
- independent deterministic contact-sampling seed per scene.

Run the resumable evaluator with:

```bash
/usr/bin/python3 scripts/evaluate_push_anything_c1_randomized.py \
  --manifest data/manifests/contact_planner_m3/hammer_c1_front180_eval50_seed20260901.jsonl \
  --output-root outputs/contact_planner_m3/hammer_c1_front180_eval50_seed20260901 \
  --tcpq-port 7730 --timeout-s 180
```

`summary.json` is rewritten after every scene.  It reports geometry, C1, and
their conjunction separately, including four goal-direction bins.  Existing
completed scene artifacts are reused on restart unless `--rerun` is supplied.

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

Afterwards, check and build only the four targets needed for the simulation and
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

On 2026-09-01, the user-local build completed all 12,587 actions for:

- `//examples/sampling_c3:franka_sim`
- `//examples/sampling_c3:franka_osc_controller`
- `//examples/sampling_c3:franka_sampling_c3_controller`
- `//examples/sampling_c3:monitor_push_anything_baseline`

`ldd` resolves Drake, the Bazel-built LCM library, and the user-local OpenBLAS
library.  The deterministic single-hammer pose+C1 gate above also completed in
the native, sudo-free runtime.  This is one accepted controller instance, not
yet a claim of randomized-scene robustness, C2, C3, or real-robot success.
