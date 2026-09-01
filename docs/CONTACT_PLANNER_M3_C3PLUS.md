# Contact Planner M3: Semantic C3+ Integration

## Decision

M3 replaces the unsuccessful fixed straight-push action family from M2 with
the open-source Push Anything sampling + C3+ controller.  Isaac Lab remains the
independent evaluator for strict target pose, C1, C2, and C3.  M2 remains a
documented ablation and is not extended with more scalar rewards or waypoints.

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

The first patch adds an optional per-object `sampling_meshes` list.  It changes
only the global candidate surface; the complete physical/contact object model
is intentionally retained.  The second patch is a one-line build compatibility
fix for C3's no-Gurobi stub: it matches the Gurobi implementation's existing
`options.M.value_or(1000)` conversion.  The C3+ algorithm is unchanged.

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
        `--> Isaac Lab replay and C1/C2/C3 acceptance
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

This is intentionally conservative at part boundaries.  Generate the three
object-local metre-scale OBJ files and their provenance manifest with:

```bash
PYTHONPATH=source/IsaacLab_nonPrehensile \
python scripts/export_domino_semantic_mesh.py \
  --domino-root /data1/linsixu/DOMINO \
  --asset-id 020_hammer:0 \
  --output-dir data/push_anything_semantics/020_hammer_0
```

The manifest records the source SHA-256, scale, thresholds, class counts, and
the original face indices.  Degenerate source triangles are removed explicitly
and their count is recorded.  A target export fails if either the safe or the
protected partition is empty.

## Constraint mapping

| Contract | C3+ change | Isaac Lab acceptance |
| --- | --- | --- |
| C1 robot-target | sample only safe faces; prohibit EE-neutral and EE-protected contact throughout the MPC horizon | zero forbidden/neutral/protected robot-target contacts |
| C2 clutter-target | require positive clearance for clutter-protected contact pairs | zero protected-region clutter contacts |
| C3 robot-clutter | prohibit EE-clutter contact in C3+ and check every arm link during reposition/execution | zero robot-clutter contacts |

Filtering the global sampler alone is not sufficient for C1: the continuous
trajectory must also keep neutral/protected gaps positive.  Likewise, the
original Push Anything object-object contacts are deliberately enabled and
cannot be reported as satisfying C2 without the protected-part constraints.

## Acceptance gates

The first controller gate remains one hammer, no clutter:

```text
XY error              < 0.020 m
height error          < 0.010 m
full SO(3) error      < 0.100 rad
dwell                 >= 5 steps
C1 violations         = 0
```

Only after this passes across randomized single-hammer scenes do C2 and C3
enter the controller.  Pose success and semantic safety are always reported
separately, followed by their conjunction.

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

Afterwards, check and build only the three binaries needed for the simulation:

```bash
cd /data1/linsixu/IsaacLab-nonPrehensile
bash scripts/build_push_anything_native.sh --check
bash scripts/build_push_anything_native.sh --build
```

The wrapper pins the audited upstream commit, keeps Bazel output on `/data1`,
adds the user-local OpenBLAS library and runtime path, explicitly disables
Gurobi, and avoids the much larger `bazel build ...` target.  Keep at least
roughly 30 GiB free for the pinned Drake source build.

### Verified on this host

On 2026-09-01, the user-local build completed all 12,587 actions for:

- `//examples/sampling_c3:franka_sim`
- `//examples/sampling_c3:franka_osc_controller`
- `//examples/sampling_c3:franka_sampling_c3_controller`

`ldd` resolves Drake, the Bazel-built LCM library, and the user-local
OpenBLAS library.  A ten-second `franka_sim --demo_name=anything` launch check
remained live until the intentional timeout.  This proves the native runtime
foundation; it does not yet prove the single-hammer pose or C1 acceptance gate.
