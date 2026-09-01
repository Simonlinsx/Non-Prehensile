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
```

The first patch adds an optional per-object `sampling_meshes` list.  It changes
only the global candidate surface; the complete physical/contact object model
is intentionally retained.

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

## Current environment limitation

The official upstream source is available locally, but this host currently has
neither Docker nor Bazel.  Therefore the official C3+ simulation has not yet
been built or reproduced here.  The public instructions use the
`xuanhien070594/push-anything-env:latest` container; enabling a compatible
container runtime or provisioning the pinned Drake/Bazel toolchain is a hard
dependency for the next reproduction gate.
