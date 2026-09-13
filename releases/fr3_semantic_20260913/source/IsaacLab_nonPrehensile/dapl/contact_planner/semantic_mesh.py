"""Conservative mesh-face semantics for contact-implicit planning.

Push Anything samples end-effector locations from triangle faces and builds
contact pairs from collision geometries.  The existing DOMINO integration,
on the other hand, exposes point-wise ``[safe, protected]`` scores.  This
module provides the simulator-independent bridge between those interfaces.

Classification is deliberately fail-closed:

* a face is protected when *any* sampled location is protected;
* a face is safe only when *every* sampled location is safe and none is
  protected;
* every uncertain or mixed face is neutral.

That contract prevents a triangle spanning an affordance boundary from being
accepted merely because its centroid lies on the safe side.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from ..domino import DominoAffordanceAnnotation, domino_point_affordance_features


class FaceSemantic(IntEnum):
    """Integer labels stored in exported semantic-mesh manifests."""

    NEUTRAL = 0
    SAFE = 1
    PROTECTED = 2


@dataclass(frozen=True)
class SemanticFacePartition:
    """Per-face scores and mutually-exclusive conservative labels."""

    labels: torch.Tensor
    minimum_safe_scores: torch.Tensor
    maximum_protected_scores: torch.Tensor
    samples_per_face: int

    def __post_init__(self) -> None:
        face_count = self.labels.numel()
        if self.labels.ndim != 1 or self.labels.dtype != torch.long:
            raise ValueError("labels must be a one-dimensional torch.long tensor")
        for name, scores in (
            ("minimum_safe_scores", self.minimum_safe_scores),
            ("maximum_protected_scores", self.maximum_protected_scores),
        ):
            if scores.shape != (face_count,) or not scores.is_floating_point():
                raise ValueError(f"{name} must have shape [faces] and floating dtype")
        if self.samples_per_face <= 0:
            raise ValueError("samples_per_face must be positive")
        valid = (
            (self.labels == int(FaceSemantic.NEUTRAL))
            | (self.labels == int(FaceSemantic.SAFE))
            | (self.labels == int(FaceSemantic.PROTECTED))
        )
        if not bool(torch.all(valid)):
            raise ValueError("labels contain an unknown face semantic")

    def mask(self, semantic: FaceSemantic) -> torch.Tensor:
        """Return the boolean mask for one semantic class."""

        return self.labels == int(semantic)

    def counts(self) -> dict[str, int]:
        """Return JSON-friendly mutually-exclusive class counts."""

        return {
            semantic.name.lower(): int(self.mask(semantic).sum().item())
            for semantic in FaceSemantic
        }


def _validate_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [vertices, 3]")
    if not vertices.is_floating_point():
        raise ValueError("vertices must use a floating-point dtype")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [faces, 3]")
    if faces.dtype not in (torch.int32, torch.int64):
        raise ValueError("faces must use an integer dtype")
    if vertices.shape[0] < 3 or faces.shape[0] < 1:
        raise ValueError("mesh must contain at least three vertices and one face")
    if not bool(torch.all(torch.isfinite(vertices))):
        raise ValueError("vertices must be finite")
    if int(faces.min().item()) < 0 or int(faces.max().item()) >= vertices.shape[0]:
        raise ValueError("faces contain an out-of-range vertex index")

    triangles = vertices[faces.to(dtype=torch.long)]
    doubled_area = torch.linalg.vector_norm(
        torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=-1,
        ),
        dim=-1,
    )
    # Machine epsilon is a relative precision measure, not a geometric area
    # threshold.  Small but valid detail triangles may legitimately have area
    # below float32 epsilon in the raw object frame.
    if bool(torch.any(doubled_area <= torch.finfo(vertices.dtype).tiny)):
        raise ValueError("mesh contains a degenerate triangle")


def triangle_semantic_samples(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Return vertices, edge midpoints, and centroid for every triangle.

    The result has shape ``[faces, 7, 3]`` and remains in the input mesh frame.
    Sampling more than the centroid is important at safe/protected boundaries.
    """

    _validate_mesh(vertices, faces)
    triangles = vertices[faces.to(dtype=torch.long)]
    edge_midpoints = torch.stack(
        (
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ),
        dim=1,
    )
    centroids = triangles.mean(dim=1, keepdim=True)
    return torch.cat((triangles, edge_midpoints, centroids), dim=1)


def partition_mesh_faces(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    annotation: DominoAffordanceAnnotation,
    *,
    safe_threshold: float = 0.25,
    protected_threshold: float = 0.25,
    safe_radius_m: float | None = None,
    protected_radius_m: float | None = None,
) -> SemanticFacePartition:
    """Map a raw DOMINO mesh into safe, protected, and neutral faces.

    ``vertices`` must use the same unscaled object frame as the DOMINO mesh and
    annotations.  Scale is handled inside :func:`domino_point_affordance_features`.
    """

    for name, value in (
        ("safe_threshold", safe_threshold),
        ("protected_threshold", protected_threshold),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    samples = triangle_semantic_samples(vertices, faces)
    features = domino_point_affordance_features(
        samples,
        annotation,
        safe_radius_m=safe_radius_m,
        protected_radius_m=protected_radius_m,
    )
    minimum_safe = features[..., 0].amin(dim=1)
    maximum_protected = features[..., 1].amax(dim=1)

    protected = maximum_protected >= float(protected_threshold)
    safe = (minimum_safe >= float(safe_threshold)) & ~protected
    labels = torch.full(
        (faces.shape[0],),
        int(FaceSemantic.NEUTRAL),
        dtype=torch.long,
        device=vertices.device,
    )
    labels[safe] = int(FaceSemantic.SAFE)
    labels[protected] = int(FaceSemantic.PROTECTED)
    return SemanticFacePartition(
        labels=labels,
        minimum_safe_scores=minimum_safe,
        maximum_protected_scores=maximum_protected,
        samples_per_face=samples.shape[1],
    )
