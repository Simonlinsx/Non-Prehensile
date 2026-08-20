"""Validated loading for the released DAPL end-effector point cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DAPL_HAND_POINT_COUNT = 256
DAPL_HAND_POINTS_ENV = "DAPL_HAND_POINTS"


def load_dapl_hand_points(path: str | Path) -> torch.Tensor:
    """Load the released ``hand_merged.npy`` as a finite ``[256, 3]`` tensor."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DAPL hand point cache does not exist: {resolved}")
    try:
        array = np.load(resolved, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to load DAPL hand point cache {resolved}: {error}") from error
    if array.shape != (DAPL_HAND_POINT_COUNT, 3):
        raise ValueError(
            f"DAPL hand point cache must have shape ({DAPL_HAND_POINT_COUNT}, 3), "
            f"got {array.shape} from {resolved}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"DAPL hand point cache must be numeric, got {array.dtype}")
    points = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
    if not torch.isfinite(points).all():
        raise ValueError(f"DAPL hand point cache contains non-finite values: {resolved}")
    return points
