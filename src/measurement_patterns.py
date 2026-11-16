"""Measurement-cell patterns for online radiation sensing."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def pattern_cells(
    cell: Tuple[int, int], free_mask: np.ndarray, pattern: str
) -> np.ndarray:
    """Return free cells covered by a named measurement pattern."""
    r, c = cell
    if pattern == "cell":
        offsets = [(0, 0)]
    elif pattern == "3x3":
        offsets = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        ]
    else:
        raise ValueError(f"Unknown measurement pattern: {pattern}")

    mask = np.asarray(free_mask, dtype=bool)
    cells = [
        (r + dr, c + dc)
        for dr, dc in offsets
        if 0 <= r + dr < mask.shape[0]
        and 0 <= c + dc < mask.shape[1]
        and mask[r + dr, c + dc]
    ]
    return np.array(cells, dtype=int).reshape(-1, 2)
