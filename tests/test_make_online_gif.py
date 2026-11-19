"""Plain-assert tests for the online GIF helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from make_online_gif import dose_upto, frame_steps, reconstruct_readings
from src.environment import RadiationEnv
from src.online_sim import run_online_episode


def test_frame_steps() -> None:
    assert frame_steps(10) == [0, 3, 6, 9, 10]
    assert frame_steps(9) == [0, 3, 6, 9]
    assert frame_steps(0) == [0]


def test_dose_upto() -> None:
    straight = [(0, 0), (0, 1), (0, 2), (0, 3)]
    diagonal = [(0, 0), (1, 1), (2, 2), (3, 3)]
    ones = np.ones((4, 4), dtype=float)
    assert dose_upto(straight, ones, 3) == 3.0
    assert abs(dose_upto(diagonal, ones, 3) - 3.0 * np.sqrt(2.0)) < 1e-12


def test_reconstruct_readings() -> None:
    shape = (12, 12)
    free = np.ones(shape, dtype=bool)
    free[4:7, 5:7] = False
    rows, cols = np.indices(shape)
    field = 50.0 / ((rows - 6) ** 2 + (cols - 6) ** 2 + 2.0)
    episode = run_online_episode(
        field,
        free,
        (0, 0),
        (11, 11),
        "idw",
        "3x3",
        1.0,
        3,
    )
    path = episode["path"]
    reconstructed = reconstruct_readings(
        path, field, free, "3x3", 1.0, 3, len(path) - 1
    )
    returned = episode["readings"]
    assert set(reconstructed) == set(returned)
    for key in reconstructed:
        assert abs(
            float(np.mean(reconstructed[key]))
            - float(np.mean(returned[key]))
        ) < 1e-9


if __name__ == "__main__":
    test_frame_steps()
    test_dose_upto()
    test_reconstruct_readings()
    print("PASS: online GIF helper tests")
