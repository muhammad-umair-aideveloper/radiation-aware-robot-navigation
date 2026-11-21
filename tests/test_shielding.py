"""Plain-assert tests for wall shielding models."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shielding import attenuation_matrix, shielded_field


def test_attenuation_and_shielded_field() -> None:
    wall = np.zeros((5, 5), dtype=bool)
    wall[2, :] = True
    atten = attenuation_matrix(wall, mu=1.0, samples=20)
    assert atten.shape == (25, 25)
    assert np.all((atten >= 0.0) & (atten <= 1.0))
    assert np.isclose(atten[12, 12], 1.0)

    field = shielded_field([(0, 0, 10.0)], wall, atten)
    assert field.shape == (5, 5)
    assert np.isfinite(field).all()
    assert field[0, 0] > field[4, 4]
