"""Focused validation tests for public simulation and estimation inputs."""

import numpy as np
import pytest

from src.estimators import (
    idw,
    parametric_sources,
    parametric_sources_shielded,
)
from src.geiger_model import counts_to_intensity, simulate_counts
from src.measurement_patterns import pattern_cells
from src.online_sim import estimate_field, inflate_unknown, run_online_episode
from src.shielding import attenuation_matrix, shielded_field
from src.survey import reachable_free_cells, survey_waypoints


def test_estimator_rejects_cells_outside_mask() -> None:
    with pytest.raises(ValueError, match="within the free_mask shape"):
        idw(np.array([[3, 0]]), np.array([1.0]), np.ones((3, 3), dtype=bool))


def test_estimator_rejects_non_finite_measurements() -> None:
    with pytest.raises(ValueError, match="finite"):
        idw(np.array([[0, 0]]), np.array([np.nan]), np.ones((3, 3), dtype=bool))


def test_parametric_estimator_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        parametric_sources(
            np.array([[0, 0]]), np.array([1.0]), np.ones((2, 2), dtype=bool), k=0
        )


def test_shielded_estimator_requires_matching_attenuation_shape() -> None:
    with pytest.raises(ValueError, match="atten must have shape"):
        parametric_sources_shielded(
            np.array([[0, 0]]),
            np.array([1.0]),
            np.ones((2, 2), dtype=bool),
            np.ones((3, 3)),
        )


def test_measurement_requires_positive_dwell() -> None:
    with pytest.raises(ValueError, match="dwell must be positive"):
        simulate_counts(np.ones((2, 2)), np.array([[0, 0]]), 0.0, seed=0)


def test_count_conversion_requires_positive_dwell() -> None:
    with pytest.raises(ValueError, match="dwell must be positive"):
        counts_to_intensity(np.array([1.0]), -1.0)


def test_measurement_rejects_malformed_cell_shape() -> None:
    with pytest.raises(ValueError, match=r"cells must have shape \(n, 2\)"):
        simulate_counts(np.ones((2, 2)), np.array([0, 0]), 1.0, seed=0)


def test_measurement_rejects_cells_outside_field() -> None:
    with pytest.raises(ValueError, match="within the field shape"):
        simulate_counts(np.ones((2, 2)), np.array([[2, 0]]), 1.0, seed=0)


def test_measurement_pattern_requires_two_dimensional_mask() -> None:
    with pytest.raises(ValueError, match="free_mask must be two-dimensional"):
        pattern_cells((0, 0), np.ones(3, dtype=bool), "cell")


def test_measurement_pattern_rejects_out_of_bounds_cell() -> None:
    with pytest.raises(ValueError, match="within the free_mask shape"):
        pattern_cells((2, 0), np.ones((2, 2), dtype=bool), "cell")


def test_survey_rejects_out_of_bounds_start() -> None:
    with pytest.raises(ValueError, match="start must be within"):
        reachable_free_cells((3, 0), np.ones((3, 3), dtype=bool))


def test_survey_waypoints_rejects_one_dimensional_mask() -> None:
    with pytest.raises(ValueError, match="free_mask must be two-dimensional"):
        survey_waypoints((0, 0), np.ones(3, dtype=bool))


def test_shielding_rejects_negative_attenuation_coefficient() -> None:
    with pytest.raises(ValueError, match="mu must be non-negative"):
        attenuation_matrix(np.zeros((2, 2), dtype=bool), mu=-1.0)


def test_shielded_field_rejects_mismatched_attenuation_shape() -> None:
    with pytest.raises(ValueError, match="atten must have shape"):
        shielded_field([], np.zeros((2, 2), dtype=bool), np.ones((2, 2)))


def test_shielded_field_rejects_negative_source_intensity() -> None:
    with pytest.raises(ValueError, match="source intensity must be non-negative"):
        shielded_field(
            [(0, 0, -1.0)],
            np.zeros((2, 2), dtype=bool),
            np.ones((4, 4)),
        )


def test_online_estimation_rejects_empty_measurements() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        estimate_field({}, np.ones((2, 2), dtype=bool), "idw")


def test_online_inflation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        inflate_unknown(
            np.zeros((2, 3)),
            {(0, 0): [1.0]},
            np.ones((2, 2), dtype=bool),
            beta=1.0,
        )


def test_online_episode_requires_matching_field_and_mask_shapes() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        run_online_episode(
            np.ones((2, 3)),
            np.ones((2, 2), dtype=bool),
            (0, 0),
            (1, 1),
            "idw",
            "cell",
            1.0,
            seed=0,
        )


def test_online_episode_requires_positive_dwell() -> None:
    with pytest.raises(ValueError, match="dwell must be positive"):
        run_online_episode(
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            (0, 0),
            (1, 1),
            "idw",
            "cell",
            0.0,
            seed=0,
        )


def test_online_episode_requires_free_endpoints() -> None:
    mask = np.ones((2, 2), dtype=bool)
    mask[1, 1] = False
    with pytest.raises(ValueError, match="start and goal must be free"):
        run_online_episode(
            np.ones((2, 2)),
            mask,
            (0, 0),
            (1, 1),
            "idw",
            "cell",
            1.0,
            seed=0,
        )
