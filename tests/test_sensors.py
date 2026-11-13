"""Unit tests for sensor modules: LiDAR ray casting and Geiger counter readings."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sensors import (
    compute_lidar_distances,
    compute_geiger_readings,
    compute_goal_features,
    get_sensor_observation,
    LIDAR_NUM_RAYS,
    LIDAR_MAX_RANGE,
    GEIGER_SIZE,
    OBS_DIM,
)


def test_lidar_known_distances_four_directions() -> None:
    """Verify LiDAR distances along four cardinal directions on a hand-built grid."""
    shape = (25, 25)
    walls = np.zeros(shape, dtype=bool)
    robot = (10, 10)

    # Place walls at exact known distances
    walls[6, 10] = True   # North: 4 cells away
    walls[10, 15] = True  # East:  5 cells away
    walls[13, 10] = True  # South: 3 cells away
    walls[10, 3] = True   # West:  7 cells away

    lidar = compute_lidar_distances(robot, walls, shape=shape)

    assert len(lidar) == LIDAR_NUM_RAYS, f"Expected {LIDAR_NUM_RAYS} rays, got {len(lidar)}"

    # Ray 0: North
    expected_n = 4.0 / LIDAR_MAX_RANGE
    assert math.isclose(lidar[0], expected_n, rel_tol=1e-5), f"North ray mismatch: {lidar[0]} vs {expected_n}"

    # Ray 4: East
    expected_e = 5.0 / LIDAR_MAX_RANGE
    assert math.isclose(lidar[4], expected_e, rel_tol=1e-5), f"East ray mismatch: {lidar[4]} vs {expected_e}"

    # Ray 8: South
    expected_s = 3.0 / LIDAR_MAX_RANGE
    assert math.isclose(lidar[8], expected_s, rel_tol=1e-5), f"South ray mismatch: {lidar[8]} vs {expected_s}"

    # Ray 12: West
    expected_w = 7.0 / LIDAR_MAX_RANGE
    assert math.isclose(lidar[12], expected_w, rel_tol=1e-5), f"West ray mismatch: {lidar[12]} vs {expected_w}"

    print("PASS: test_lidar_known_distances_four_directions")


def test_lidar_diagonal_ray() -> None:
    """Verify LiDAR distance along a diagonal ray (45 deg, North-East)."""
    shape = (25, 25)
    walls = np.zeros(shape, dtype=bool)
    robot = (10, 10)

    # Place wall at (7, 13) -> dr = -3, dc = +3 (NE diagonal)
    walls[7, 13] = True
    expected_dist = math.sqrt((-3.0)**2 + 3.0**2)  # 3 * sqrt(2)
    expected_norm = expected_dist / LIDAR_MAX_RANGE

    lidar = compute_lidar_distances(robot, walls, shape=shape)

    # Ray 2: North-East (45 degrees)
    assert math.isclose(lidar[2], expected_norm, rel_tol=1e-5), (
        f"NE diagonal ray mismatch: {lidar[2]} vs {expected_norm}"
    )

    print("PASS: test_lidar_diagonal_ray")


def test_lidar_grid_border() -> None:
    """Verify LiDAR accurately detects grid boundaries when no walls are in the way."""
    shape = (10, 10)
    walls = np.zeros(shape, dtype=bool)
    robot = (2, 5)

    lidar = compute_lidar_distances(robot, walls, shape=shape)

    # Ray 0 (North): Robot at row 2. Cells (1, 5) and (0, 5) are inside; cell (-1, 5) is out-of-bounds.
    # Out of bounds occurs at distance 3.0 cells.
    expected_n = 3.0 / LIDAR_MAX_RANGE
    assert math.isclose(lidar[0], expected_n, rel_tol=1e-5), f"Border North mismatch: {lidar[0]} vs {expected_n}"

    # Ray 12 (West): Robot at col 5. Out of bounds at col -1 (distance 6.0 cells).
    expected_w = 6.0 / LIDAR_MAX_RANGE
    assert math.isclose(lidar[12], expected_w, rel_tol=1e-5), f"Border West mismatch: {lidar[12]} vs {expected_w}"

    print("PASS: test_lidar_grid_border")


def test_geiger_layout_and_wall_masking() -> None:
    """Verify 3x3 Geiger counter layout, neighbor mapping, and zeroing on wall cells."""
    shape = (20, 20)
    walls = np.zeros(shape, dtype=bool)
    radiation = np.ones(shape, dtype=np.float64) * 42.0

    robot = (10, 10)
    # Set wall at North neighbor (9, 10)
    walls[9, 10] = True
    # Custom radiation value at South neighbor (11, 10)
    radiation[11, 10] = 77.5
    # Custom radiation value at robot's own cell
    radiation[10, 10] = 12.3

    readings = compute_geiger_readings(robot, walls, radiation, shape=shape)

    assert len(readings) == GEIGER_SIZE, f"Expected {GEIGER_SIZE} readings, got {len(readings)}"

    # Order:
    # 0: (-1, -1), 1: (-1, 0), 2: (-1, 1)
    # 3: ( 0, -1), 4: ( 0, 0), 5: ( 0, 1)
    # 6: ( 1, -1), 7: ( 1, 0), 8: ( 1, 1)

    # North neighbor (index 1) is a wall, so it MUST read 0.0
    assert readings[1] == 0.0, f"Expected wall cell to read 0.0, got {readings[1]}"

    # Center cell (index 4) should read 12.3
    assert math.isclose(readings[4], 12.3, rel_tol=1e-5), f"Center reading mismatch: {readings[4]}"

    # South neighbor (index 7) should read 77.5
    assert math.isclose(readings[7], 77.5, rel_tol=1e-5), f"South reading mismatch: {readings[7]}"

    # Free neighbor (e.g. index 0) should read default radiation 42.0
    assert math.isclose(readings[0], 42.0, rel_tol=1e-5), f"Free neighbor reading mismatch: {readings[0]}"

    # Test corner robot pos (0, 0) where neighbors are out-of-bounds
    corner_readings = compute_geiger_readings((0, 0), walls, radiation, shape=shape)
    # Indices with dr < 0 (0, 1, 2) or dc < 0 (3, 6) must read 0.0
    for oob_idx in [0, 1, 2, 3, 6]:
        assert corner_readings[oob_idx] == 0.0, f"OOB index {oob_idx} expected 0.0, got {corner_readings[oob_idx]}"
    # Center (4) should read 42.0
    assert math.isclose(corner_readings[4], 42.0, rel_tol=1e-5)

    print("PASS: test_geiger_layout_and_wall_masking")


def test_full_observation_vector() -> None:
    """Verify total observation vector length (28), bounds, and component assembly."""
    shape = (50, 50)
    walls = np.zeros(shape, dtype=bool)
    radiation = np.zeros(shape, dtype=np.float64)
    robot = (5, 5)
    goal = (45, 45)

    obs = get_sensor_observation(robot, goal, walls, radiation, shape=shape)

    assert obs.shape == (OBS_DIM,), f"Expected observation shape ({OBS_DIM},), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32 dtype, got {obs.dtype}"

    # All LiDAR in [0, 1]
    assert np.all(obs[:16] >= 0.0) and np.all(obs[:16] <= 1.0)
    # Geiger in [0, inf)
    assert np.all(obs[16:25] >= 0.0)
    # Relative goal vector in [-1, 1]
    assert np.all(obs[25:27] >= -1.0) and np.all(obs[25:27] <= 1.0)
    # Goal distance in [0, 1]
    assert 0.0 <= obs[27] <= 1.0

    print("PASS: test_full_observation_vector")


def run_all_tests() -> None:
    print("Running sensor unit tests...")
    test_lidar_known_distances_four_directions()
    test_lidar_diagonal_ray()
    test_lidar_grid_border()
    test_geiger_layout_and_wall_masking()
    test_full_observation_vector()
    print("All sensor tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()

