"""Sensor simulation: 16-ray planar LiDAR and local 3x3 Geiger counter array."""

from __future__ import annotations

import math
from typing import Tuple, List, Optional
import numpy as np

# Sensor constants
LIDAR_NUM_RAYS: int = 16
LIDAR_MAX_RANGE: float = 15.0
GEIGER_SIZE: int = 9  # 3x3 local neighborhood
OBS_DIM: int = 28     # 16 lidar + 9 geiger + 2 goal_vec + 1 goal_dist


def _build_precomputed_rays(
    num_rays: int = LIDAR_NUM_RAYS,
    max_range: float = LIDAR_MAX_RANGE,
    num_samples: int = 300,
) -> Tuple[Tuple[Tuple[int, int, float], ...], ...]:
    """Precompute the sequence of relative grid cell offsets and Euclidean distances for each ray.
    
    Ray 0 points North (dr = -1, dc = 0) with angles progressing clockwise:
      Ray 0: 0.0°   (North)
      Ray 1: 22.5°
      Ray 2: 45.0°  (North-East)
      Ray 4: 90.0°  (East)
      Ray 6: 135.0° (South-East)
      Ray 8: 180.0° (South)
      Ray 10: 225.0° (South-West)
      Ray 12: 270.0° (West)
      Ray 14: 315.0° (North-West)
    """
    rays = []
    for k in range(num_rays):
        theta = k * (2.0 * math.pi / num_rays)
        dr_unit = -math.cos(theta)
        dc_unit = math.sin(theta)
        ray_cells = []
        seen = {(0, 0)}
        for s in np.linspace(0.1, max_range, num_samples):
            cr = int(round(s * dr_unit))
            cc = int(round(s * dc_unit))
            if (cr, cc) not in seen:
                dist = math.sqrt(cr * cr + cc * cc)
                if dist <= max_range:
                    seen.add((cr, cc))
                    ray_cells.append((cr, cc, dist))
        rays.append(tuple(ray_cells))
    return tuple(rays)


PRECOMPUTED_RAYS = _build_precomputed_rays()


def compute_lidar_distances(
    robot_pos: Tuple[int, int],
    walls: np.ndarray,
    shape: Tuple[int, int] = (50, 50),
    max_range: float = LIDAR_MAX_RANGE,
) -> np.ndarray:
    """Cast 16 planar LiDAR beams and return normalized distances in [0, 1].
    
    If an obstacle or boundary is intersected, the normalized distance (d / max_range)
    is returned; if no obstacle is within max_range, returns 1.0.
    """
    r, c = robot_pos
    H, W = shape
    res = np.empty(LIDAR_NUM_RAYS, dtype=np.float32)

    for k in range(LIDAR_NUM_RAYS):
        hit_dist = max_range
        for dr, dc, dist in PRECOMPUTED_RAYS[k]:
            nr = r + dr
            nc = c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W or walls[nr, nc]:
                hit_dist = dist
                break
        res[k] = hit_dist / max_range

    return res


def compute_geiger_readings(
    robot_pos: Tuple[int, int],
    walls: np.ndarray,
    radiation: np.ndarray,
    shape: Tuple[int, int] = (50, 50),
) -> np.ndarray:
    """Sample radiation intensity at the robot's cell and its 8 Moore neighbors (3x3).
    
    Order is row-major from top-left to bottom-right:
      (-1, -1), (-1, 0), (-1, 1),
      ( 0, -1), ( 0, 0), ( 0, 1),
      ( 1, -1), ( 1, 0), ( 1, 1)
      
    Any cell outside the grid or inside an obstacle wall reads strictly 0.0.
    """
    r, c = robot_pos
    H, W = shape
    out = np.zeros(GEIGER_SIZE, dtype=np.float32)
    idx = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W and not walls[nr, nc]:
                out[idx] = float(radiation[nr, nc])
            idx += 1
    return out


def compute_goal_features(
    robot_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    shape: Tuple[int, int] = (50, 50),
) -> np.ndarray:
    """Compute relative goal direction and normalized Euclidean distance.
    
    Features (3 numbers):
      - dr / 50.0  (vertical offset normalized)
      - dc / 50.0  (horizontal offset normalized)
      - dist / (50 * sqrt(2))  (Euclidean distance normalized by max diagonal)
    """
    dr = float(goal_pos[0] - robot_pos[0])
    dc = float(goal_pos[1] - robot_pos[1])
    dist = math.sqrt(dr * dr + dc * dc)
    max_diag = 50.0 * math.sqrt(2.0)
    return np.array([dr / 50.0, dc / 50.0, dist / max_diag], dtype=np.float32)


def get_sensor_observation(
    robot_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    walls: np.ndarray,
    radiation: np.ndarray,
    shape: Tuple[int, int] = (50, 50),
) -> np.ndarray:
    """Return the complete 28-dimensional egocentric observation vector.
    
    Indices:
      [0:16]  : 16-ray LiDAR distances in [0, 1]
      [16:25] : 3x3 Geiger radiation readings (wall/OOB = 0)
      [25:27] : (dr, dc) / 50.0 relative goal vector
      [27]    : Euclidean goal distance / (50 * sqrt(2))
    """
    lidar = compute_lidar_distances(robot_pos, walls, shape=shape)
    geiger = compute_geiger_readings(robot_pos, walls, radiation, shape=shape)
    goal_feat = compute_goal_features(robot_pos, goal_pos, shape=shape)
    return np.concatenate([lidar, geiger, goal_feat], axis=0).astype(np.float32)
