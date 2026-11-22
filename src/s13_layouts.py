"""Generate and validate the fixed layouts used by the S13 replication study."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import generate_evaluation_starts
from src.layouts import check_connectivity, generate_random_layout


LAYOUT_SEEDS: List[int] = list(range(200, 220))
EARLIER_LAYOUT_SEEDS: set[int] = set(range(100, 110))
STARTS_PER_LAYOUT: int = 10


def generate_s13_layout_set() -> Dict[str, object]:
    """Create layouts and fixed starts, with a validity report."""
    layouts: Dict[str, object] = {}
    checks: List[Dict[str, object]] = []

    for seed in LAYOUT_SEEDS:
        layout = generate_random_layout(seed=seed)
        env = layout.to_env()
        starts = generate_evaluation_starts(env, STARTS_PER_LAYOUT, seed=seed * 10 + 7)
        distances = []
        free_distances = generate_free_space_distances(env)
        for start in starts:
            distances.append(float(free_distances[start]))

        source_positions = [position for position, _ in layout.sources]
        layout_check = {
            "seed": seed,
            "wall_fraction": float(layout.wall_fraction),
            "connected": bool(check_connectivity(layout.walls, layout.start, layout.goal)),
            "wall_fraction_valid": bool(0.10 <= layout.wall_fraction <= 0.25),
            "source_count_valid": len(layout.sources) == 3,
            "sources_free_valid": all(
                not layout.walls[position] and position != layout.start and position != layout.goal
                for position in source_positions
            ),
            "source_intensity_valid": all(
                120.0 <= intensity <= 200.0 for _, intensity in layout.sources
            ),
            "start_count_valid": len(starts) == STARTS_PER_LAYOUT,
            "start_distance_valid": all(distance >= 20.0 for distance in distances),
            "start_free_valid": all(not layout.walls[start] for start in starts),
            "seed_disjoint_valid": seed not in EARLIER_LAYOUT_SEEDS,
        }
        checks.append(layout_check)
        layouts[str(seed)] = {
            "seed": seed,
            "shape": [int(layout.shape[0]), int(layout.shape[1])],
            "wall_fraction": float(layout.wall_fraction),
            "sources": [
                {"position": [int(position[0]), int(position[1])], "i0": float(intensity)}
                for position, intensity in layout.sources
            ],
            "start": [int(layout.start[0]), int(layout.start[1])],
            "goal": [int(layout.goal[0]), int(layout.goal[1])],
            "starts": [[int(start[0]), int(start[1])] for start in starts],
        }

    passed = all(all(value for key, value in check.items() if key != "seed" and key != "wall_fraction")
                 for check in checks)
    return {
        "metadata": {
            "layout_seeds": LAYOUT_SEEDS,
            "earlier_layout_seeds": sorted(EARLIER_LAYOUT_SEEDS),
            "starts_per_layout": STARTS_PER_LAYOUT,
            "minimum_free_distance": 20.0,
            "wall_fraction_range": [0.10, 0.25],
            "source_intensity_range": [120.0, 200.0],
            "source_count": 3,
        },
        "validity": {"passed": bool(passed), "layouts": checks},
        "layouts": layouts,
    }


def generate_free_space_distances(env) -> np.ndarray:
    """Compute exact 8-connected free-space distance to the layout goal."""
    import heapq
    from src.environment import ACTION_LENGTHS, ACTIONS

    dist = np.full(env.shape, np.inf, dtype=np.float64)
    dist[env.goal] = 0.0
    queue: List[Tuple[float, int, int]] = [(0.0, env.goal[0], env.goal[1])]
    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance > dist[row, col]:
            continue
        for action_index, (dr, dc) in enumerate(ACTIONS):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < env.shape[0] and 0 <= nc < env.shape[1]):
                continue
            if env.walls[nr, nc]:
                continue
            if abs(dr) == 1 and abs(dc) == 1:
                if env.walls[row + dr, col] or env.walls[row, col + dc]:
                    continue
            new_distance = distance + ACTION_LENGTHS[action_index]
            if new_distance < dist[nr, nc]:
                dist[nr, nc] = new_distance
                heapq.heappush(queue, (new_distance, nr, nc))
    return dist


def main() -> None:
    result = generate_s13_layout_set()
    output = ROOT / "results" / "s13_layouts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["validity"], indent=2))
    if not result["validity"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
