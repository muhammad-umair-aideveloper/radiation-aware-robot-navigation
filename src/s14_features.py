"""Compute pre-declared layout features for S14 failure analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RadiationEnv
from src.layouts import generate_random_layout
from src.planners import astar_search
from src.s13_study import path_length
S13_SUMMARY = ROOT / "results" / "s13_summary.json"
S13_LAYOUTS = ROOT / "results" / "s13_layouts.json"
OUTPUT_JSON = ROOT / "results" / "s14_features.json"
OUTPUT_MD = ROOT / "results" / "s14_features.md"


def compute_wall_fraction(walls: np.ndarray) -> float:
    """Fraction of grid cells occupied by walls."""
    return float(np.mean(walls))


def compute_min_source_distance(sources: Sequence[tuple[float, float] | Sequence[float]]) -> float:
    """Minimum Euclidean distance between any pair of sources."""
    coords = [np.array(s, dtype=np.float64)[:2] for s in sources]
    n = len(coords)
    if n < 2:
        return 0.0
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def compute_mean_source_to_path_distance(
    sources: Sequence[tuple[float, float] | Sequence[float]],
    paths: Sequence[Sequence[tuple[int, int]]],
) -> float:
    """Mean distance from all sources to shortest path, averaged across start cells."""
    if not paths or not sources:
        return 0.0
    src_arr = np.array([s[:2] for s in sources], dtype=np.float64)  # (S, 2)
    per_path_means: list[float] = []
    for path in paths:
        if not path:
            continue
        p_arr = np.array(path, dtype=np.float64)  # (P, 2)
        # Distance from each source to closest point on path:
        # dists shape: (S, P) -> min along axis 1 -> (S,)
        dists = np.min(np.linalg.norm(src_arr[:, None, :] - p_arr[None, :, :], axis=2), axis=1)
        per_path_means.append(float(np.mean(dists)))
    return float(np.mean(per_path_means)) if per_path_means else 0.0


def compute_corridor_intensity_share(
    radiation: np.ndarray,
    walls: np.ndarray,
    paths: Sequence[Sequence[tuple[int, int]]],
    radius: float = 5.0,
) -> float:
    """Share of free-space radiation within radius of shortest path, averaged over starts."""
    free_mask = ~walls
    total_free_radiation = float(np.sum(radiation[free_mask]))
    if total_free_radiation <= 0.0 or not paths:
        return 0.0

    free_rows, free_cols = np.where(free_mask)
    free_coords = np.stack([free_rows, free_cols], axis=1).astype(np.float64)  # (K, 2)
    free_rad_vals = radiation[free_mask]

    start_shares: list[float] = []
    for path in paths:
        if not path:
            continue
        p_arr = np.array(path, dtype=np.float64)  # (P, 2)
        # Distance from each free cell to nearest point on path:
        # (K, P) -> min across path points -> (K,)
        dists = np.min(
            np.linalg.norm(free_coords[:, None, :] - p_arr[None, :, :], axis=2),
            axis=1,
        )
        in_corridor = dists <= radius
        corridor_rad = float(np.sum(free_rad_vals[in_corridor]))
        start_shares.append(corridor_rad / total_free_radiation)

    return float(np.mean(start_shares)) if start_shares else 0.0


def compute_mean_shortest_path_length(paths: Sequence[Sequence[tuple[int, int]]]) -> float:
    """Mean shortest-path geometric length across start cells."""
    if not paths:
        return 0.0
    lengths = [path_length(list(p)) for p in paths if p]
    return float(np.mean(lengths)) if lengths else 0.0


def compute_all_features_for_layout(
    env: RadiationEnv,
    starts: Sequence[tuple[int, int]],
    oracle_saving_percent: float,
) -> dict[str, float]:
    """Compute the six pre-declared features for a given environment and start set."""
    source_positions = [
        tuple(s[0]) if isinstance(s, (tuple, list)) else tuple(getattr(s, "position", s))
        for s in env.sources
    ]
    paths: list[list[tuple[int, int]]] = []
    for s in starts:
        p, _, _, _ = astar_search(tuple(s), env.goal, mode="shortest", env=env)
        if p is not None:
            paths.append(p)

    return {
        "oracle_saving_percent": float(oracle_saving_percent),
        "wall_fraction": compute_wall_fraction(env.walls),
        "min_source_distance": compute_min_source_distance(source_positions),
        "mean_source_to_shortest_dist": compute_mean_source_to_path_distance(source_positions, paths),
        "corridor_intensity_share": compute_corridor_intensity_share(env.radiation, env.walls, paths, radius=5.0),
        "mean_shortest_path_length": compute_mean_shortest_path_length(paths),
    }


def main() -> None:
    s13_summary = json.loads(S13_SUMMARY.read_text(encoding="utf-8"))
    s13_layouts = json.loads(S13_LAYOUTS.read_text(encoding="utf-8"))["layouts"]

    records = []
    for row in s13_summary["per_layout"]:
        seed = row["layout_seed"]
        k_reg = row["online_kernel_regret_percent"]
        o_reg = row["online_omp3_regret_percent"]
        diff = k_reg - o_reg
        group = "omp3 loses" if diff < 0.0 else "omp3 wins"

        layout_meta = s13_layouts[str(seed)]
        starts = [tuple(s) for s in layout_meta["starts"]]
        env = generate_random_layout(seed).to_env()

        features = compute_all_features_for_layout(
            env=env,
            starts=starts,
            oracle_saving_percent=row["oracle_saving_percent"],
        )

        records.append({
            "layout_seed": seed,
            "paired_online_difference": diff,
            "group": group,
            "features": features,
        })

    summary_data = {
        "metadata": {
            "layout_count": len(records),
            "study": "S14",
            "description": "Pre-declared layout features for S14 exploratory failure analysis",
            "features_declared": [
                "oracle_saving_percent",
                "wall_fraction",
                "min_source_distance",
                "mean_source_to_shortest_dist",
                "corridor_intensity_share",
                "mean_shortest_path_length",
            ],
        },
        "records": records,
    }
    OUTPUT_JSON.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# S14 Layout Features Table",
        "",
        "| Layout | Group | Paired diff (%) | Oracle saving (%) | Wall fraction | Min source dist | Mean source-path dist | Corridor share | Shortest length |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        f = r["features"]
        md_lines.append(
            f"| {r['layout_seed']} | {r['group']} | {r['paired_online_difference']:.2f} | "
            f"{f['oracle_saving_percent']:.1f} | {f['wall_fraction']:.4f} | {f['min_source_distance']:.2f} | "
            f"{f['mean_source_to_shortest_dist']:.2f} | {f['corridor_intensity_share']:.4f} | "
            f"{f['mean_shortest_path_length']:.2f} |"
        )
    OUTPUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON.name} and {OUTPUT_MD.name} for {len(records)} layouts.")


if __name__ == "__main__":
    main()
