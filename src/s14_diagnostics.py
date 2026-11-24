"""Diagnostics runner for S14 exploratory failure analysis of online OMP3."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RadiationEnv
from src.estimators import parametric_sources
from src.layouts import generate_random_layout
from src.s13_study import run_online_episode

S13_DIR = ROOT / "results" / "s13"
S14_DIR = ROOT / "results" / "s14"
S13_LAYOUTS = ROOT / "results" / "s13_layouts.json"

DEFAULT_LAYOUT_SEEDS = list(range(200, 220))
DEFAULT_NOISE_SEEDS = [0, 1, 2]
CONTROL_TOLERANCE = 1e-9


def compute_matched_source_error(
    true_sources: Sequence[Sequence[float] | tuple[float, float]],
    fitted_sources: Sequence[Sequence[float] | tuple[float, float]],
) -> tuple[float, bool]:
    """Compute optimal 1-to-1 assignment position error and all-within-3 flag.

    Returns:
        (mean_matched_euclidean_distance, all_within_3_cells)
    """
    t_coords = [np.array(s[:2], dtype=np.float64) for s in true_sources]
    f_coords = [np.array(s[:2], dtype=np.float64) for s in fitted_sources]
    n = len(t_coords)
    if n == 0 or len(f_coords) == 0:
        return 0.0, False

    best_mean_dist = float("inf")
    best_all_within_3 = False

    for perm in itertools.permutations(range(len(f_coords))):
        dists = [
            float(np.linalg.norm(t_coords[i] - f_coords[perm[i]]))
            for i in range(min(n, len(perm)))
        ]
        mean_d = float(np.mean(dists))
        if mean_d < best_mean_dist:
            best_mean_dist = mean_d
            best_all_within_3 = all(d <= 3.0 for d in dists)

    return best_mean_dist, best_all_within_3


def compute_relative_l2_error(
    fitted_field: np.ndarray,
    true_field: np.ndarray,
    free_mask: np.ndarray,
) -> float:
    """Relative L2 error of fitted field versus true field on free cells."""
    true_free = true_field[free_mask]
    fit_free = fitted_field[free_mask]
    norm_true = float(np.linalg.norm(true_free))
    if norm_true == 0.0:
        return 0.0
    return float(np.linalg.norm(fit_free - true_free) / norm_true)


def replay_and_fit_diagnostics(
    readings: dict[tuple[int, int], list[float]],
    free_mask: np.ndarray,
    true_field: np.ndarray,
    true_sources: Sequence[tuple[tuple[int, int], float] | Sequence[Any]],
    k: int = 3,
) -> dict[str, Any]:
    """Replay accumulated episode readings and fit OMP3 for diagnostics."""
    cells = np.array(list(readings.keys()), dtype=int)
    values = np.array([np.mean(v) for v in readings.values()], dtype=float)

    if len(cells) < k:
        # Fallback if fewer cells than sources
        fitted_field = np.zeros(free_mask.shape, dtype=float)
        fitted_field[free_mask] = float(np.mean(values)) if len(values) > 0 else 0.0
        fitted_sources: list[tuple[int, int, float]] = []
        matched_err = float("inf")
        all_within_3 = False
    else:
        fitted_field, fitted_sources = parametric_sources(cells, values, free_mask, k=k)
        true_pos = [
            tuple(s[0]) if isinstance(s, (tuple, list)) and len(s) == 2 and isinstance(s[0], (tuple, list))
            else tuple(s[:2])
            for s in true_sources
        ]
        fit_pos = [(s[0], s[1]) for s in fitted_sources]
        matched_err, all_within_3 = compute_matched_source_error(true_pos, fit_pos)

    rel_l2 = compute_relative_l2_error(fitted_field, true_field, free_mask)

    return {
        "matched_source_position_error": matched_err,
        "all_sources_within_3": bool(all_within_3),
        "relative_l2_field_error": rel_l2,
        "distinct_measured_cells": len(readings),
        "fitted_sources": [[int(s[0]), int(s[1]), float(s[2])] for s in fitted_sources],
    }


def aggregate_layout_diagnostics(episodes: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Compute mean layout-level metrics from per-episode records."""
    if not episodes:
        return {}
    return {
        "mean_matched_source_position_error": float(
            np.mean([e["matched_source_position_error"] for e in episodes])
        ),
        "all_sources_within_3_rate": float(
            np.mean([1.0 if e["all_sources_within_3"] else 0.0 for e in episodes])
        ),
        "mean_relative_l2_field_error": float(
            np.mean([e["relative_l2_field_error"] for e in episodes])
        ),
        "mean_distinct_measured_cells": float(
            np.mean([e["distinct_measured_cells"] for e in episodes])
        ),
        "mean_true_dose": float(np.mean([e["dose"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episodes])),
    }


def run_single_episode(
    env: RadiationEnv,
    start: tuple[int, int],
    seed: int,
) -> dict[str, Any]:
    """Run one online OMP3 episode and compute its diagnostic metrics."""
    ep = run_online_episode(
        true_field=env.radiation,
        free_mask=~env.walls,
        start=start,
        goal=env.goal,
        estimator="omp3",
        pattern="3x3",
        dwell=1.0,
        seed=seed,
        replan_every=3,
        lam=2.5,
        max_steps=300,
    )
    diag = replay_and_fit_diagnostics(
        readings=ep["readings"],
        free_mask=~env.walls,
        true_field=env.radiation,
        true_sources=env.sources,
        k=3,
    )
    return {
        "start": list(start),
        "seed": seed,
        "dose": ep["dose"],
        "steps": ep["steps"],
        "success": ep["success"],
        "path": [list(c) for c in ep["path"]],
        **diag,
    }


def run_fidelity_control(
    test_layouts: Sequence[int] = (200, 201),
    tolerance: float = CONTROL_TOLERANCE,
) -> bool:
    """Re-simulate online OMP3 episodes on control layouts and verify S13 dose match."""
    print(f"Running fidelity control on layouts {list(test_layouts)}...")
    for seed in test_layouts:
        stored_path = S13_DIR / f"layout_{seed}.json"
        if not stored_path.exists():
            raise FileNotFoundError(f"Missing S13 baseline file: {stored_path}")
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        stored_mean = stored["online"]["omp3"]["mean_true_dose"]

        env = generate_random_layout(seed).to_env()
        starts = [tuple(s) for s in stored["metadata"]["starts"]]
        noise_seeds = stored["metadata"]["noise_seeds"]

        recomputed_doses: list[float] = []
        for s_idx, noise_seed in enumerate(noise_seeds):
            seed_doses: list[float] = []
            for idx, start in enumerate(starts):
                ep = run_online_episode(
                    env.radiation,
                    ~env.walls,
                    start,
                    env.goal,
                    "omp3",
                    "3x3",
                    1.0,
                    noise_seed * 1000 + idx,
                    replan_every=3,
                    lam=2.5,
                    max_steps=300,
                )
                if not ep["success"]:
                    raise RuntimeError(f"Fidelity control failed: episode unsuccessful on layout {seed}")
                seed_doses.append(ep["dose"])

            stored_seed_dose = stored["online"]["omp3"]["per_seed"][s_idx]["mean_true_dose"]
            seed_diff = abs(float(np.mean(seed_doses)) - stored_seed_dose)
            if seed_diff > tolerance:
                raise RuntimeError(
                    f"Fidelity control failed: layout {seed} seed {noise_seed} diff {seed_diff} > {tolerance}"
                )
            recomputed_doses.extend(seed_doses)

        overall_diff = abs(float(np.mean(recomputed_doses)) - stored_mean)
        if overall_diff > tolerance:
            raise RuntimeError(
                f"Fidelity control failed: layout {seed} overall dose diff {overall_diff} > {tolerance}"
            )
        print(f"  Layout {seed}: PASS (dose diff = {overall_diff:.2e})")

    print("FIDELITY CONTROL PASSED: Re-simulated doses match S13 baseline within 1e-9.")
    return True


def run_diagnostics_for_layout(
    seed: int,
    starts: Sequence[tuple[int, int]],
    noise_seeds: Sequence[int],
    out_dir: Path = S14_DIR,
    force: bool = False,
) -> Path:
    """Run full diagnostics for one layout and save JSON record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"layout_{seed}.json"
    if out_path.exists() and not force:
        print(f"Layout {seed} already exists at {out_path.name}; skipping.")
        return out_path

    env = generate_random_layout(seed).to_env()
    episodes: list[dict[str, Any]] = []

    for noise_seed in noise_seeds:
        for idx, start in enumerate(starts):
            ep_seed = noise_seed * 1000 + idx
            ep_rec = run_single_episode(env, tuple(start), ep_seed)
            ep_rec["noise_seed"] = noise_seed
            episodes.append(ep_rec)

    layout_summary = aggregate_layout_diagnostics(episodes)
    result = {
        "layout_seed": seed,
        "starts_count": len(starts),
        "noise_seeds": list(noise_seeds),
        "episodes_count": len(episodes),
        "summary": layout_summary,
        "episodes": episodes,
    }
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run 1 layout 1 start quick benchmark")
    parser.add_argument("--control", action="store_true", help="Run fidelity control only")
    parser.add_argument("--force", action="store_true", help="Overwrite existing result files")
    parser.add_argument("--layouts", nargs="+", type=int, default=DEFAULT_LAYOUT_SEEDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_NOISE_SEEDS)
    args = parser.parse_args()

    if args.control:
        run_fidelity_control()
        return 0

    if args.quick:
        print("S14 Diagnostics Quick Mode:")
        layout_meta = json.loads(S13_LAYOUTS.read_text(encoding="utf-8"))["layouts"]
        seed = 200
        starts = [tuple(s) for s in layout_meta[str(seed)]["starts"]]
        start0 = starts[0]

        env = generate_random_layout(seed).to_env()
        t0 = time.time()
        rec = run_single_episode(env, start0, 0)
        elapsed = time.time() - t0

        total_episodes_3seeds = len(args.layouts) * len(starts) * len(args.seeds)
        projected_min_3seeds = (elapsed * total_episodes_3seeds) / 60.0

        print(f"  Quick benchmark single episode elapsed: {elapsed:.3f} s")
        print(f"  Sample matched source error: {rec['matched_source_position_error']:.2f} cells")
        print(f"  Sample relative L2 error: {rec['relative_l2_field_error']:.4f}")
        print(f"  Projected total runtime ({total_episodes_3seeds} episodes, {len(args.layouts)} layouts): {projected_min_3seeds:.1f} minutes")
        print(f"  Exact number of result files expected: {len(args.layouts)} JSON files in results/s14/")
        return 0

    # Full run
    run_fidelity_control()

    layout_meta = json.loads(S13_LAYOUTS.read_text(encoding="utf-8"))["layouts"]
    total = len(args.layouts)
    for i, seed in enumerate(args.layouts, 1):
        starts = [tuple(s) for s in layout_meta[str(seed)]["starts"]]
        print(f"[{i}/{total}] Processing layout {seed}...")
        run_diagnostics_for_layout(
            seed=seed,
            starts=starts,
            noise_seeds=args.seeds,
            out_dir=S14_DIR,
            force=args.force,
        )

    print(f"Finished {total} layouts. Results saved in {S14_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
