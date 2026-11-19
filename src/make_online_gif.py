"""Animate one seed of the online kernel and omp3 estimation episodes."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RadiationEnv
from src.geiger_model import counts_to_intensity, simulate_counts
from src.measurement_patterns import pattern_cells
from src.online_sim import estimate_field, run_online_episode
from src.online_study2 import FREE_MASK, GOAL, TRUE_FIELD
from src.planners import astar_search


def frame_steps(n_steps: int, every: int = 3) -> list[int]:
    """Return replay frame indices from 0 through n_steps."""
    frames = list(range(0, n_steps + 1, every))
    if frames[-1] != n_steps:
        frames.append(n_steps)
    return frames


def reconstruct_readings(
    path: List[Tuple[int, int]],
    true_field: np.ndarray,
    free_mask: np.ndarray,
    pattern: str,
    dwell: float,
    seed: int,
    upto: int,
) -> Dict[Tuple[int, int], List[float]]:
    """Replay the noisy readings for a path prefix."""
    readings: Dict[Tuple[int, int], List[float]] = {}
    for j in range(upto + 1):
        cells = pattern_cells(path[j], free_mask, pattern)
        counts = simulate_counts(
            true_field, cells, dwell, seed * 100003 + j
        )
        values = counts_to_intensity(counts, dwell)
        for cell, value in zip(cells, values):
            key = (int(cell[0]), int(cell[1]))
            readings.setdefault(key, []).append(float(value))
    return readings


def dose_upto(
    path: List[Tuple[int, int]], field: np.ndarray, upto: int
) -> float:
    """Return true dose accumulated over the first upto moves."""
    dose = 0.0
    for j in range(1, upto + 1):
        dr = path[j][0] - path[j - 1][0]
        dc = path[j][1] - path[j - 1][1]
        length = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
        dose += float(field[path[j]]) * length
    return dose


def main() -> None:
    true_field = np.asarray(TRUE_FIELD)
    free_mask = np.asarray(FREE_MASK, dtype=bool)
    wall = ~free_mask
    seed = 0
    dwell = 1.0
    pattern = "3x3"

    episodes = {}
    for estimator in ("kernel", "omp3"):
        episode = run_online_episode(
            true_field,
            free_mask,
            (2, 2),
            GOAL,
            estimator,
            pattern,
            dwell,
            seed,
        )
        episodes[estimator] = episode

        path = episode["path"]
        reconstructed = reconstruct_readings(
            path, true_field, free_mask, pattern, dwell, seed,
            len(path) - 1,
        )
        returned = episode["readings"]
        if set(reconstructed) != set(returned):
            raise RuntimeError(
                "reconstructed and returned reading keys differ"
            )
        for key in reconstructed:
            if not np.isclose(
                np.mean(reconstructed[key]),
                np.mean(returned[key]),
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"reconstructed and returned readings differ at {key}"
                )

    kernel_path = episodes["kernel"]["path"]
    omp3_path = episodes["omp3"]["path"]
    oracle_env = RadiationEnv(
        goal=GOAL, walls=wall, radiation=true_field
    )
    oracle_result = astar_search(
        (2, 2), GOAL, mode="aware", env=oracle_env, lambda_rad=2.5
    )
    shortest_result = astar_search(
        (2, 2), GOAL, mode="shortest", env=oracle_env
    )
    oracle_path = oracle_result[0]
    shortest_path = shortest_result[0]

    oracle_dose = dose_upto(oracle_path, true_field, len(oracle_path) - 1)
    shortest_dose = dose_upto(
        shortest_path, true_field, len(shortest_path) - 1
    )
    kernel_dose = episodes["kernel"]["dose"]
    omp3_dose = episodes["omp3"]["dose"]

    frames = frame_steps(
        max(len(kernel_path), len(omp3_path)) - 1, every=3
    )
    norm = LogNorm(vmin=0.05, vmax=float(np.max(true_field)))

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=70)

    def draw_frame(index: int) -> None:
        for axis in axes:
            axis.clear()
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")
            axis.imshow(true_field, cmap="inferno", norm=norm)
            axis.imshow(
                np.ma.masked_where(~wall, wall),
                cmap="Greys",
                alpha=0.65,
                vmin=0,
                vmax=1,
            )

        axes[0].set_title("True field")
        if oracle_path is not None:
            oracle = np.asarray(oracle_path)
            axes[0].plot(
                oracle[:, 1], oracle[:, 0],
                linestyle="--", linewidth=1.5,
                color="cyan", label="Oracle",
            )
        if shortest_path is not None:
            shortest = np.asarray(shortest_path)
            axes[0].plot(
                shortest[:, 1], shortest[:, 0],
                linestyle="--", linewidth=1.5,
                color="grey", label="A* shortest",
            )

        kernel_count = min(index, len(kernel_path) - 1)
        omp3_count = min(index, len(omp3_path) - 1)
        kernel_prefix = np.asarray(kernel_path[: kernel_count + 1])
        omp3_prefix = np.asarray(omp3_path[: omp3_count + 1])
        axes[0].plot(
            kernel_prefix[:, 1], kernel_prefix[:, 0],
            linewidth=2.0, color="orange", label="Kernel trail",
        )
        axes[0].plot(
            omp3_prefix[:, 1], omp3_prefix[:, 0],
            linewidth=2.0, color="lime", label="omp3 trail",
        )
        axes[0].scatter(
            [kernel_prefix[-1, 1]], [kernel_prefix[-1, 0]],
            s=70, marker="o", color="orange", edgecolor="white",
            label="Kernel robot",
        )
        axes[0].scatter(
            [omp3_prefix[-1, 1]], [omp3_prefix[-1, 0]],
            s=70, marker="o", color="lime", edgecolor="white",
            label="omp3 robot",
        )
        axes[0].text(
            0.02, 0.02,
            f"Oracle dose: {oracle_dose:.2f} a.u.\n"
            f"A* shortest dose: {shortest_dose:.2f} a.u.",
            transform=axes[0].transAxes,
            fontsize=9,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.7},
        )
        axes[0].legend(loc="upper right", fontsize=8)

        for axis, estimator, path, count in (
            (axes[1], "kernel", kernel_path, kernel_count),
            (axes[2], "omp3", omp3_path, omp3_count),
        ):
            readings = reconstruct_readings(
                path,
                true_field,
                free_mask,
                pattern,
                dwell,
                seed,
                count,
            )
            estimate = estimate_field(readings, free_mask, estimator)
            axis.imshow(estimate, cmap="inferno", norm=norm)
            axis.imshow(
                np.ma.masked_where(~wall, wall),
                cmap="Greys",
                alpha=0.65,
                vmin=0,
                vmax=1,
            )
            measured = np.asarray(list(readings.keys()), dtype=int)
            axis.scatter(
                measured[:, 1], measured[:, 0],
                s=5, facecolors="none", edgecolors="white", linewidths=0.4,
            )
            prefix = np.asarray(path[: count + 1])
            color = "orange" if estimator == "kernel" else "lime"
            axis.plot(
                prefix[:, 1], prefix[:, 0],
                linewidth=2.0, color=color,
            )
            axis.scatter(
                [prefix[-1, 1]], [prefix[-1, 0]],
                s=70, marker="o", color=color, edgecolor="white",
            )
            so_far = dose_upto(path, true_field, count)
            label = "Kernel" if estimator == "kernel" else "omp3"
            axis.set_title(f"{label} | dose so far: {so_far:.2f} a.u.")

    animation = __import__("matplotlib.animation", fromlist=["FuncAnimation"]).FuncAnimation(
        fig,
        draw_frame,
        frames=frames + [frames[-1]] * 5,
        repeat=False,
    )
    result = ROOT / "results"
    result.mkdir(parents=True, exist_ok=True)
    gif_path = result / "online_estimation.gif"
    animation.save(gif_path, writer=PillowWriter(fps=3), dpi=70)

    if gif_path.stat().st_size > 10 * 1024 * 1024:
        fig.set_dpi(60)
        animation.save(gif_path, writer=PillowWriter(fps=3), dpi=60)

    draw_frame(frames[-1])
    fig.tight_layout()
    tmp = ROOT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp / "gif_last_frame.png", dpi=70)
    plt.close(fig)

    frame_count = len(frames) + 5
    size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"frames: {frame_count}")
    print(f"gif_size_mb: {size_mb:.3f}")
    print(f"oracle_dose: {oracle_dose:.2f} a.u.")
    print(f"shortest_dose: {shortest_dose:.2f} a.u.")
    print(f"kernel_dose: {kernel_dose:.2f} a.u.")
    print(f"omp3_dose: {omp3_dose:.2f} a.u.")


if __name__ == "__main__":
    main()
