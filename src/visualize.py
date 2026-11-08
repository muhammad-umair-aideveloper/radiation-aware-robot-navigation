"""Visualization utilities: trajectories, RL convergence, and navigation GIF."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np

from src.environment import (
    ACTIONS,
    ACTION_LENGTHS,
    GOAL_CELL,
    GRID_SHAPE,
    START_CELL,
    RadiationEnv,
)


def compute_cumulative_dose_series(
    path: List[Tuple[int, int]], env: RadiationEnv
) -> np.ndarray:
    """Compute cumulative dose at each waypoint along a path."""
    doses = [0.0]
    curr_dose = 0.0
    for i in range(len(path) - 1):
        p_curr = path[i]
        p_next = path[i + 1]
        dr = p_next[0] - p_curr[0]
        dc = p_next[1] - p_curr[1]
        step_l = math.sqrt(dr * dr + dc * dc)
        curr_dose += env.radiation[p_next] * step_l
        doses.append(curr_dose)
    return np.array(doses, dtype=np.float64)


def plot_trajectory_comparison(
    path_shortest: List[Tuple[int, int]],
    path_aware: List[Tuple[int, int]],
    path_rl: List[Tuple[int, int]],
    outcome_rl: str,
    env: RadiationEnv,
    save_path: Path,
    dpi: int = 200,
) -> None:
    """Plot radiation heatmap with paths (left) and cumulative dose progress (right)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#111111")
    ax1.set_facecolor("#111111")
    ax2.set_facecolor("#181818")

    # --- Left: Radiation Heatmap & Trajectories ---
    im = ax1.imshow(
        env.radiation,
        cmap="inferno",
        origin="upper",
        extent=(0, env.shape[1], env.shape[0], 0),
    )
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Radiation Intensity (a.u.)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Overlay walls in grey
    wall_overlay = np.ma.masked_where(~env.walls, np.ones_like(env.walls, dtype=float))
    ax1.imshow(
        wall_overlay,
        cmap="gray",
        vmin=0,
        vmax=2,
        origin="upper",
        extent=(0, env.shape[1], env.shape[0], 0),
        alpha=0.6,
    )

    # Convert (row, col) to (x, y) = (col + 0.5, row + 0.5)
    def to_xy(p: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
        return [c + 0.5 for _, c in p], [r + 0.5 for r, _ in p]

    xs_s, ys_s = to_xy(path_shortest)
    xs_a, ys_a = to_xy(path_aware)
    xs_r, ys_r = to_xy(path_rl)

    ax1.plot(xs_s, ys_s, linestyle="--", color="cyan", linewidth=2.0, label="A* Shortest")
    ax1.plot(xs_a, ys_a, linestyle="-", color="#39ff14", linewidth=2.2, label="A* Radiation-Aware")
    rl_label = f"Q-Learning (Seed 0, {outcome_rl})"
    ax1.plot(xs_r, ys_r, linestyle=":", color="yellow", linewidth=2.2, label=rl_label)

    # Mark Start & Goal
    ax1.scatter([START_CELL[1] + 0.5], [START_CELL[0] + 0.5], color="deepskyblue", s=80, zorder=5, label="Start (2,2)")
    ax1.scatter([GOAL_CELL[1] + 0.5], [GOAL_CELL[0] + 0.5], color="red", s=80, marker="*", zorder=5, label="Goal (47,47)")

    ax1.set_title("Navigation Trajectories on Radiation Field", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Column (x)", color="white")
    ax1.set_ylabel("Row (y)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=8)

    # --- Right: Cumulative Dose vs Path Progress (%) ---
    dose_s = compute_cumulative_dose_series(path_shortest, env)
    dose_a = compute_cumulative_dose_series(path_aware, env)
    dose_r = compute_cumulative_dose_series(path_rl, env)

    prog_s = np.linspace(0, 100, len(dose_s))
    prog_a = np.linspace(0, 100, len(dose_a))
    prog_r = np.linspace(0, 100, len(dose_r))

    ax2.plot(prog_s, dose_s, linestyle="--", color="cyan", linewidth=2.0, label=f"A* Shortest (Final: {dose_s[-1]:.1f})")
    ax2.plot(prog_a, dose_a, linestyle="-", color="#39ff14", linewidth=2.2, label=f"A* Aware (Final: {dose_a[-1]:.1f})")
    ax2.plot(prog_r, dose_r, linestyle=":", color="yellow", linewidth=2.2, label=f"Q-Agent Seed 0 (Final: {dose_r[-1]:.1f})")

    ax2.set_title("Cumulative Dose vs Path Progress", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Path Progress (%)", color="white")
    ax2.set_ylabel("Cumulative Absorbed Dose (a.u.)", color="white")
    ax2.grid(True, linestyle=":", color="#444444", alpha=0.7)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def moving_average(data: List[float], window: int = 1000) -> np.ndarray:
    """Compute centered moving average with edge padding."""
    if len(data) < window:
        window = max(1, len(data) // 10)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_rl_convergence(
    all_rewards: List[List[float]],
    all_doses: List[List[float]],
    save_path: Path,
    window: int = 1000,
    dpi: int = 200,
) -> None:
    """Plot moving average of episode reward and dose over training episodes."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor="#111111", sharex=True)
    ax1.set_facecolor("#181818")
    ax2.set_facecolor("#181818")

    # Compute smoothed curves
    smoothed_rewards = [moving_average(r, window) for r in all_rewards]
    smoothed_doses = [moving_average(d, window) for d in all_doses]

    min_len_r = min(len(s) for s in smoothed_rewards)
    min_len_d = min(len(s) for s in smoothed_doses)

    arr_r = np.array([s[:min_len_r] for s in smoothed_rewards])
    arr_d = np.array([s[:min_len_d] for s in smoothed_doses])

    mean_r = np.mean(arr_r, axis=0)
    std_r = np.std(arr_r, axis=0)
    mean_d = np.mean(arr_d, axis=0)
    std_d = np.std(arr_d, axis=0)

    episodes_x = np.arange(window // 2, window // 2 + min_len_r)

    # Subplot 1: Episode Reward
    ax1.plot(episodes_x, smoothed_rewards[0][:min_len_r], color="yellow", alpha=0.8, linewidth=1.5, label="Seed 0")
    ax1.plot(episodes_x, mean_r, color="deepskyblue", linewidth=2.0, label="Mean over 5 seeds")
    ax1.fill_between(episodes_x, mean_r - std_r, mean_r + std_r, color="deepskyblue", alpha=0.2)
    ax1.set_title(f"Tabular Q-Learning Convergence (Moving Average, Window={window})", color="white", fontsize=12)
    ax1.set_ylabel("Episode Reward", color="white")
    ax1.grid(True, linestyle=":", color="#444444", alpha=0.7)
    ax1.tick_params(colors="white")
    ax1.legend(loc="lower right", facecolor="#222222", edgecolor="#444444")

    # Subplot 2: Episode Dose
    ax2.plot(episodes_x, smoothed_doses[0][:min_len_d], color="yellow", alpha=0.8, linewidth=1.5, label="Seed 0")
    ax2.plot(episodes_x, mean_d, color="#39ff14", linewidth=2.0, label="Mean over 5 seeds")
    ax2.fill_between(episodes_x, mean_d - std_d, mean_d + std_d, color="#39ff14", alpha=0.2)
    ax2.set_xlabel("Episode", color="white")
    ax2.set_ylabel("Absorbed Dose (a.u.)", color="white")
    ax2.grid(True, linestyle=":", color="#444444", alpha=0.7)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper right", facecolor="#222222", edgecolor="#444444")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def generate_navigation_gif(
    path_shortest: List[Tuple[int, int]],
    path_aware: List[Tuple[int, int]],
    path_rl: List[Tuple[int, int]],
    env: RadiationEnv,
    save_path: Path,
    num_frames: int = 40,
    fps: int = 8,
) -> None:
    """Generate 40-frame GIF of the 3 real trajectories from (2,2)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#111111")
    ax.set_facecolor("#111111")

    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(save_path), dpi=100):
        for frame in range(num_frames):
            ax.clear()
            ax.set_facecolor("#111111")

            # Background heatmap
            ax.imshow(
                env.radiation,
                cmap="inferno",
                origin="upper",
                extent=(0, env.shape[1], env.shape[0], 0),
            )
            # Walls
            wall_overlay = np.ma.masked_where(~env.walls, np.ones_like(env.walls, dtype=float))
            ax.imshow(
                wall_overlay,
                cmap="gray",
                vmin=0,
                vmax=2,
                origin="upper",
                extent=(0, env.shape[1], env.shape[0], 0),
                alpha=0.6,
            )

            frac = (frame + 1) / float(num_frames)

            def get_subpath(p: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
                count = max(1, min(len(p), int(math.ceil(frac * len(p)))))
                sub = p[:count]
                return [c + 0.5 for _, c in sub], [r + 0.5 for r, _ in sub]

            xs_s, ys_s = get_subpath(path_shortest)
            xs_a, ys_a = get_subpath(path_aware)
            xs_r, ys_r = get_subpath(path_rl)

            ax.plot(xs_s, ys_s, linestyle="--", color="cyan", linewidth=2.0, label="A* Shortest")
            ax.plot(xs_a, ys_a, linestyle="-", color="#39ff14", linewidth=2.2, label="A* Aware")
            ax.plot(xs_r, ys_r, linestyle=":", color="yellow", linewidth=2.2, label="Q-Agent (Seed 0)")

            # Current agent heads
            ax.scatter([xs_s[-1]], [ys_s[-1]], color="cyan", s=40)
            ax.scatter([xs_a[-1]], [ys_a[-1]], color="#39ff14", s=40)
            ax.scatter([xs_r[-1]], [ys_r[-1]], color="yellow", s=40)

            # Start and Goal
            ax.scatter([START_CELL[1] + 0.5], [START_CELL[0] + 0.5], color="deepskyblue", s=70, label="Start (2,2)")
            ax.scatter([GOAL_CELL[1] + 0.5], [GOAL_CELL[0] + 0.5], color="red", s=80, marker="*", label="Goal (47,47)")

            ax.set_title(f"Radiation Navigation Simulation ({int(frac * 100)}%)", color="white", fontsize=11)
            ax.set_xlim(0, env.shape[1])
            ax.set_ylim(env.shape[0], 0)
            ax.axis("off")
            ax.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=8)

            writer.grab_frame()

    plt.close(fig)

