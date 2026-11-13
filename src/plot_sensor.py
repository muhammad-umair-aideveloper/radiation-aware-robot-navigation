"""Plotting script for sensor-based DQN learning curves and evaluation trajectories.

Generates:
  - results/sensor_learning_curves.png: Training reward and success rate curves.
  - results/sensor_trajectories.png: 2x2 comparison of trajectories across first 4 held-out pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv
from src.sensor_env import SensorNavigationEnv
from src.sensors import get_sensor_observation
from src.dqn_agent import DQNAgent
from src.sensor_pairs import load_sensor_pairs
from src.planners import astar_search

RESULTS_DIR = REPO_ROOT / "results"
SENSOR_RUNS_DIR = RESULTS_DIR / "sensor_runs"
FIG_CURVES = RESULTS_DIR / "sensor_learning_curves.png"
FIG_TRAJS = RESULTS_DIR / "sensor_trajectories.png"


def plot_learning_curves(
    window: int = 50,
    max_episodes: int = 1600,
    save_path: Path = FIG_CURVES,
    dpi: int = 200,
) -> None:
    """Plot training reward and goal success rate over episodes."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#111111")
    ax1.set_facecolor("#181818")
    ax2.set_facecolor("#181818")

    groups = [
        ("Shaped, λ = 2.5", [
            SENSOR_RUNS_DIR / f"dqn_shaped_lam2.5_seed{s}_log.csv" for s in range(3)
        ], "#00d2ff", "-"),
        ("Shaped, λ = 0.0", [
            SENSOR_RUNS_DIR / f"dqn_shaped_lam0.0_seed{s}_log.csv" for s in range(3)
        ], "#ff7675", "-"),
        ("Plain, λ = 2.5 (Seed 0)", [
            SENSOR_RUNS_DIR / "dqn_plain_lam2.5_seed0_log.csv"
        ], "#a29bfe", "--"),
    ]

    episodes = np.arange(1, max_episodes + 1)

    for label, file_list, color, style in groups:
        rew_series = []
        succ_series = []

        for f in file_list:
            if not f.exists():
                continue
            df = pd.read_csv(f)
            # Clip to max_episodes
            sub_df = df.iloc[:max_episodes]
            r_roll = sub_df["reward"].rolling(window=window, min_periods=1).mean().to_numpy()
            s_roll = ((sub_df["outcome"] == "goal").astype(float) * 100.0).rolling(window=window, min_periods=1).mean().to_numpy()
            
            # Pad if shorter than max_episodes
            if len(r_roll) < max_episodes:
                r_pad = np.pad(r_roll, (0, max_episodes - len(r_roll)), mode="edge")
                s_pad = np.pad(s_roll, (0, max_episodes - len(s_roll)), mode="edge")
            else:
                r_pad = r_roll[:max_episodes]
                s_pad = s_roll[:max_episodes]
            rew_series.append(r_pad)
            succ_series.append(s_pad)

        if not rew_series:
            continue

        rew_arr = np.array(rew_series)
        succ_arr = np.array(succ_series)

        mean_rew = np.mean(rew_arr, axis=0)
        mean_succ = np.mean(succ_arr, axis=0)

        ax1.plot(episodes, mean_rew, linestyle=style, color=color, linewidth=2.0, label=label)
        ax2.plot(episodes, mean_succ, linestyle=style, color=color, linewidth=2.0, label=label)

        if len(rew_series) > 1:
            std_rew = np.std(rew_arr, axis=0)
            std_succ = np.std(succ_arr, axis=0)
            ax1.fill_between(episodes, mean_rew - std_rew, mean_rew + std_rew, color=color, alpha=0.18)
            ax2.fill_between(episodes, mean_succ - std_succ, mean_succ + std_succ, color=color, alpha=0.18)

    ax1.set_title(f"Episode Reward ({window}-Episode Rolling Mean)", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Training Episode", color="white")
    ax1.set_ylabel("Reward (scaled)", color="white")
    ax1.grid(True, linestyle=":", color="#333333", alpha=0.7)
    ax1.tick_params(colors="white")
    ax1.legend(loc="lower right", facecolor="#222222", edgecolor="#444444", fontsize=9)

    ax2.set_title(f"Goal Success Rate % ({window}-Episode Rolling Mean)", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Training Episode", color="white")
    ax2.set_ylabel("Success Rate (%)", color="white")
    ax2.grid(True, linestyle=":", color="#333333", alpha=0.7)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved learning curves to {save_path}")


def to_xy(traj: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
    """Convert (row, col) to plotting (x, y) coordinates."""
    return [c + 0.5 for _, c in traj], [r + 0.5 for r, _ in traj]


def plot_sensor_trajectories(
    save_path: Path = FIG_TRAJS,
    dpi: int = 200,
) -> None:
    """Plot 2x2 grid of trajectories comparing A* and DQN agents on first 4 eval pairs."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    env_base = RadiationEnv()
    pairs = load_sensor_pairs()["eval_pairs"][:4]

    # Load agent models
    agent_25 = DQNAgent()
    weights_25 = SENSOR_RUNS_DIR / "dqn_shaped_lam2.5_seed0_weights.npz"
    if weights_25.exists():
        agent_25.load(weights_25)

    agent_00 = DQNAgent()
    weights_00 = SENSOR_RUNS_DIR / "dqn_shaped_lam0.0_seed0_weights.npz"
    if weights_00.exists():
        agent_00.load(weights_00)

    env_25 = SensorNavigationEnv(base_env=env_base, lambda_rad=2.5, shaping=False)
    env_00 = SensorNavigationEnv(base_env=env_base, lambda_rad=0.0, shaping=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), facecolor="#111111")
    wall_overlay = np.ma.masked_where(~env_base.walls, np.ones_like(env_base.walls, dtype=float))

    for idx, p in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#181818")

        s = tuple(p["start"])
        g = tuple(p["goal"])
        d_free = p["free_space_distance"]

        # 1. Heatmap & Walls
        im = ax.imshow(
            env_base.radiation,
            cmap="inferno",
            origin="upper",
            extent=(0, env_base.shape[1], env_base.shape[0], 0),
        )
        ax.imshow(
            wall_overlay,
            cmap="gray",
            vmin=0,
            vmax=2,
            origin="upper",
            extent=(0, env_base.shape[1], env_base.shape[0], 0),
            alpha=0.65,
        )

        # 2. A* Shortest
        path_s, d_s, st_s, _ = astar_search(s, g, mode="shortest", env=env_base)
        if path_s:
            xs_s, ys_s = to_xy(path_s)
            ax.plot(xs_s, ys_s, linestyle="--", color="#00ffff", linewidth=2.0, label=f"A* Shortest (succ, {int(st_s)} st, {d_s:.0f} dose)")

        # 3. A* Aware
        path_a, d_a, st_a, _ = astar_search(s, g, mode="aware", env=env_base, lambda_rad=2.5)
        if path_a:
            xs_a, ys_a = to_xy(path_a)
            ax.plot(xs_a, ys_a, linestyle="-", color="#39ff14", linewidth=2.2, label=f"A* Aware (succ, {int(st_a)} st, {d_a:.0f} dose)")

        # 4. Sensor DQN (Shaped λ=2.5, Seed 0)
        obs = env_25.reset(start=s, goal=g)
        traj_25 = [env_25.robot_pos]
        while not env_25.done:
            act = agent_25.select_action(obs, epsilon=0.0)
            obs, r, d, info = env_25.step(act)
            traj_25.append(env_25.robot_pos)
        out_25 = info["outcome"]
        xs_25, ys_25 = to_xy(traj_25)
        ax.plot(xs_25, ys_25, linestyle=":", color="#f39c12", linewidth=2.0, label=f"DQN λ=2.5 ({out_25}, {len(traj_25)-1} st)")

        # 5. Sensor DQN (Shaped λ=0.0, Seed 0)
        obs = env_00.reset(start=s, goal=g)
        traj_00 = [env_00.robot_pos]
        while not env_00.done:
            act = agent_00.select_action(obs, epsilon=0.0)
            obs, r, d, info = env_00.step(act)
            traj_00.append(env_00.robot_pos)
        out_00 = info["outcome"]
        xs_00, ys_00 = to_xy(traj_00)
        ax.plot(xs_00, ys_00, linestyle="-.", color="#fd79a8", linewidth=1.8, label=f"DQN λ=0.0 ({out_00}, {len(traj_00)-1} st)")

        # Start & Goal markers
        ax.scatter([s[1] + 0.5], [s[0] + 0.5], color="#0984e3", s=70, zorder=5, label=f"Start {s}")
        ax.scatter([g[1] + 0.5], [g[0] + 0.5], color="#ff4757", s=100, marker="*", zorder=5, label=f"Goal {g}")

        ax.set_title(f"Pair {idx}: Start {s} -> Goal {g} (d_free={d_free:.1f})", color="white", fontsize=10, pad=6)
        ax.set_xlabel("Col (x)", color="white", fontsize=8)
        ax.set_ylabel("Row (y)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved trajectory comparisons to {save_path}")


def main() -> None:
    plot_learning_curves()
    plot_sensor_trajectories()


if __name__ == "__main__":
    main()
