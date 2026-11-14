"""Plotting script for Study S6: sensor agent diagnosis, variants, and test evaluation.

Generates:
  - results/s6_distance_bins.png   : Success & timeout rates by distance bin for baseline vs selected variant.
  - results/s6_learning_curves.png : Training curves across the four Stage A improvement variants.
  - results/s6_trajectories.png    : 2x2 comparison of trajectories on the first 4 test pairs.
"""

from __future__ import annotations

import json
import math
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
from src.dqn_agent import DQNAgent
from src.sensor_pairs import load_pair_set
from src.planners import astar_search

RESULTS_DIR = REPO_ROOT / "results"
SENSOR_RUNS_DIR = RESULTS_DIR / "sensor_runs"
TEST_JSON_FILE = RESULTS_DIR / "sensor_s6_final.json"
TEST_PAIRS_FILE = RESULTS_DIR / "sensor_test_pairs.json"

FIG_BINS = RESULTS_DIR / "s6_distance_bins.png"
FIG_CURVES = RESULTS_DIR / "s6_learning_curves.png"
FIG_TRAJS = RESULTS_DIR / "s6_trajectories.png"


def plot_distance_bins(save_path: Path = FIG_BINS, dpi: int = 200) -> None:
    """Plot success rate and outcome breakdown across distance bins on test set."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    with open(TEST_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(TEST_PAIRS_FILE, "r", encoding="utf-8") as f:
        test_pairs = json.load(f)["pairs"]

    def get_bin(d_free: float) -> str:
        if d_free < 30.0:
            return "[20, 30)"
        elif d_free < 45.0:
            return "[30, 45)"
        else:
            return "[>= 45]"

    bins = ["[20, 30)", "[30, 45)", "[>= 45]"]
    models = [
        ("Baseline (Shaped Seed 0)", "dqn_shaped_lam2.5_seed0", "#00d2ff"),
        ("M1 (Frame Stack k=4)", "dqn_s6_m1_lam2.5_seed0", "#ff7675"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#111111")
    ax1.set_facecolor("#181818")
    ax2.set_facecolor("#181818")

    x = np.arange(len(bins))
    width = 0.35

    for idx, (label, tag, color) in enumerate(models):
        recs = data["sensor_agents"][tag]["per_pair"]
        succ_rates = []
        timeo_rates = []

        for b in bins:
            b_pairs = [r for r, tp in zip(recs, test_pairs) if get_bin(tp["free_space_distance"]) == b]
            n_tot = max(1, len(b_pairs))
            succ = sum(r["success"] for r in b_pairs)
            timeo = sum(r["outcome"] == "timeout" for r in b_pairs)
            succ_rates.append((succ / n_tot) * 100.0)
            timeo_rates.append((timeo / n_tot) * 100.0)

        offset = (idx - 0.5) * width
        ax1.bar(x + offset, succ_rates, width, label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, timeo_rates, width, label=label, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax1.set_title("Test Set Success Rate (%) by Distance Bin", color="white", fontsize=12, pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n(N={sum(get_bin(tp['free_space_distance']) == b for tp in test_pairs)})" for b in bins], color="white")
    ax1.set_ylabel("Success Rate (%)", color="white")
    ax1.set_ylim(-0.5, 10.0)
    ax1.grid(True, linestyle=":", color="#333333", alpha=0.7)
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", fontsize=9)

    ax2.set_title("Test Set Timeout Rate (%) by Distance Bin", color="white", fontsize=12, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{b}\n(N={sum(get_bin(tp['free_space_distance']) == b for tp in test_pairs)})" for b in bins], color="white")
    ax2.set_ylabel("Timeout Rate (%)", color="white")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", color="#333333", alpha=0.7)
    ax2.tick_params(colors="white")
    ax2.legend(loc="lower right", facecolor="#222222", edgecolor="#444444", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved distance bin plot to {save_path}")


def plot_s6_learning_curves(
    window: int = 50,
    max_episodes: int = 1600,
    save_path: Path = FIG_CURVES,
    dpi: int = 200,
) -> None:
    """Plot training reward and success rate across the 4 Stage A improvement variants."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="#111111")
    ax1.set_facecolor("#181818")
    ax2.set_facecolor("#181818")

    variants = [
        ("Baseline (Shaped, λ=2.5, Seed 0)", SENSOR_RUNS_DIR / "dqn_shaped_lam2.5_seed0_log.csv", "#636e72", "--"),
        ("M1 (Frame Stack k=4)", SENSOR_RUNS_DIR / "dqn_s6_m1_lam2.5_seed0_log.csv", "#00d2ff", "-"),
        ("M2 (Distance Curriculum)", SENSOR_RUNS_DIR / "dqn_s6_m2_lam2.5_seed0_log.csv", "#ff7675", "-"),
        ("M3 (Stack + Curriculum)", SENSOR_RUNS_DIR / "dqn_s6_m3_lam2.5_seed0_log.csv", "#a29bfe", "-"),
        ("M4 (150-Goal Pool)", SENSOR_RUNS_DIR / "dqn_s6_m4_lam2.5_seed0_log.csv", "#fdcb6e", "-"),
    ]

    episodes = np.arange(1, max_episodes + 1)

    for label, log_path, color, style in variants:
        if not log_path.exists():
            continue
        df = pd.read_csv(log_path)
        sub_df = df.iloc[:max_episodes]
        r_roll = sub_df["reward"].rolling(window=window, min_periods=1).mean().to_numpy()
        s_roll = ((sub_df["outcome"] == "goal").astype(float) * 100.0).rolling(window=window, min_periods=1).mean().to_numpy()

        if len(r_roll) < max_episodes:
            r_roll = np.pad(r_roll, (0, max_episodes - len(r_roll)), mode="edge")
            s_roll = np.pad(s_roll, (0, max_episodes - len(s_roll)), mode="edge")
        else:
            r_roll = r_roll[:max_episodes]
            s_roll = s_roll[:max_episodes]

        ax1.plot(episodes, r_roll, linestyle=style, color=color, linewidth=1.8, label=label)
        ax2.plot(episodes, s_roll, linestyle=style, color=color, linewidth=1.8, label=label)

    ax1.set_title(f"Stage A Training Return ({window}-Episode Rolling Mean)", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Training Episode", color="white")
    ax1.set_ylabel("Reward (scaled)", color="white")
    ax1.grid(True, linestyle=":", color="#333333", alpha=0.7)
    ax1.tick_params(colors="white")
    ax1.legend(loc="lower right", facecolor="#222222", edgecolor="#444444", fontsize=8.5)

    ax2.set_title(f"Stage A Training Success Rate % ({window}-Episode Rolling Mean)", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Training Episode", color="white")
    ax2.set_ylabel("Success Rate (%)", color="white")
    ax2.grid(True, linestyle=":", color="#333333", alpha=0.7)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved learning curves to {save_path}")


def to_xy(traj: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
    """Convert (row, col) grid coordinates to plotting (x, y) coordinates."""
    return [c + 0.5 for _, c in traj], [r + 0.5 for r, _ in traj]


def plot_s6_trajectories(save_path: Path = FIG_TRAJS, dpi: int = 200) -> None:
    """Plot 2x2 grid of trajectories on the first 4 test pairs."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    env_base = RadiationEnv()
    test_pairs = load_pair_set("test")[:4]

    # Load agent models
    agent_m1 = DQNAgent(obs_dim=112, num_actions=8)
    weights_m1 = SENSOR_RUNS_DIR / "dqn_s6_m1_lam2.5_seed0_weights.npz"
    if weights_m1.exists():
        agent_m1.load(weights_m1)

    agent_lam0 = DQNAgent(obs_dim=28, num_actions=8)
    weights_lam0 = SENSOR_RUNS_DIR / "dqn_shaped_lam0.0_seed0_weights.npz"
    if weights_lam0.exists():
        agent_lam0.load(weights_lam0)

    env_m1 = SensorNavigationEnv(base_env=env_base, lambda_rad=2.5, shaping=False, num_stack=4)
    env_lam0 = SensorNavigationEnv(base_env=env_base, lambda_rad=0.0, shaping=False, num_stack=1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13), facecolor="#111111")
    wall_overlay = np.ma.masked_where(~env_base.walls, np.ones_like(env_base.walls, dtype=float))

    for idx, p in enumerate(test_pairs):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#181818")

        s = tuple(p["start"])
        g = tuple(p["goal"])
        d_free = p["free_space_distance"]

        # Heatmap & Walls
        ax.imshow(
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

        # 1. A* Shortest
        p_s, d_s, st_s, _ = astar_search(s, g, mode="shortest", env=env_base)
        if p_s:
            xs_s, ys_s = to_xy(p_s)
            ax.plot(xs_s, ys_s, linestyle="--", color="#00ffff", linewidth=2.0, label=f"A* Shortest (goal, {int(st_s)} st, {d_s:.0f} dose)")

        # 2. A* Aware
        p_a, d_a, st_a, _ = astar_search(s, g, mode="aware", env=env_base, lambda_rad=2.5)
        if p_a:
            xs_a, ys_a = to_xy(p_a)
            ax.plot(xs_a, ys_a, linestyle="-", color="#39ff14", linewidth=2.2, label=f"A* Aware (goal, {int(st_a)} st, {d_a:.0f} dose)")

        # 3. Selected Variant: M1 (Stack 4, λ=2.5, Seed 0)
        obs = env_m1.reset(start=s, goal=g)
        traj_m1 = [env_m1.robot_pos]
        while not env_m1.done:
            act = agent_m1.select_action(obs, epsilon=0.0)
            obs, r, d, info = env_m1.step(act)
            traj_m1.append(env_m1.robot_pos)
        out_m1 = info["outcome"]
        xs_m1, ys_m1 = to_xy(traj_m1)
        ax.plot(xs_m1, ys_m1, linestyle=":", color="#f39c12", linewidth=2.0, label=f"M1 Stack-4 ({out_m1}, {len(traj_m1)-1} st)")

        # 4. Selected Variant / Baseline (λ=0.0, Seed 0)
        obs = env_lam0.reset(start=s, goal=g)
        traj_lam0 = [env_lam0.robot_pos]
        while not env_lam0.done:
            act = agent_lam0.select_action(obs, epsilon=0.0)
            obs, r, d, info = env_lam0.step(act)
            traj_lam0.append(env_lam0.robot_pos)
        out_lam0 = info["outcome"]
        xs_lam0, ys_lam0 = to_xy(traj_lam0)
        ax.plot(xs_lam0, ys_lam0, linestyle="-.", color="#fd79a8", linewidth=1.8, label=f"Shaped λ=0.0 ({out_lam0}, {len(traj_lam0)-1} st)")

        # Start & Goal markers
        ax.scatter([s[1] + 0.5], [s[0] + 0.5], color="#0984e3", s=70, zorder=5, label=f"Start {s}")
        ax.scatter([g[1] + 0.5], [g[0] + 0.5], color="#ff4757", s=100, marker="*", zorder=5, label=f"Goal {g}")

        ax.set_title(f"Test Pair {idx}: Start {s} -> Goal {g} (d_free={d_free:.1f})", color="white", fontsize=10, pad=6)
        ax.set_xlabel("Col (x)", color="white", fontsize=8)
        ax.set_ylabel("Row (y)", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(loc="upper left", facecolor="#222222", edgecolor="#444444", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved trajectory comparisons to {save_path}")


def main() -> None:
    plot_distance_bins()
    plot_s6_learning_curves()
    plot_s6_trajectories()


if __name__ == "__main__":
    main()
