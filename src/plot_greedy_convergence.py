"""Generate greedy success convergence plot across training episodes.

Evaluates greedy policy success rate on the 50 evaluation starts as training progresses
(Q-learning, lambda 2.5, seed 0; and SARSA if available).
Saves:
  results/diagnostic_q_lam2.5_seed0.csv
  results/convergence_greedy_success.png
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv, START_CELL, GOAL_CELL
from src.planners import QLearningAgent
from main import generate_evaluation_starts

RESULTS_DIR = REPO_ROOT / "results"
DIAGNOSTIC_CSV = RESULTS_DIR / "diagnostic_q_lam2.5_seed0.csv"
SARSA_CSV = RESULTS_DIR / "diagnostic_sarsa_lam2.5_seed0.csv"
OUT_PLOT_PATH = RESULTS_DIR / "convergence_greedy_success.png"


def run_diagnostic_qlearning(
    env: RadiationEnv,
    eval_starts: List[Tuple[int, int]],
    lambda_rad: float = 2.5,
    seed: int = 0,
    episodes: int = 50000,
    eval_interval: int = 2500,
) -> Path:
    """Train Q-learning while periodically recording greedy evaluation success."""
    print(f"Generating diagnostic training data for Q-learning (λ={lambda_rad}, seed={seed})...")
    rng = np.random.RandomState(seed)
    agent = QLearningAgent(
        env=env,
        episodes=episodes,
        alpha=0.15,
        gamma=0.98,
        lambda_rad=lambda_rad,
    )

    Q = np.zeros((*agent.env.shape, len(agent.next_r[0, 0])), dtype=np.float32)
    eps_decay = (agent.eps_min / agent.eps_start) ** (1.0 / max(1, agent.episodes - 1))
    epsilon = agent.eps_start

    records: List[Dict[str, Any]] = []

    def evaluate_greedy(current_q: np.ndarray, ep_num: int) -> Dict[str, Any]:
        succ_count = 0
        doses = []
        steps_list = []
        for s in eval_starts:
            path, outcome, dose, st = agent.rollout(current_q, start=s)
            if outcome == "goal":
                succ_count += 1
                doses.append(dose)
                steps_list.append(st)
        return {
            "episode": ep_num,
            "success_count": succ_count,
            "total_starts": len(eval_starts),
            "success_rate": succ_count / len(eval_starts),
            "mean_dose": float(np.mean(doses)) if doses else 0.0,
            "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        }

    # Initial evaluation before training (episode 0)
    records.append(evaluate_greedy(Q, 0))

    t0 = time.time()
    num_free = len(agent.free_cells)

    for ep in range(1, episodes + 1):
        if rng.rand() < 0.5:
            r, c = START_CELL
        else:
            idx = rng.randint(0, num_free)
            r, c = agent.free_cells[idx]

        for _ in range(agent.max_steps):
            if rng.rand() < epsilon:
                a = rng.randint(0, 8)
            else:
                q_vals = Q[r, c]
                max_q = np.max(q_vals)
                best_actions = np.where(q_vals == max_q)[0]
                a = int(best_actions[rng.randint(len(best_actions))])

            nr = int(agent.next_r[r, c, a])
            nc = int(agent.next_c[r, c, a])
            rew = float(agent.rewards[r, c, a])
            term = bool(agent.terminals[r, c, a])

            target = rew if term else rew + agent.gamma * float(np.max(Q[nr, nc]))
            Q[r, c, a] += agent.alpha * (target - Q[r, c, a])

            if term:
                break
            r, c = nr, nc

        epsilon *= eps_decay

        if ep % eval_interval == 0 or ep == episodes:
            rec = evaluate_greedy(Q, ep)
            records.append(rec)
            print(f"  Ep {ep:>5}/{episodes}: greedy success = {rec['success_count']}/{rec['total_starts']} ({rec['success_rate']*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"Diagnostic training complete in {elapsed:.2f} s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIAGNOSTIC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "success_count", "total_starts", "success_rate", "mean_dose", "mean_steps"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved diagnostic CSV to: {DIAGNOSTIC_CSV}")
    return DIAGNOSTIC_CSV


def load_diagnostic_csv(csv_path: Path) -> Tuple[List[int], List[float]]:
    """Load episodes and success rates from a diagnostic CSV file."""
    episodes: List[int] = []
    success_rates: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            success_rates.append(float(row["success_rate"]) * 100.0)
    return episodes, success_rates


def plot_greedy_convergence(out_path: Path, dpi: int = 200) -> None:
    """Plot greedy success rate vs training episodes for Q-learning (and SARSA if present)."""
    env = RadiationEnv()
    eval_starts = generate_evaluation_starts(env, num_starts=50, seed=123)

    if not DIAGNOSTIC_CSV.exists():
        run_diagnostic_qlearning(env, eval_starts, lambda_rad=2.5, seed=0)

    q_eps, q_succ = load_diagnostic_csv(DIAGNOSTIC_CSV)

    sarsa_eps, sarsa_succ = None, None
    if SARSA_CSV.exists():
        sarsa_eps, sarsa_succ = load_diagnostic_csv(SARSA_CSV)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#111111")
    ax.set_facecolor("#181818")

    ax.plot(
        q_eps,
        q_succ,
        color="deepskyblue",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Q-Learning (λ=2.5, Seed 0)",
    )

    if sarsa_eps and sarsa_succ:
        ax.plot(
            sarsa_eps,
            sarsa_succ,
            color="#39ff14",
            linewidth=2.2,
            marker="s",
            markersize=5,
            label="SARSA (λ=2.5, Seed 0)",
        )

    ax.axhline(100.0, color="grey", linestyle=":", alpha=0.6, label="100% Target")

    ax.set_title("Greedy Policy Success Rate vs. Training Episodes (λ = 2.5)", color="white", fontsize=12, pad=12)
    ax.set_xlabel("Training Episodes", color="white", fontsize=11)
    ax.set_ylabel("Greedy Success Rate on 50 Evaluation Starts (%)", color="white", fontsize=11)
    ax.set_ylim(-2, 105)
    ax.grid(True, linestyle=":", color="#444444", alpha=0.7)
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", facecolor="#222222", edgecolor="#444444", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved greedy success convergence plot to: {out_path}")


def main() -> None:
    plot_greedy_convergence(OUT_PLOT_PATH)


if __name__ == "__main__":
    main()

