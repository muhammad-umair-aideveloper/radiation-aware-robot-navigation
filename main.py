"""Main evaluation script for radiation-aware robot navigation.

Runs A* (shortest & radiation-aware) and Tabular Q-learning (5 seeds).
Evaluates on 50 fixed start cells and the canonical (2,2) -> (47,47) path.
Generates publication-quality dark-themed plots and GIF.
"""

from __future__ import annotations

import collections
import heapq
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import (
    ACTIONS,
    ACTION_LENGTHS,
    GOAL_CELL,
    GRID_SHAPE,
    START_CELL,
    RadiationEnv,
    compute_radiation_field,
    create_wall_mask,
)
from src.planners import QLearningAgent, astar_search, octile_distance
from src.visualize import (
    generate_navigation_gif,
    plot_rl_convergence,
    plot_trajectory_comparison,
)


def compute_free_space_distances(
    env: RadiationEnv, goal: Tuple[int, int]
) -> np.ndarray:
    """Compute exact 8-connected free-space shortest distance from all cells to goal."""
    dist = np.full(env.shape, np.inf, dtype=np.float64)
    dist[goal] = 0.0

    pq: List[Tuple[float, int, int]] = [(0.0, goal[0], goal[1])]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue

        for a_idx, (dr, dc) in enumerate(ACTIONS):
            # Reverse move: from (nr, nc) to (r, c)
            nr, nc = r + dr, c + dc
            if not (0 <= nr < env.shape[0] and 0 <= nc < env.shape[1]):
                continue
            if env.walls[nr, nc]:
                continue
            # Corner cutting check
            if abs(dr) == 1 and abs(dc) == 1:
                if env.walls[r + dr, c] or env.walls[r, c + dc]:
                    continue

            cost = ACTION_LENGTHS[a_idx]
            new_d = d + cost
            if new_d < dist[nr, nc]:
                dist[nr, nc] = new_d
                heapq.heappush(pq, (new_d, nr, nc))

    return dist


def generate_evaluation_starts(
    env: RadiationEnv, num_starts: int = 50, seed: int = 123
) -> List[Tuple[int, int]]:
    """Select 50 fixed start cells with free-space distance to goal >= 20."""
    dist_map = compute_free_space_distances(env, env.goal)

    candidates: List[Tuple[int, int]] = []
    for r in range(env.shape[0]):
        for c in range(env.shape[1]):
            if not env.walls[r, c] and dist_map[r, c] >= 20.0:
                candidates.append((r, c))

    rng = np.random.RandomState(seed)
    chosen_indices = rng.choice(len(candidates), size=num_starts, replace=False)
    return [candidates[i] for i in chosen_indices]


def main() -> None:
    np.random.seed(42)
    env = RadiationEnv()
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Radiation-Aware Path Planning & Reinforcement Learning Study")
    print("=" * 70)

    # 1. Generate 50 fixed evaluation starts
    eval_starts = generate_evaluation_starts(env, num_starts=50, seed=123)
    print(f"Generated {len(eval_starts)} evaluation start cells (free-space dist >= 20).")

    # 2. Evaluate A* Shortest
    print("\n[1/3] Running A* Shortest Planner...")
    t0 = time.time()
    astar_s_res = []
    for s in eval_starts:
        path, dose, steps, _ = astar_search(s, GOAL_CELL, mode="shortest", env=env)
        succ = path is not None
        astar_s_res.append({"start": s, "success": succ, "dose": dose, "steps": steps})
    astar_s_time = time.time() - t0

    # 3. Evaluate A* Radiation-Aware
    print("[2/3] Running A* Radiation-Aware Planner...")
    t0 = time.time()
    astar_a_res = []
    for s in eval_starts:
        path, dose, steps, _ = astar_search(s, GOAL_CELL, mode="aware", env=env)
        succ = path is not None
        astar_a_res.append({"start": s, "success": succ, "dose": dose, "steps": steps})
    astar_a_time = time.time() - t0

    # 4. Train and Evaluate Tabular Q-Learning (5 Seeds)
    print("[3/3] Training & Evaluating Tabular Q-Learning (5 Seeds, 50,000 episodes each)...")
    q_agent = QLearningAgent(env=env, episodes=50000, alpha=0.15, gamma=0.98)

    q_models = []
    q_all_rewards = []
    q_all_doses = []
    q_train_times = []
    q_eval_res = []  # list of lists per seed

    for seed in range(5):
        print(f"  -> Training Seed {seed}...")
        Q, ep_rewards, ep_doses, elapsed = q_agent.train(seed=seed)
        q_models.append(Q)
        q_all_rewards.append(ep_rewards)
        q_all_doses.append(ep_doses)
        q_train_times.append(elapsed)
        print(f"     Seed {seed} finished in {elapsed:.2f} s")

        # Evaluate on the 50 starts
        seed_res = []
        for s in eval_starts:
            path, outcome, dose, steps = q_agent.rollout(Q, start=s)
            succ = outcome == "goal"
            seed_res.append({"start": s, "success": succ, "outcome": outcome, "dose": dose, "steps": steps})
        q_eval_res.append(seed_res)

    # 5. Canonical (2, 2) -> (47, 47) Evaluation
    print("\n--- Canonical Route: (2,2) -> (47,47) ---")
    path_s, dose_s, steps_s, _ = astar_search(START_CELL, GOAL_CELL, mode="shortest", env=env)
    path_a, dose_a, steps_a, _ = astar_search(START_CELL, GOAL_CELL, mode="aware", env=env)

    q_canonical_paths = []
    q_canonical_outcomes = []
    q_canonical_doses = []
    q_canonical_steps = []

    for seed, Q in enumerate(q_models):
        p, outcome, d, st = q_agent.rollout(Q, start=START_CELL)
        q_canonical_paths.append(p)
        q_canonical_outcomes.append(outcome)
        q_canonical_doses.append(d)
        q_canonical_steps.append(st)
        print(f"  Q-Learning Seed {seed}: outcome={outcome}, steps={st}, dose={d:.2f} a.u.")

    print(f"  A* Shortest: outcome=goal, steps={int(steps_s)}, dose={dose_s:.2f} a.u.")
    print(f"  A* Aware:    outcome=goal, steps={int(steps_a)}, dose={dose_a:.2f} a.u.")

    # 6. Aggregate Statistics over 50 starts
    def calc_stats(res_list):
        succs = [r for r in res_list if r["success"]]
        succ_count = len(succs)
        mean_steps = float(np.mean([r["steps"] for r in succs])) if succs else float("nan")
        mean_dose = float(np.mean([r["dose"] for r in succs])) if succs else float("nan")
        return succ_count, mean_steps, mean_dose

    s_succ, s_steps, s_dose = calc_stats(astar_s_res)
    a_succ, a_steps, a_dose = calc_stats(astar_a_res)

    q_succs = []
    q_steps_list = []
    q_doses_list = []
    for seed in range(5):
        sc, st, ds = calc_stats(q_eval_res[seed])
        q_succs.append(sc)
        if not math.isnan(st):
            q_steps_list.append(st)
        if not math.isnan(ds):
            q_doses_list.append(ds)

    mean_q_succ = float(np.mean(q_succs))
    std_q_succ = float(np.std(q_succs))
    mean_q_steps = float(np.mean(q_steps_list)) if q_steps_list else float("nan")
    std_q_steps = float(np.std(q_steps_list)) if q_steps_list else float("nan")
    mean_q_dose = float(np.mean(q_doses_list)) if q_doses_list else float("nan")
    std_q_dose = float(np.std(q_doses_list)) if q_doses_list else float("nan")
    mean_q_time = float(np.mean(q_train_times))
    std_q_time = float(np.std(q_train_times))

    # 7. Print Markdown Table
    print("\n" + "=" * 70)
    print("### Benchmark Results on 50 Evaluation Starts")
    print("=" * 70)
    header = "| Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Planning/Training Time |"
    sep = "|---|---|---|---|---|"
    row_s = f"| A* Shortest | {s_succ}/50 (100%) | {s_steps:.2f} | {s_dose:.2f} | {astar_s_time:.3f} s total |"
    row_a = f"| A* Aware | {a_succ}/50 (100%) | {a_steps:.2f} | {a_dose:.2f} | {astar_a_time:.3f} s total |"
    row_q = (
        f"| Tabular Q-Learning (5 seeds) | {mean_q_succ:.1f} ± {std_q_succ:.1f}/50 "
        f"({mean_q_succ/50*100:.1f}%) | {mean_q_steps:.2f} ± {std_q_steps:.2f} | "
        f"{mean_q_dose:.2f} ± {std_q_dose:.2f} | {mean_q_time:.2f} ± {std_q_time:.2f} s / seed |"
    )

    print(header)
    print(sep)
    print(row_s)
    print(row_a)
    print(row_q)

    # 8. Paired Comparison
    print("\n### Paired Comparisons (Starts where both methods succeeded)")
    print("-" * 70)
    # A* Aware vs A* Shortest
    paired_s_a = [
        (s["dose"], a["dose"], s["steps"], a["steps"])
        for s, a in zip(astar_s_res, astar_a_res)
        if s["success"] and a["success"]
    ]
    if paired_s_a:
        d_diff = [a - s for s, a, _, _ in paired_s_a]
        step_diff = [a - s for _, _, s, a in paired_s_a]
        print(
            f"A* Aware vs A* Shortest (N={len(paired_s_a)} pairs):\n"
            f"  Mean dose difference: {np.mean(d_diff):+.2f} a.u. "
            f"({np.mean(d_diff)/s_dose*100:+.1f}%)\n"
            f"  Mean step difference: {np.mean(step_diff):+.2f} steps "
            f"({np.mean(step_diff)/s_steps*100:+.1f}%)"
        )

    # Q-Learning (Seed 0) vs A* Shortest
    paired_s_q0 = [
        (s["dose"], q["dose"], s["steps"], q["steps"])
        for s, q in zip(astar_s_res, q_eval_res[0])
        if s["success"] and q["success"]
    ]
    if paired_s_q0:
        d_diff_q = [q - s for s, q, _, _ in paired_s_q0]
        step_diff_q = [q - s for _, _, s, q in paired_s_q0]
        print(
            f"Q-Learning (Seed 0) vs A* Shortest (N={len(paired_s_q0)} pairs):\n"
            f"  Mean dose difference: {np.mean(d_diff_q):+.2f} a.u.\n"
            f"  Mean step difference: {np.mean(step_diff_q):+.2f} steps"
        )

    # 9. Generate Plots & GIF
    print("\nGenerating publication plots and animation...")
    plot_trajectory_comparison(
        path_shortest=path_s,
        path_aware=path_a,
        path_rl=q_canonical_paths[0],
        outcome_rl=q_canonical_outcomes[0],
        env=env,
        save_path=results_dir / "trajectory_comparison.png",
    )
    print("  -> Saved results/trajectory_comparison.png")

    plot_rl_convergence(
        all_rewards=q_all_rewards,
        all_doses=q_all_doses,
        save_path=results_dir / "rl_convergence.png",
    )
    print("  -> Saved results/rl_convergence.png")

    generate_navigation_gif(
        path_shortest=path_s,
        path_aware=path_a,
        path_rl=q_canonical_paths[0],
        env=env,
        save_path=results_dir / "radiation_navigation.gif",
    )
    print("  -> Saved results/radiation_navigation.gif")
    print("\nBenchmark and evaluation complete!")


if __name__ == "__main__":
    main()
