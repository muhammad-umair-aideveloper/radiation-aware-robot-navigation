"""Training script for sensor-based DQN agent with shaped/plain reward variants.

Supports S6 extensions:
  --stack K       : Observation frame stacking (default: 1, e.g. 4 -> 112 features)
  --curriculum    : Distance curriculum (linear 5 to 20 over first 40% of steps)
  --goal-pool N   : Number of training goals (default: 30, e.g. 150)
  --tag TAG       : Custom run tag for output files
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv
from src.sensor_env import SensorNavigationEnv, FrameStack
from src.dqn_agent import DQNAgent, compute_epsilon
from src.sensor_pairs import load_sensor_pairs, POOL_150_FILE
from main import compute_free_space_distances
from config import SENSOR_DQN_STEPS

DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "sensor_runs"


def compute_curriculum_min_dist(step: int, total_steps: int) -> float:
    """Compute minimum start-goal distance under linear curriculum schedule.

    Rises linearly from 5.0 to 20.0 cells over the first 40% of total_steps,
    then remains constant at 20.0 cells for the remaining 60%.
    """
    frac = 0.4
    curriculum_steps = frac * total_steps
    if curriculum_steps <= 0 or step >= curriculum_steps:
        return 20.0
    alpha = max(0.0, float(step)) / curriculum_steps
    return 5.0 + alpha * 15.0


def precompute_goal_distances(
    env: RadiationEnv,
    train_goals: List[Tuple[int, int]],
    candidate_starts: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], Tuple[np.ndarray, List[float], List[Tuple[int, int]]]]:
    """Precompute geodesic distance maps and sorted candidate starts for all training goals."""
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, List[float], List[Tuple[int, int]]]] = {}
    for g in train_goals:
        d_map = compute_free_space_distances(env, g)
        pairs_sorted = sorted([
            (float(d_map[c]), c)
            for c in candidate_starts
            if d_map[c] < 1e9 and c != g
        ])
        dists = [d for d, _ in pairs_sorted]
        starts = [c for _, c in pairs_sorted]
        cache[g] = (d_map, dists, starts)
    return cache


def train_dqn_sensor_agent(
    variant: str = "shaped",
    lambda_rad: float = 2.5,
    seed: int = 0,
    total_steps: int = SENSOR_DQN_STEPS,
    stack: int = 1,
    curriculum: bool = False,
    goal_pool: int = 30,
    run_tag: Optional[str] = None,
    out_dir: Path = DEFAULT_RUNS_DIR,
) -> Dict[str, Any]:
    """Train a sensor-based DQN agent and record performance logs and weights.

    Resumable: skips training if weights, log, and summary files already exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine run tag
    if run_tag is None:
        if stack == 1 and not curriculum and goal_pool == 30:
            run_tag = f"dqn_{variant}_lam{lambda_rad}_seed{seed}"
        else:
            tag_parts = [f"dqn_{variant}"]
            if stack > 1:
                tag_parts.append(f"stack{stack}")
            if curriculum:
                tag_parts.append("curr")
            if goal_pool != 30:
                tag_parts.append(f"pool{goal_pool}")
            tag_parts.append(f"lam{lambda_rad}_seed{seed}")
            run_tag = "_".join(tag_parts)

    weights_file = out_dir / f"{run_tag}_weights.npz"
    log_file = out_dir / f"{run_tag}_log.csv"
    summary_file = out_dir / f"{run_tag}_summary.json"

    # Resumability check
    if weights_file.exists() and log_file.exists() and summary_file.exists():
        print(f"Skipping completed run: {run_tag}")
        with open(summary_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"\n=======================================================")
    print(f"Starting Training: {run_tag}")
    print(
        f"Variant: {variant} | Lambda: {lambda_rad} | Seed: {seed} | Steps: {total_steps} | "
        f"Stack: {stack} | Curriculum: {curriculum} | GoalPool: {goal_pool}"
    )
    print(f"=======================================================")

    # Load training goals
    base_env = RadiationEnv()
    if goal_pool > 30:
        if POOL_150_FILE.exists():
            with open(POOL_150_FILE, "r", encoding="utf-8") as f:
                p150 = json.load(f)
            train_goals = [tuple(g) for g in p150["train_goals_150"][:goal_pool]]
        else:
            from src.sensor_pairs import SENSOR_PAIRS_S6_FILE
            with open(SENSOR_PAIRS_S6_FILE, "r", encoding="utf-8") as f:
                s6 = json.load(f)
            train_goals = [tuple(g) for g in s6["train_goals_150"][:goal_pool]]
    else:
        pairs_data = load_sensor_pairs("heldout")
        train_goals = [tuple(g) for g in pairs_data["train_goals"][:goal_pool]]

    from src.sensor_pairs import get_low_radiation_cells
    candidate_starts = get_low_radiation_cells(base_env, max_intensity=0.05, exclude_cell=base_env.goal)

    goal_cache = precompute_goal_distances(base_env, train_goals, candidate_starts)

    use_shaping = (variant.lower() == "shaped")
    base_sensor_env = SensorNavigationEnv(
        base_env=base_env,
        lambda_rad=lambda_rad,
        shaping=use_shaping,
        gamma=0.99,
        max_steps=300,
        scale_reward=True,
        num_stack=stack,
    )
    env = base_sensor_env

    # Prime distance cache
    for g, (d_map, _, _) in goal_cache.items():
        base_sensor_env._distance_cache[g] = d_map

    agent = DQNAgent(
        obs_dim=env.obs_dim,
        num_actions=8,
        hidden_sizes=(64, 64),
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        update_frequency=4,
        target_update_frequency=1000,
        huber_delta=1.0,
        seed=seed,
    )

    rng = np.random.RandomState(seed * 1000 + 7)

    # Initialize CSV logger
    csv_file = open(log_file, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "steps", "total_steps", "reward", "outcome", "dose", "epsilon", "loss"])

    step_count = 0
    episode_num = 0
    recent_losses: List[float] = []

    start_time = time.time()

    while step_count < total_steps:
        episode_num += 1

        # Sample training goal
        chosen_goal = train_goals[rng.randint(0, len(train_goals))]
        _, dists, starts = goal_cache[chosen_goal]

        # Determine start cell given curriculum or fixed threshold
        if curriculum:
            min_dist = compute_curriculum_min_dist(step_count, total_steps)
        else:
            min_dist = 20.0

        idx = bisect.bisect_left(dists, min_dist)
        if idx >= len(starts):
            idx = max(0, len(starts) - 1)
        chosen_start = starts[rng.randint(idx, len(starts))]

        obs = env.reset(start=chosen_start, goal=chosen_goal)
        ep_reward = 0.0
        ep_dose = 0.0
        ep_losses: List[float] = []

        while not env.done and step_count < total_steps:
            eps = compute_epsilon(step_count, total_steps, start_eps=1.0, end_eps=0.05, decay_fraction=0.6)
            action = agent.select_action(obs, epsilon=eps)

            next_obs, rew, done, info = env.step(action)
            step_count += 1
            ep_reward += rew
            ep_dose += info["step_dose"]

            loss = agent.step_and_update(obs, action, rew, next_obs, done)
            if loss is not None:
                ep_losses.append(loss)
                recent_losses.append(loss)

            obs = next_obs

        mean_ep_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        csv_writer.writerow([
            episode_num,
            env.steps,
            step_count,
            f"{ep_reward:.4f}",
            env.outcome,
            f"{ep_dose:.4f}",
            f"{eps:.4f}",
            f"{mean_ep_loss:.6f}",
        ])
        csv_file.flush()

        if episode_num % 50 == 0 or step_count >= total_steps:
            elapsed = time.time() - start_time
            sps = step_count / max(1e-5, elapsed)
            mean_loss_str = f"{np.mean(recent_losses[-100:]):.5f}" if recent_losses else "N/A"
            print(
                f"  [Ep {episode_num:4d} | Step {step_count:6d}/{total_steps}] "
                f"Outcome: {env.outcome:10s} | EpRew: {ep_reward:7.2f} | EpDose: {ep_dose:6.2f} | "
                f"Eps: {eps:.3f} | Loss: {mean_loss_str} | Speed: {sps:.1f} step/s"
            )

    wall_clock_s = time.time() - start_time
    steps_per_sec = step_count / max(1e-5, wall_clock_s)
    csv_file.close()

    # Save trained network weights
    agent.save(weights_file)
    print(f"Saved weights to: {weights_file}")

    summary = {
        "run_tag": run_tag,
        "variant": variant,
        "lambda_rad": lambda_rad,
        "seed": seed,
        "stack": stack,
        "curriculum": curriculum,
        "goal_pool": goal_pool,
        "total_steps": step_count,
        "total_episodes": episode_num,
        "wall_clock_time_s": float(wall_clock_s),
        "steps_per_sec": float(steps_per_sec),
        "weights_file": str(weights_file.name),
        "log_file": str(log_file.name),
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Finished {run_tag}: {step_count} steps in {wall_clock_s:.2f}s ({steps_per_sec:.1f} steps/s)")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN sensor navigation agent with S6 options.")
    parser.add_argument("--variant", choices=["shaped", "plain"], default="shaped", help="Reward variant")
    parser.add_argument("--lam", type=float, default=2.5, help="Radiation weight lambda")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--steps", type=int, default=SENSOR_DQN_STEPS, help="Total environment steps budget")
    parser.add_argument("--stack", type=int, default=1, help="Number of frames to stack (default: 1)")
    parser.add_argument("--curriculum", action="store_true", help="Enable distance curriculum schedule")
    parser.add_argument("--goal-pool", type=int, default=30, help="Goal pool size (default: 30)")
    parser.add_argument("--tag", type=str, default=None, help="Custom run tag")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_RUNS_DIR), help="Output directory")

    args = parser.parse_args()

    train_dqn_sensor_agent(
        variant=args.variant,
        lambda_rad=args.lam,
        seed=args.seed,
        total_steps=args.steps,
        stack=args.stack,
        curriculum=args.curriculum,
        goal_pool=args.goal_pool,
        run_tag=args.tag,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
