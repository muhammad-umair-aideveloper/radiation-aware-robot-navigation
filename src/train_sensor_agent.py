"""Training script for sensor-based DQN agent with shaped/plain reward variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment import RadiationEnv
from src.sensor_env import SensorNavigationEnv
from src.dqn_agent import DQNAgent, compute_epsilon
from src.sensor_pairs import load_sensor_pairs
from main import compute_free_space_distances

DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "sensor_runs"


def precompute_training_goals(
    env: RadiationEnv,
    train_goals: List[Tuple[int, int]],
    min_dist: float = 20.0,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """Precompute geodesic distance maps and valid starting cells (d >= min_dist) for all training goals."""
    goal_cache: Dict[Tuple[int, int], Tuple[np.ndarray, List[Tuple[int, int]]]] = {}
    for g in train_goals:
        d_map = compute_free_space_distances(env, g)
        valid_starts: List[Tuple[int, int]] = []
        for r in range(env.shape[0]):
            for c in range(env.shape[1]):
                if not env.walls[r, c] and d_map[r, c] >= min_dist and d_map[r, c] < 1e9:
                    valid_starts.append((r, c))
        if len(valid_starts) == 0:
            raise RuntimeError(f"Training goal {g} has no valid start locations with distance >= {min_dist}")
        goal_cache[g] = (d_map, valid_starts)
    return goal_cache


def train_dqn_sensor_agent(
    variant: str = "shaped",
    lambda_rad: float = 2.5,
    seed: int = 0,
    total_steps: int = 50000,
    out_dir: Path = DEFAULT_RUNS_DIR,
) -> Dict[str, Any]:
    """Train a sensor-based DQN agent and record performance logs and weights.
    
    Resumable: skips training if weights, log, and summary files already exist.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"dqn_{variant}_lam{lambda_rad}_seed{seed}"

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
    print(f"Variant: {variant} | Lambda: {lambda_rad} | Seed: {seed} | Steps: {total_steps}")
    print(f"=======================================================")

    # Load pre-generated training goals
    pairs_data = load_sensor_pairs()
    train_goals = [tuple(g) for g in pairs_data["train_goals"]]

    base_env = RadiationEnv()
    goal_cache = precompute_training_goals(base_env, train_goals, min_dist=20.0)

    use_shaping = (variant.lower() == "shaped")
    env = SensorNavigationEnv(
        base_env=base_env,
        lambda_rad=lambda_rad,
        shaping=use_shaping,
        gamma=0.99,
        max_steps=300,
        scale_reward=True,
    )

    # Prime distance cache with precomputed maps
    for g, (d_map, _) in goal_cache.items():
        env._distance_cache[g] = d_map

    agent = DQNAgent(
        obs_dim=28,
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

        # Sample random training goal and valid start
        chosen_goal = train_goals[rng.randint(0, len(train_goals))]
        valid_starts = goal_cache[chosen_goal][1]
        chosen_start = valid_starts[rng.randint(0, len(valid_starts))]

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
    parser = argparse.ArgumentParser(description="Train DQN sensor navigation agent.")
    parser.add_argument("--variant", choices=["shaped", "plain"], default="shaped", help="Reward variant")
    parser.add_argument("--lam", type=float, default=2.5, help="Radiation weight lambda")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--steps", type=int, default=50000, help="Total environment steps budget")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_RUNS_DIR), help="Output directory")

    args = parser.parse_args()

    train_dqn_sensor_agent(
        variant=args.variant,
        lambda_rad=args.lam,
        seed=args.seed,
        total_steps=args.steps,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()

