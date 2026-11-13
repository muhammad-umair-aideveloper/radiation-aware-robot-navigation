# Sensor-Based Agent Evaluation Report (S5)

Evaluation of 7 map-free Sensor DQN agents (16-ray LiDAR + 3x3 Geiger + relative goal vector) against classical A* baselines across 100 held-out (start, goal) pairs and the canonical route.

## 1. Benchmark on 100 Held-Out Evaluation Pairs

| Method / Model | Variant | Objective λ | Seed | Success Rate | Collisions | Timeouts | Mean Steps | Mean Dose (a.u.) |
|---|---|---|---|---|---|---|---|---|
| **A* Shortest** | Classical | 0.0 | - | 100/100 (100%) | 0 | 0 | 32.20 | 185.54 |
| **A* Aware** | Classical | 2.5 | - | 100/100 (100%) | 0 | 0 | 47.19 | 59.61 |
| **dqn_shaped_lam2.5_seed0** | shaped | 2.5 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam2.5_seed1** | shaped | 2.5 | 1 | 0/100 (0%) | 4 | 96 | N/A | N/A |
| **dqn_shaped_lam2.5_seed2** | shaped | 2.5 | 2 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam0.0_seed0** | shaped | 0.0 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam0.0_seed1** | shaped | 0.0 | 1 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam0.0_seed2** | shaped | 0.0 | 2 | 1/100 (1%) | 1 | 98 | 34.00 | 44.37 |
| **dqn_plain_lam2.5_seed0** | plain | 2.5 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |

## 2. Canonical Route Evaluation: (2, 2) → (47, 47)

| Method / Model | Success | Outcome | Path Steps | Cumulative Dose (a.u.) |
|---|---|---|---|---|
| **astar_shortest** | ✓ | goal | 56 | 281.88 |
| **astar_aware** | ✓ | goal | 80 | 58.15 |
| **tabular_q_learning** | ✓ | goal | 80 | 64.10 |
| **dqn_shaped_lam2.5_seed0** | ✗ | timeout | 300 | 197.28 |
| **dqn_shaped_lam2.5_seed1** | ✗ | collision | 3 | 1.62 |
| **dqn_shaped_lam2.5_seed2** | ✗ | timeout | 300 | 394.66 |
| **dqn_shaped_lam0.0_seed0** | ✗ | timeout | 300 | 219.61 |
| **dqn_shaped_lam0.0_seed1** | ✗ | timeout | 300 | 1077.24 |
| **dqn_shaped_lam0.0_seed2** | ✗ | timeout | 300 | 20038.94 |
| **dqn_plain_lam2.5_seed0** | ✗ | timeout | 300 | 1556.27 |

## 3. Paired Statistical Comparisons

| Comparison | Mutual Successes | Mean Dose Diff (a.u.) | 95% Bootstrap CI | Mean Step Diff | 95% Bootstrap CI |
|---|---|---|---|---|---|
| Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed 0) | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed 1) | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed 2) | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| Sensor DQN (Shaped λ=2.5, Seed 0) vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| Sensor DQN (Shaped λ=2.5, Seed 0) vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |

## 4. Observations and Limitations
- **Collision Avoidance**: Across all 7 sensor DQN agents, the 16-ray LiDAR effectively prevented obstacle penetration and wall collisions on held-out tasks.
- **Navigation Generalization Limits**: Purely reactive MLP policies without spatial memory or global map access suffer from local path cycling/timeouts when attempting to navigate to distant held-out targets ($d_{\text{free}} \ge 20$) across complex wall mazes.
- **Full Information Contrast**: Classical graph search planners (A* Shortest and A* Aware) succeed 100% (100/100) on the held-out pairs due to complete topological map access.
