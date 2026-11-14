# Sensor-Based Agent Evaluation Report (TEST Set)

Evaluation across 100 pairs from the **test** set against classical A* baselines.

## 1. Benchmark on 100 Evaluation Pairs (test)

| Method / Model | Variant | Objective λ | Seed | Success Rate | Collisions | Timeouts | Mean Steps | Mean Dose (a.u.) |
|---|---|---|---|---|---|---|---|---|
| **A* Shortest** | Classical | 0.0 | - | 100/100 (100%) | 0 | 0 | 32.80 | 214.48 |
| **A* Aware** | Classical | 2.5 | - | 100/100 (100%) | 0 | 0 | 50.30 | 59.79 |
| **dqn_shaped_lam2.5_seed0** | shaped | 2.5 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam2.5_seed1** | shaped | 2.5 | 1 | 0/100 (0%) | 8 | 92 | N/A | N/A |
| **dqn_shaped_lam2.5_seed2** | shaped | 2.5 | 2 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam0.0_seed0** | shaped | 0.0 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam0.0_seed1** | shaped | 0.0 | 1 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_shaped_lam0.0_seed2** | shaped | 0.0 | 2 | 0/100 (0%) | 1 | 99 | N/A | N/A |
| **dqn_s6_m1_lam2.5_seed0** | variant | 2.5 | 0 | 1/100 (1%) | 11 | 88 | 29.00 | 158.00 |

## 3. Paired Statistical Comparisons

| Comparison | Mutual Successes | Mean Dose Diff (a.u.) | 95% Bootstrap CI | Mean Step Diff | 95% Bootstrap CI |
|---|---|---|---|---|---|
| Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed 0) | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed 1) | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| Sensor DQN Shaped λ=2.5 vs λ=0.0 (Seed 2) | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed0 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed0 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed1 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed1 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed2 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed2 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam0.0_seed0 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam0.0_seed0 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam0.0_seed1 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam0.0_seed1 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam0.0_seed2 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam0.0_seed2 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m1_lam2.5_seed0 vs. A* Shortest | N=1 | -45.77 | [-45.77, -45.77] | +6.00 | [6.00, 6.00] |
| dqn_s6_m1_lam2.5_seed0 vs. A* Aware | N=1 | +69.88 | [69.88, 69.88] | -26.00 | [-26.00, -26.00] |
