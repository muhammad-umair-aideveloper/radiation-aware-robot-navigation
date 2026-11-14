# Sensor-Based Agent Evaluation Report (VAL Set)

Evaluation across 100 pairs from the **val** set against classical A* baselines.

## 1. Benchmark on 100 Evaluation Pairs (val)

| Method / Model | Variant | Objective λ | Seed | Success Rate | Collisions | Timeouts | Mean Steps | Mean Dose (a.u.) |
|---|---|---|---|---|---|---|---|---|
| **A* Shortest** | Classical | 0.0 | - | 100/100 (100%) | 0 | 0 | 31.26 | 192.81 |
| **A* Aware** | Classical | 2.5 | - | 100/100 (100%) | 0 | 0 | 47.34 | 54.29 |
| **dqn_shaped_lam2.5_seed0** | shaped | 2.5 | 0 | 1/100 (1%) | 0 | 99 | 21.00 | 138.79 |
| **dqn_shaped_lam2.5_seed1** | shaped | 2.5 | 1 | 0/100 (0%) | 6 | 94 | N/A | N/A |
| **dqn_shaped_lam2.5_seed2** | shaped | 2.5 | 2 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_s6_m1_lam2.5_seed0** | variant | 2.5 | 0 | 0/100 (0%) | 6 | 94 | N/A | N/A |
| **dqn_s6_m2_lam2.5_seed0** | variant | 2.5 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_s6_m3_lam2.5_seed0** | variant | 2.5 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |
| **dqn_s6_m4_lam2.5_seed0** | variant | 2.5 | 0 | 0/100 (0%) | 0 | 100 | N/A | N/A |

## 3. Selection Rule Application & Decision

- **Pre-Declared Selection Rule**: Pick the variant with highest validation success rate; ties broken by fewer timeouts.
- **Usability Threshold**: If the best variant achieves validation success $< 10\%$, declare no usable variant, skip Stage B, and proceed directly to final test evaluation of baseline and the best variant (Seed 0) to report a negative result.
- **Variant Validation Performance**:
  - **M1 (Frame Stacking, k=4)**: 0/100 (0.0%) success, 6 collisions, 94 timeouts.
  - **M2 (Distance Curriculum)**: 0/100 (0.0%) success, 0 collisions, 100 timeouts.
  - **M3 (M1 + M2)**: 0/100 (0.0%) success, 0 collisions, 100 timeouts.
  - **M4 (150-Goal Pool)**: 0/100 (0.0%) success, 0 collisions, 100 timeouts.
- **Baseline Day-5 Performance**:
  - `dqn_shaped_lam2.5_seed0`: 1/100 (1.0%) success, 0 collisions, 99 timeouts.
  - `dqn_shaped_lam2.5_seed1`: 0/100 (0.0%) success, 6 collisions, 94 timeouts.
  - `dqn_shaped_lam2.5_seed2`: 0/100 (0.0%) success, 0 collisions, 100 timeouts.
- **Decision**:
  The best variant (M1, with 94 timeouts breaking the 0% tie) achieved **0.0% validation success**, which is strictly below the pre-declared 10% usability threshold.
  **Stage B is skipped**. In accordance with the pre-declared protocol, we evaluate the baseline models and the best variant (M1, Seed 0) on the test set once and report a negative result.

| Comparison | Mutual Successes | Mean Dose Diff (a.u.) | 95% Bootstrap CI | Mean Step Diff | 95% Bootstrap CI |
|---|---|---|---|---|---|
| dqn_shaped_lam2.5_seed0 vs. A* Shortest | N=1 | +39.00 | [39.00, 39.00] | +4.00 | [4.00, 4.00] |
| dqn_shaped_lam2.5_seed0 vs. A* Aware | N=1 | +93.56 | [93.56, 93.56] | -3.00 | [-3.00, -3.00] |
| dqn_shaped_lam2.5_seed1 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed1 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed2 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_shaped_lam2.5_seed2 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m1_lam2.5_seed0 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m1_lam2.5_seed0 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m2_lam2.5_seed0 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m2_lam2.5_seed0 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m3_lam2.5_seed0 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m3_lam2.5_seed0 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m4_lam2.5_seed0 vs. A* Shortest | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
| dqn_s6_m4_lam2.5_seed0 vs. A* Aware | N=0 (no mutual success) | N/A | N/A | N/A | N/A |
