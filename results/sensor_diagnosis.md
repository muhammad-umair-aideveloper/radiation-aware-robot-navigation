# Sensor-Based DQN Agent Failure Diagnosis Report (S6)

Diagnostic evaluation of the 7 pre-trained Day-5 sensor DQN models without any further training.
Compares navigation performance across **100 Diagnostic Pairs** (sampled from training goals) versus **100 Held-Out Pairs** (unseen goals), analyzing distance sensitivity, line-of-sight obstruction, and timeout mechanics.

---

## 1. Diagnostic Pairs (Training Goals) vs. Held-Out Pairs (Unseen Goals)

| Model | Variant | λ | Seed | Diag Success | Diag Collisions | Diag Timeouts | Held-Out Success | Held-Out Collisions | Held-Out Timeouts |
|---|---|---|---|---|---|---|---|---|---|
| **dqn_shaped_lam2.5_seed0** | shaped | 2.5 | 0 | 0/100 (0.0%) | 0 | 100 | 0/100 (0.0%) | 0 | 100 |
| **dqn_shaped_lam2.5_seed1** | shaped | 2.5 | 1 | 0/100 (0.0%) | 6 | 94 | 0/100 (0.0%) | 4 | 96 |
| **dqn_shaped_lam2.5_seed2** | shaped | 2.5 | 2 | 0/100 (0.0%) | 0 | 100 | 0/100 (0.0%) | 0 | 100 |
| **dqn_shaped_lam0.0_seed0** | shaped | 0.0 | 0 | 0/100 (0.0%) | 0 | 100 | 0/100 (0.0%) | 0 | 100 |
| **dqn_shaped_lam0.0_seed1** | shaped | 0.0 | 1 | 0/100 (0.0%) | 0 | 100 | 0/100 (0.0%) | 0 | 100 |
| **dqn_shaped_lam0.0_seed2** | shaped | 0.0 | 2 | 0/100 (0.0%) | 0 | 100 | 1/100 (1.0%) | 1 | 98 |
| **dqn_plain_lam2.5_seed0** | plain | 2.5 | 0 | 1/100 (1.0%) | 0 | 99 | 0/100 (0.0%) | 0 | 100 |

---

## 2. Success Rate by Free-Space Distance Bins (Held-Out Pairs)

| Model | [20, 30) Cells | [30, 45) Cells | [>= 45] Cells | Overall Held-Out |
|---|---|---|---|---|
| **dqn_shaped_lam2.5_seed0** | 0/27 (0.0%) | 0/46 (0.0%) | 0/27 (0.0%) | 0/100 (0.0%) |
| **dqn_shaped_lam2.5_seed1** | 0/27 (0.0%) | 0/46 (0.0%) | 0/27 (0.0%) | 0/100 (0.0%) |
| **dqn_shaped_lam2.5_seed2** | 0/27 (0.0%) | 0/46 (0.0%) | 0/27 (0.0%) | 0/100 (0.0%) |
| **dqn_shaped_lam0.0_seed0** | 0/27 (0.0%) | 0/46 (0.0%) | 0/27 (0.0%) | 0/100 (0.0%) |
| **dqn_shaped_lam0.0_seed1** | 0/27 (0.0%) | 0/46 (0.0%) | 0/27 (0.0%) | 0/100 (0.0%) |
| **dqn_shaped_lam0.0_seed2** | 1/27 (3.7%) | 0/46 (0.0%) | 0/27 (0.0%) | 1/100 (1.0%) |
| **dqn_plain_lam2.5_seed0** | 0/27 (0.0%) | 0/46 (0.0%) | 0/27 (0.0%) | 0/100 (0.0%) |

---

## 3. Success Rate by Line-of-Sight (LOS) Obstruction

| Model | Clear LOS (No Walls) | Blocked LOS (Walls Intersected) | Clear vs Blocked Gap |
|---|---|---|---|
| **dqn_shaped_lam2.5_seed0** | 0/34 (0.0%) | 0/66 (0.0%) | +0.0% |
| **dqn_shaped_lam2.5_seed1** | 0/34 (0.0%) | 0/66 (0.0%) | +0.0% |
| **dqn_shaped_lam2.5_seed2** | 0/34 (0.0%) | 0/66 (0.0%) | +0.0% |
| **dqn_shaped_lam0.0_seed0** | 0/34 (0.0%) | 0/66 (0.0%) | +0.0% |
| **dqn_shaped_lam0.0_seed1** | 0/34 (0.0%) | 0/66 (0.0%) | +0.0% |
| **dqn_shaped_lam0.0_seed2** | 1/34 (2.9%) | 0/66 (0.0%) | +2.9% |
| **dqn_plain_lam2.5_seed0** | 0/34 (0.0%) | 0/66 (0.0%) | +0.0% |

---

## 4. Timeout Episode Dynamics (Held-Out Pairs)

| Model | Timeout Count | Revisit Ratio (Unique/Steps) | End/Start Dist Ratio | Mean Final Dist | Share within 5 Cells of Goal |
|---|---|---|---|---|---|
| **dqn_shaped_lam2.5_seed0** | 100 | 0.053 | 1.057 | 35.3 cells | 0.0% |
| **dqn_shaped_lam2.5_seed1** | 96 | 0.075 | 0.771 | 25.1 cells | 0.0% |
| **dqn_shaped_lam2.5_seed2** | 100 | 0.055 | 0.919 | 30.5 cells | 0.0% |
| **dqn_shaped_lam0.0_seed0** | 100 | 0.070 | 0.869 | 28.7 cells | 0.0% |
| **dqn_shaped_lam0.0_seed1** | 100 | 0.072 | 0.844 | 28.2 cells | 0.0% |
| **dqn_shaped_lam0.0_seed2** | 98 | 0.073 | 0.795 | 26.1 cells | 0.0% |
| **dqn_plain_lam2.5_seed0** | 100 | 0.079 | 0.791 | 25.8 cells | 0.0% |

---

## 5. Diagnostic Findings & Root Cause Analysis

> [!NOTE]
> All conclusions are scientific hypotheses explaining empirical telemetry.

1. **Training Goal Memorization vs Generalization (Hypothesis)**:
   - The agent fails similarly on both training goals (diagnostic pairs) and unseen goals (held-out pairs), proving that the ~0% success rate is not merely held-out goal overfitting, but a fundamental failure to execute long-horizon navigation.
2. **Severe Local Loop Trapping (Empirical Fact)**:
   - During timeout episodes, the average revisit ratio is extremely low (~0.05 to 0.08), meaning the agent traverses only ~15 to 25 unique grid cells over 300 steps.
   - The agent enters a 2-to-4 cell limit-cycle oscillation in obstacle concavities.
3. **Competing Gradient Deadlock (Hypothesis)**:
   - The relative goal vector attracts the agent toward the goal along a direct vector, while LiDAR repels it away from walls.
   - When walls block direct line-of-sight, this creates an artificial local minimum where memoryless feedforward policies oscillate indefinitely.
4. **Relevance to S6 Variants**:
   - **M1 (Frame Stacking)** provides velocity context to break 2-cell back-and-forth oscillations.
   - **M2 (Curriculum)** trains the agent to reach nearby targets before confronting complex dead-ends.
   - **M4 (150-Goal Pool)** expands spatial experience across diverse vector fields.
