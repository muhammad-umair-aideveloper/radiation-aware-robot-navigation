# Experimental Protocol and Plans

## Radiation Weight Sweep Plan

**Status**: Plan fixed before running experiments.

### 1. Objective
Study how the radiation weight $\lambda \in \{0, 0.5, 1, 2.5, 5, 10\}$ trades absorbed radiation dose against trajectory path length, and validate planner optimality against an exact dynamic programming (Dijkstra) solver.

### 2. Method Configurations
For each step transition to cell $s'$ with action length $L \in \{1, \sqrt{2}\}$ and radiation intensity $I(s')$:
- **Step Cost**:
  $$c(s, s') = L \cdot (1 + \lambda \cdot I(s'))$$
- **A\* Shortest Path**: Reference planner minimizing path length only ($\lambda = 0$, step cost $L$).
- **A\* Radiation-Aware**: Graph search minimizing path length + radiation cost with weighting $\lambda \in \{0, 0.5, 1, 2.5, 5, 10\}$.
- **Exact DP / Dijkstra Reference**: Dijkstra algorithm computing global minimum cost paths on the 8-connected grid with identical collision constraints (no corner cutting).
- **Tabular Q-Learning**:
  - State space: $50 \times 50$ grid, 8 discrete actions
  - Reward per step: $-L \cdot (1 + \lambda \cdot I(s'))$
  - Terminal goal bonus: $+1000$, Terminal collision penalty: $-1000$
  - Hyperparameters: $\alpha = 0.15$, $\gamma = 0.98$, $\epsilon$ geometric decay $1.0 \to 0.02$, max 300 steps/episode, 50,000 episodes
  - Seeds: $\{0, 1, 2\}$
  - 50% random free start, 50% canonical start $(2, 2)$

### 3. Evaluation Protocol
- **Evaluation Set**: The fixed set of 50 start cells (seed 123, free-space geodesic distance $\ge 20$ to goal $(47, 47)$).
- **Canonical Pair**: Start $(2, 2)$ to goal $(47, 47)$.
- **Metrics**:
  1. Success count / rate on the 50 starts (reaching goal within 300 steps without collision).
  2. Mean steps (path length in cells) on successful episodes.
  3. Mean cumulative dose (arbitrary units, a.u.) on successful episodes.
  4. Paired differences on starts where both compared methods succeed. Failed episodes are never averaged into dose.

