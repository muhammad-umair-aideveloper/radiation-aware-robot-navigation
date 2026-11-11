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

---

## Multi-Layout Generalization Study (L1)

**Status**: Plan fixed before running experiments.

### 1. Objective
Evaluate whether the radiation-aware path planning advantages generalize across diverse environments beyond the hand-crafted canonical layout.

### 2. Layout Generation Specification
- 10 pseudo-random procedural layouts generated using deterministic seeds 100 to 109.
- Grid shape: $50 \times 50$, with outer border free.
- Walls: 4 to 7 axis-aligned rectangular obstacle blocks, each with side lengths randomly chosen between 3 and 12 cells (inclusive).
- Acceptance criteria:
  - Total wall area fraction strictly between 10% and 25% of total grid cells ($250 \le N_{\text{wall}} \le 625$).
  - Canonical start $(2, 2)$ and canonical goal $(47, 47)$ must be free cells (if not free, resample with the next seed until satisfied).
  - All free cells must form a single connected component (verified via flood fill / BFS from $(2, 2)$).
- Radiation sources: 3 point sources placed uniformly on free cells (distinct from each other and start/goal). Source base intensity $I_0$ drawn uniformly from $[120.0, 200.0]$.
- Field model: Softened inverse-square law with $\epsilon = 2.0$, identical to the canonical environment.

### 3. Evaluation Protocol
- For each layout $k \in \{100, \dots, 109\}$, sample a fixed set of 50 start cells (seed $k \times 10 + 7$, free cells with 8-connected free-space distance to $(47, 47) \ge 20$).
- Compare **A\* Shortest** ($\lambda = 0$) versus **A\* Radiation-Aware** ($\lambda = 2.5$).
- Compute paired dose differences and path length differences strictly on mutual successes ($N \le 50$).
- Statistical uncertainty: compute two-sided 95% bootstrap confidence intervals (10,000 resamples, RNG seed 42) per layout and pooled across all layouts.
- Q-learning: retrained on each layout for 1 seed (seed 0) with default hyperparameters (50,000 episodes, $\alpha = 0.15$, $\gamma = 0.98$, $\epsilon: 1.0 \to 0.02$).

---

## Map-Error Robustness Study (L2)

**Status**: Plan fixed before running experiments.

### 1. Objective
Measure how errors and uncertainty in prior radiation mapping degrade navigation safety when using A* Radiation-Aware planning.

### 2. Perturbation Model
On the canonical fixed layout (sources at $(18,20)$, $(32,30)$, $(25,12)$ with $I_0 \in \{150, 200, 120\}$):
- Perturbation levels: $\Delta_{\text{pos}} \in \{1, 2, 3\}$ cells maximum displacement per axis.
- For each perturbation level: generate 10 pseudo-random perturbations (seeded with deterministic seeds $1000 + 10 \cdot \Delta_{\text{pos}} + i$ for $i \in \{0, \dots, 9\}$).
- For each source $s$:
  - Shift position: $r'_s = r_s + \delta_r$, $c'_s = c_s + \delta_c$ where $\delta_r, \delta_c \sim \text{Uniform}(\{-\Delta_{\text{pos}}, \dots, \Delta_{\text{pos}}\})$, clamped to grid $[0, 49]$.
  - Scale intensity: $I'_0 = I_0 \cdot f$, where $f \sim \text{Uniform}([0.8, 1.2])$.
- The planner plans on the *perturbed* estimated field $\tilde{I}(r, c)$ using A* Aware ($\lambda = 2.5$).
- The execution evaluates the *true* absorbed dose of the resulting trajectory on the unperturbed ground-truth radiation field $I(r, c)$.

### 3. Evaluation & Comparisons
On the fixed 50 evaluation starts:
1. True dose of perturbed A* Aware plans vs. Oracle A* Aware (planned on true field).
2. True dose of perturbed A* Aware plans vs. A* Shortest (radiation-blind baseline).
3. Report success rate, mean executed dose, and dose degradation percentage across perturbation levels $\Delta_{\text{pos}} \in \{1, 2, 3\}$.


