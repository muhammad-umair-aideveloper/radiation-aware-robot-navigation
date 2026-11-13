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

---

## S5: Sensor-Based Agent Study

**Status**: Plan fixed before running experiments.

### 1. Objective
Train a map-free, sensor-driven navigation agent using a custom NumPy-only Deep Q-Network (DQN) architecture equipped with a 16-ray LiDAR and a local $3 \times 3$ Geiger radiation sensor array. Evaluate whether the agent learns generalizable obstacle avoidance and radiation avoidance on unseen, held-out goal locations and start configurations without access to absolute coordinate locations or global grid maps.

### 2. Observation Space (28 Features)
All observations are continuous values derived strictly from local egocentric perception and relative goal guidance (no absolute $(r, c)$ coordinates):
1. **16-Ray LiDAR Distances (16 features)**:
   - 16 planar range beams cast every $22.5^\circ$ ($0^\circ, 22.5^\circ, \dots, 337.5^\circ$).
   - Maximum sensor range: $R_{\max} = 15$ cells.
   - Values normalized to $[0, 1]$ via $d / R_{\max}$. If no obstacle or grid boundary is intersected within $R_{\max}$, the reading is $1.0$.
2. **Local $3 \times 3$ Geiger Intensity Readings (9 features)**:
   - Radiation intensity sampled at the agent's current cell and its 8 immediate Moore neighbors (ordered consistently row-major from top-left to bottom-right).
   - Any neighbor cell outside grid bounds or inside a wall obstacle reads strictly $0.0$.
3. **Relative Goal Vector (2 features)**:
   - Coordinate difference $(\Delta r, \Delta c) = (r_{\text{goal}} - r_{\text{agent}}, c_{\text{goal}} - c_{\text{agent}})$, normalized by grid dimension: $(\Delta r / 50, \Delta c / 50)$.
4. **Euclidean Goal Distance (1 feature)**:
   - Direct Euclidean distance $\|p_{\text{goal}} - p_{\text{agent}}\|_2$ normalized by maximum grid diagonal: $d_{\text{goal}} / (50\sqrt{2})$.
- Sensors are deterministic and noise-free.

### 3. Action Space, Dynamics & Reward Structure
- **Actions**: Discrete 8-connected actions (N, NE, E, SE, S, SW, W, NW) with step length $L \in \{1, \sqrt{2}\}$ and diagonal corner-cutting constraints identical to existing planners.
- **Base Environment Reward**:
  $$R_{\text{step}} = -L \cdot (1 + \lambda \cdot I(s'))$$
  $$R_{\text{goal}} = +1000.0 \quad (\text{terminal}), \quad R_{\text{collision}} = -1000.0 \quad (\text{terminal})$$
  Maximum episode length: 300 steps.
- **Network Training Scaling**: To ensure gradient stability during neural network backpropagation, rewards fed to the replay buffer / loss computation are scaled down by a factor of 100:
  $$\tilde{R} = R / 100.0$$
  Reported cumulative doses and metrics are never scaled and remain in true arbitrary units (a.u.).

### 4. Goal Pools and Evaluation Sets
- **Training Goal Pool**: 30 seeded goal cells uniformly sampled from free space satisfying low radiation threshold $I < 0.05$ a.u.
- **Held-Out Goal Pool**: 20 seeded goal cells disjoint from the training pool, also on free space with $I < 0.05$ a.u.
- Neither pool contains the canonical goal $(47, 47)$.
- **Training Episode Sampling**: Each training episode randomly samples a target goal from the training pool and a valid free start cell with 8-connected free-space geodesic distance $d_{\text{free}} \ge 20$ cells to that goal.
- **Held-Out Evaluation Benchmark (100 pairs)**:
  - 20 held-out goals $\times$ 5 seeded start cells each ($d_{\text{free}} \ge 20$), fixed and saved to `results/sensor_pairs.json` prior to any model training.
- Canonical route $(2, 2) \to (47, 47)$ is evaluated and reported separately.

### 5. DQN Architecture & Hyperparameters
- **Implementation**: Pure NumPy with manual analytical backpropagation and Adam optimizer (no external deep learning frameworks).
- **Network**: Multi-Layer Perceptron (MLP) with dimensions $28 \to 64 \to 64 \to 8$.
- **Activations**: ReLU for hidden layers, linear output for action values $Q(s, a)$.
- **Optimizer**: Adam ($\text{lr} = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999, \epsilon_{\text{adam}} = 10^{-8}$).
- **Loss Function**: Huber loss (smooth L1 with $\delta = 1.0$) for robust TD-error updates.
- **Discount Factor**: $\gamma = 0.99$.
- **Replay Buffer**: Capacity of 50,000 transitions, uniform random minibatch sampling of size $B = 64$.
- **Update Frequency**: 1 gradient update step every 4 environment steps.
- **Target Network**: Periodic hard copy parameter sync every 1,000 environment steps.
- **Exploration Schedule**: Epsilon-greedy decaying geometrically from $\epsilon = 1.0$ down to $\epsilon = 0.05$ over the first 60% of total training step budget, remaining fixed at $0.05$ thereafter.

### 6. Experimental Variants & Training Runs
- **Reward Shaping Variants**:
  - `shaped`: Potential-based reward shaping $F(s, s') = \gamma \Phi(s') - \Phi(s)$, with potential function $\Phi(s) = -d_{\text{free}}(s) / 100.0$ (where $d_{\text{free}}$ is the precomputed geodesic free-space distance to the episode's goal ignoring radiation, and $\Phi(\text{terminal}) = 0$). By Ng et al. (1999), this guarantees policy invariance while providing dense navigation gradients.
  - `plain`: Unshaped base environment reward ($F = 0$).
- **Planned Experimental Runs (7 total)**:
  1. `shaped`, $\lambda = 2.5$, Seed 0
  2. `shaped`, $\lambda = 2.5$, Seed 1
  3. `shaped`, $\lambda = 2.5$, Seed 2
  4. `shaped`, $\lambda = 0.0$, Seed 0
  5. `shaped`, $\lambda = 0.0$, Seed 1
  6. `shaped`, $\lambda = 0.0$, Seed 2
  7. `plain`, $\lambda = 2.5$, Seed 0
- **Budget**: Uniform total environment step budget across all 7 runs, determined once from an empirical speed test (~8 minutes execution per run on 2-core CPU).
- **Checkpointing**: Final model weights of each run only are evaluated; no cherry-picking or checkpoint selection on evaluation pairs.

### 7. Evaluation Protocol & Statistical Metrics
- Evaluated on the 100 held-out pairs under a purely greedy policy ($\epsilon = 0.0$) with a 300-step loop guard.
- Outcomes categorized as: `success` (reached goal), `collision` (hit obstacle or boundary), or `timeout` (exceeded 300 steps).
- For successful trajectories: measure path steps and cumulative absorbed radiation dose (a.u.).
- **Paired Statistical Comparisons**:
  - Evaluated strictly on the subset of starts where both compared methods succeed.
  - Compute paired difference in steps and dose with two-sided 95% bootstrap confidence intervals (10,000 resamples, deterministic seed 42).
  - Key comparisons:
    1. Sensor DQN ($\lambda = 2.5$) vs. A* Shortest
    2. Sensor DQN ($\lambda = 2.5$) vs. A* Aware ($\lambda = 2.5$)
    3. Primary hypothesis test: Sensor DQN $\lambda = 2.5$ vs. $\lambda = 0.0$ for the same seed.
  - Tabular Q-learning agent was trained specifically on the canonical single-goal task and is compared strictly on the canonical $(2, 2) \to (47, 47)$ route.

### 8. Declared Step Budget from Measured Speed
- **Speed Test Measurement**: 20,000 environment steps completed in 22.45 seconds ($891.0$ steps/second) on a 2-core CPU.
- **Target Wall-Clock Duration**: ~7.5 to 8.0 minutes ($450$ to $480$ seconds) per run.
- **Declared Step Budget**: **400,000 steps** per run ($400,000 / 891.0 \approx 448.9\text{ s} \approx 7.5\text{ minutes}$).
- **Status**: Fixed uniformly for all 7 experimental runs before launching training.



