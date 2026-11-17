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

---

## S6: Improving the Sensor Agent

**Status**: Plan fixed before running experiments. No step budget increase and no extra variants after seeing results.

### 1. Objective & Motivation
In Study S5, the purely reactive sensor-based DQN agent achieved near-zero collisions (obstacle avoidance succeeded via LiDAR) but suffered a 0% to 1% success rate on 100 held-out pairs due to 96% to 100% timeouts. The objective of Study S6 is:
1. Diagnose why timeouts occur using the existing 7 finished runs without any retraining.
2. Formulate and test 4 pre-declared architectural and training improvements under a leakage-free protocol.
3. Report the outcome honestly, including negative results if no variant meets the pre-declared usability threshold.

### 2. Fixed Parameters Across All Runs
Unless explicitly overridden by a variant below, all experimental settings remain strictly identical to Study S5:
- **DQN Architecture**: MLP ($H_1=64, H_2=64$, ReLU hidden, linear output), Adam ($\text{lr}=10^{-3}, \beta_1=0.9, \beta_2=0.999$), Huber loss ($\delta=1.0$), gradient norm clipping at $10.0$.
- **Replay & Updates**: Replay buffer 50,000 transitions, batch size 64, 1 update per 4 environment steps, target network sync every 1,000 steps.
- **Exploration**: Geometric epsilon decay from $1.0$ down to $0.05$ over the first 60% of steps (240,000 steps), constant $0.05$ thereafter.
- **Reward & Physics**: Scaled reward $\tilde{R} = R/100$, base cost $L \cdot (1 + \lambda \cdot I)$, terminal $+10.0$ goal, $-10.0$ collision, potential-based shaping with geodesic free-space distance, $\lambda = 2.5$, episode timeout guard at 300 steps.
- **Training Step Budget**: Exactly 400,000 steps per run.
- **Evaluation Rules**: Purely greedy policy ($\epsilon = 0.0$), 300-step timeout guard, final model evaluation only (no intermediate checkpoint selection).

### 3. Four Pre-Declared Improvement Variants (Stage A, $\lambda = 2.5$, Seed 0)
- **M1 (Frame Stacking)**:
  - Observation concatenates the last 4 consecutive 28-dimensional observations into a 112-dimensional vector.
  - At episode reset, the initial observation is repeated 4 times.
  - Oldest observation is dropped upon each transition.
  - Network input layer dimension: 112 ($112 \to 64 \to 64 \to 8$).
  - *Hypothesis*: Short temporal history allows the network to infer velocity and direction, breaking memoryless limit-cycle oscillations.
- **M2 (Distance Curriculum)**:
  - The minimum free-space start-goal distance $d_{\text{min}}$ of training episodes rises linearly from $5$ to $20$ cells over the first 40% of the training step budget ($160,000$ steps), then stays fixed at $20$ cells for the remaining 60%.
  - Network input dimension remains 28 ($28 \to 64 \to 64 \to 8$).
  - *Hypothesis*: Learning to reach nearby goals first builds effective local gradient navigation before tackling long-range topological detours.
- **M3 (M1 + M2 Combined)**:
  - Both 4-frame stacking (112-dim observation) and the linear distance curriculum ($5 \to 20$ cells over 40% of steps).
- **M4 (Larger Goal Pool)**:
  - Expands the training goal pool from 30 goals to 150 goals (retaining the 30 existing training goals and adding 120 new disjoint goals sampled under the same rules).
  - Network input dimension remains 28.
  - *Hypothesis*: Training over 5x more diverse goal locations prevents overfitting to specific local gradient fields and improves spatial generalization.

### 4. Pair Sets & Leakage Prevention Protocol
All goal locations must be free cells satisfying low radiation $I < 0.05$ a.u., excluding the canonical goal $(47, 47)$. To strictly prevent data leakage:
- **Pairwise Disjoint Goal Sets**:
  - `training_pool_30`: 30 existing training goals.
  - `heldout_set`: 20 existing held-out goals from S5 (used strictly for offline diagnosis, never for model selection or final testing).
  - `val_set`: NEW validation set composed of 20 new goals $\times$ 5 starts each ($N=100$ pairs, $d_{\text{free}} \ge 20$). Used solely for Stage A model selection.
  - `test_set`: NEW test set composed of 20 new goals $\times$ 5 starts each ($N=100$ pairs, $d_{\text{free}} \ge 20$). Used ONCE for final guarded evaluation.
  - `diag_set`: 100 diagnostic pairs constructed from the 30 training goals ($20 \times 5$ pairs, $d_{\text{free}} \ge 20$) to measure training-goal memorization vs. generalization.
  - `training_pool_150`: The 30 existing training goals plus 120 newly sampled training goals (used only by M4).
- **Separation Constraint**: Every new goal in `val_set`, `test_set`, and `training_pool_150` must maintain a minimum Euclidean distance of at least 5.0 cells from all goals belonging to other sets.

### 5. Selection Rule (Stage A $\to$ Stage B Gate)
- After Stage A training (M1, M2, M3, M4 on Seed 0, $\lambda = 2.5$), all 4 variants and the baseline Day-5 models (shaped $\lambda = 2.5$, Seeds 0, 1, 2) are evaluated on the NEW 100 validation pairs under greedy rollout.
- **Primary Selection Metric**: Highest validation success rate.
- **Tie-Breaker**: Fewer timeouts.
- **Usability Gate**:
  - If the best variant achieves validation success $\ge 10\%$, it is selected as the winning variant for Stage B.
  - If the best variant achieves validation success $< 10\%$, there is no usable variant. Stage B is skipped entirely, and only the final test evaluation of the baseline and the best variant (Seed 0) is executed to report a definitive negative result.

### 6. Stage B Training (Conditional)
- If the usability gate passes ($\ge 10\%$ validation success):
  - Train the selected variant on $\lambda = 2.5$ for Seeds 1 and 2 (reusing Seed 0 from Stage A).
  - Train the selected variant on $\lambda = 0.0$ for Seeds 0, 1, and 2.
  - Total runs in Stage B: 5 new training runs (reusing 1).

### 7. Final Test Evaluation Protocol (Guarded, Executed Once)
- Evaluated on the 100 NEW test pairs (strictly once, guarded by `evaluate_sensor.py` logging to `results/test_usage_log.md`):
  - Selected variant (6 runs: 3 seeds at $\lambda = 2.5$, 3 seeds at $\lambda = 0.0$; or Seed 0 if negative result path).
  - Baseline Day-5 models (6 runs: 3 seeds at $\lambda = 2.5$, 3 seeds at $\lambda = 0.0$).
  - Classical baselines: A* Shortest ($\lambda = 0.0$) and A* Aware ($\lambda = 2.5$).
- **Reported Metrics**:
  - Success rate, collision count, timeout count.
  - Mean steps and mean dose (a.u.) computed strictly over successful trajectories.
  - Paired comparisons on mutual successes with 95% bootstrap confidence intervals (10,000 resamples, RNG seed 42):
    1. Selected variant $\lambda = 2.5$ vs. $\lambda = 0.0$ per seed (primary hypothesis test on radiation awareness).
    2. Selected variant vs. A* Shortest.
    3. Selected variant vs. A* Aware.
- All explanatory conclusions are explicitly labeled as hypotheses.

## S7: measure, estimate, plan (fixed before running)
- True field I (a.u.) comes from the environment. A reading at a free cell with dwell time t: counts ~ Poisson(t * I(cell)); the estimate of I at that cell is counts / t.
- Dwell times t in {100, 10, 1, 0.1}. Measurement cells: N distinct free cells chosen uniformly at random with a seed; N in {10, 25, 50, 100, 200}. The travel cost of measuring is not counted (limitation).
- Estimators (numpy only; they receive ONLY the measurement cells, their estimates and the free mask): "kernel" = Gaussian Nadaraya-Watson regression with the bandwidth chosen by leave-one-out cross-validation on the measurements, from (1, 2, 3, 4, 6, 8, 12) cells; "idw" = inverse-distance weighting, power 2, eps 0.5. Estimates are clipped at 0 and set to 0 on wall cells.
- Planning: A* aware (step cost L * (1 + 2.5 * I_est)) on the estimated field. References: A* aware on the true field (oracle) and A* shortest. The path is always judged on the TRUE field.
- Evaluation: the same 50 fixed start cells as the main benchmark; measurement seeds 0 to 9; 2 estimators x 5 values of N x 4 values of t = 40 configurations. Per configuration: mean steps, mean TRUE dose, dose regret vs oracle in percent ((dose - oracle dose) / oracle dose * 100), dose reduction vs A* shortest in percent, and the relative L2 error of the estimated field vs the true field on free cells. Report the mean over seeds with a 95% bootstrap confidence interval over the 10 seed means (10,000 resamples, fixed seed).
- Explanations of results are hypotheses. No new configurations after seeing results.
### Code interface notes
- true field: `RadiationEnv().radiation` (or `compute_radiation_field()`).
- wall grid: `RadiationEnv().walls` (`True` means wall; free cells are `~env.walls`).
- A* aware with a custom field: `astar_search(start, goal, mode="aware", env=RadiationEnv(radiation=custom_field), lambda_rad=2.5)`.
- 50 start cells: `main.generate_evaluation_starts(env, num_starts=50, seed=123)`; free-space distance to the goal is at least 20.
- bootstrap helper: `src/evaluate_sensor.py`, `compute_bootstrap_ci`.
- dose of a path: `src/map_error_study.py`, `evaluate_path_on_true_field`.

## S8: online measure, estimate, replan (fixed before running)
- Question: if the robot measures only along its own path and replans while walking, how much of the dose saving of the oracle (A* aware on the true field) is kept, compared with offline random measurement cells with the same number of distinct cells?
- Assumptions: walls and the goal are known; the radiation field is unknown; movement is noise-free; a reading takes no extra time and no extra dose (the dwell value t is the counter sensitivity: counts ~ Poisson(t * I(cell)), intensity estimate = counts / t); the robot reads at the start cell before the first plan.
- Measurement patterns: "cell" = the robot's current cell; "3x3" = the free cells of the 3 x 3 neighbourhood of the robot's cell (the same sensor idea as the Day-5 Geiger array). A reading is taken at every visited cell.
- Estimators (from S7, unchanged): "kernel" (Gaussian, leave-one-out bandwidth) and "idw". Readings at the same cell are pooled by averaging their intensity estimates. With fewer than 3 distinct measured cells the estimate is a constant field equal to the mean of the readings on free cells (0 on walls); this is a fixed rule.
- Replanning: A* aware (step cost L * (1 + 2.5 * I_est)) from the current cell on the current estimate, every 3 steps (and at the start); the maximum is 300 steps.
- Dwell values t in {10, 1, 0.1}; patterns {cell, 3x3}; estimators {kernel, idw}: 12 configurations. Starts: every second one of the 50 fixed start cells (25 starts); noise seeds 0 to 4.
- References on the same 25 starts: oracle (A* aware on the true field), A* shortest, and "offline matched": the S7 offline procedure with N = the rounded mean number of distinct cells that the online robot measured in that configuration, same estimator, same dwell, seeds 0 to 4.
- Metrics: success rate (reaching the goal within 300 steps); mean steps, mean TRUE dose of successful episodes (if the success rate is below 100% say so next to the dose); dose regret vs oracle and dose reduction vs A* shortest in percent (same formulas as S7); mean readings, mean distinct cells, mean replans. Seed means with a 95% bootstrap confidence interval (10,000 resamples, fixed seed) over the 5 seed means.
- Explanations are hypotheses. No new configurations after seeing results.
### Code interface notes
- plan A* aware on a field: `src/planners.py`, `astar_search(start, goal, mode="aware", env=RadiationEnv(radiation=field), lambda_rad=2.5)` returns `(path, total_dose, total_steps, nodes_expanded)`, where `path` is a list of `(row, col)` cells or `None`.
- dose of a path: `src/map_error_study.py`, `evaluate_path_on_true_field(path, true_field)` returns `(dose, path_length)`.
- offline configuration runner: `src/estimation_study.py`, `run_configuration(estimator, n, dwell, env, starts)` runs each seed and collects per-start true doses before aggregating seed metrics.
- regret / reduction formula: `run_configuration` uses `(dose - oracle_dose) / oracle_dose * 100` and `(shortest_dose - dose) / shortest_dose * 100`.
- bootstrap helper: `src/estimation_study.py`, `bootstrap(values)`, using 10,000 resamples with fixed seed 42.
- true field, wall grid, goal, starts, free_mask: `env.radiation`, `env.walls` (`True` means wall), `env.goal`, and `main.generate_evaluation_starts(env, num_starts=50, seed=123)`; `free_mask = ~env.walls`.
- oracle / shortest references: `src/estimation_study.py`, `references(env, starts)` calls `astar_search` on each start with `mode="aware"` on the true environment and `mode="shortest"`, returning mean doses.

## S9: controls, sensor range and unexplored cells (fixed before running)
- Controls (must pass before any study run). On the 25 S8 starts, run_online_episode with a bypass of the estimator: C1 "true": every replan uses the TRUE field; the cost of the executed path under the objective sum of L * (1 + 2.5 * I(new cell)) must equal the cost of the oracle's A* aware path from the same start within a relative difference of 1e-9, and must never be lower than it by more than 1e-9. C2 "zero": every replan uses a field of zeros; the executed path length (sum of step lengths L) must equal the A* shortest path length within 1e-9. If C1 or C2 fails, the simulator has a bug: stop, report, run no study.
- Diagnostic "noise-free": the robot's readings equal the true intensity at the measured cells (no Poisson noise). It separates the effect of noise from the effect of seeing only part of the map.
- Sensor range: patterns "disc3" and "disc6" = the free cells whose Euclidean distance from the robot's cell is at most 3 or 6 cells (29 and 113 cells in open space), added to the S8 patterns "cell" and "3x3".
- Unexplored-cell inflation: when planning, every free cell whose distance to the nearest measured cell is larger than 4 cells gets the planning value max(estimate, beta * m), where m is the mean over measured cells of their averaged intensity estimates. beta = 0 means no inflation (the S8 behaviour); beta = 2 is the pessimistic setting. The executed path is always judged on the TRUE field.
- Study grid: (a) noise-free: estimators {kernel, idw} x patterns {cell, 3x3, disc3, disc6}, beta 0, one deterministic run (8 configurations, no confidence interval); (b) noisy: estimators {kernel, idw} x patterns {3x3, disc3, disc6} x dwell {10, 1} x beta {0, 2}, except 3x3 with beta 0 which already exists from S8 (20 new configurations), seeds 0 to 4. The same 25 starts, replan every 3 steps, 300 steps maximum, lambda 2.5 as in S8.
- Metrics as in S8 (success, mean steps, mean TRUE dose of successful episodes, dose regret vs oracle %, dose reduction vs A* shortest %, mean readings, distinct cells, replans); noisy configurations: mean over seeds with a 95% bootstrap confidence interval over the 5 seed means (10,000 resamples, fixed seed).
- Runtime rule: if the projected total runtime of the study is above 90 minutes, use seeds 0 to 2 for all noisy configurations (decided now, not after seeing results).
- Explanations of results are hypotheses. No new configurations after seeing results.




