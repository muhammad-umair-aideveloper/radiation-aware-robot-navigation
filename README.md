# Radiation-Aware Robot Navigation: Path Planning & Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%20%7C%20matplotlib%20%7C%20pillow-green.svg)](#)

A simulation study comparing classical graph search (shortest vs. radiation-aware A*) and Tabular Q-Learning for autonomous mobile robot navigation across a 2D environment containing radiation hotspots and structural obstacles.

![Radiation Navigation Animation](results/radiation_navigation.gif)

---

## Results at a Glance

Benchmark summary on 50 fixed evaluation starts ($\ge 20$ cells to goal $(47, 47)$, $\lambda = 2.5$):

| Method | Type | Success Rate | Mean Steps | Mean Dose (a.u.) | Optimality Gap vs. Dijkstra | Planning / Training Time * |
|---|---|---|---|---|---|---|
| **Dijkstra (Exact Reference)** | Global Optimal | 50/50 (100%) | 45.04 | 54.42 | **0.00% (Exact)** | 1.85 s total |
| **A\* Radiation-Aware** | Heuristic Search | 50/50 (100%) | 45.04 | 54.42 | **0.00% (Consistency Check)** | 2.08 s total |
| **Tabular Q-Learning (Best Seed)** | Off-Policy RL | 50/50 (100%) | 43.68 | 65.59 | **18.58% ± 7.02%** | 97.4 ± 15.7 s |
| **Tabular Q-Learning (3 seeds)** | Off-Policy RL | 49.7 ± 0.5/50 (99.3%) | 45.22 ± 0.82 | 64.89 ± 0.35 | *not measured* | 97.4 ± 15.7 s |
| **SARSA (3 seeds, λ=2.5 only)** | On-Policy RL | 49.0 ± 0.8/50 (98.0%) | 46.83 ± 0.69 | 67.01 ± 1.24 | *not measured* | 160.6 ± 5.6 s |
| **A\* Shortest (Radiation-Blind)** | Baseline Search | 50/50 (100%) | 36.18 | 119.15 | **+70.27% ± 68.11%** | 0.45 s total |

*\* Note: Training times were measured sequentially on a 2-core laptop CPU. SARSA was measured only at $\lambda = 2.5$. The 0.00% optimality gap of A* Aware is a consistency check (both methods optimize the identical objective with an admissible heuristic), not evidence that A* is superior. The per-layout Q-learning study uses 1 seed per layout.*

---

## 1. Abstract

Mobile robots deployed in nuclear facilities, particle accelerator halls, and post-accident environments must navigate to target locations while mitigating cumulative radiation exposure. Minimizing path length alone frequently routes agents into high-intensity radiation flux, accelerating hardware degradation or failure. In this study, we formulate radiation avoidance as a cost-augmented path planning and reinforcement learning problem on a $50 \times 50$ grid world. We evaluate Shortest-Path A*, Radiation-Aware A*, and Tabular Q-Learning (trained over 5 seeds) on a benchmark of 50 distant start locations. Empirically, Radiation-Aware A* achieves a 54.3% reduction in mean absorbed radiation dose at the cost of a 24.5% increase in trajectory length compared to Shortest A*. Tabular Q-Learning learns effective detour policies that cut absorbed dose by over 45% compared to the shortest path reference.

---

## 2. Problem Statement & Physics Context

In high-energy particle accelerator facilities (e.g., JINR synchrotrons, beam injection lines, and target stations), secondary particle showers and residual activation generate spatially non-uniform radiation fields. Robotic inspection systems are dispatched to perform telemetry, leak detection, and maintenance to uphold the **ALARA** (As Low As Reasonably Achievable) radiation protection principle for human personnel.

However, onboard electronics (semiconductor CMOS sensors, microcontrollers) are susceptible to Single Event Effects (SEE) and Total Ionizing Dose (TID). A robot that takes the Euclidean shortest path risks navigating near intense activation sources, risking equipment failure. The objective is to compute trajectories that balance path length $L$ and cumulative dose $D$.

---

## 3. Mathematical Formulation

### 3.1 Radiation Field Model
The environment features $S = 3$ point radiation hotspots with source positions $(r_s, c_s)$ and source intensities $I_{0, s}$. Intensity decreases with squared Euclidean distance, with a softening parameter $\epsilon = 2.0$ preventing mathematical singularities at source centers:

$$I(r, c) = \sum_{s=1}^{S} \frac{I_{0, s}}{(r - r_s)^2 + (c - c_s)^2 + 2.0}$$

For an isolated source at $(r_s, c_s)$, the peak field value is exactly $I(r_s, c_s) = \frac{I_{0, s}}{2.0}$.

### 3.2 Step Cost and Cumulative Dose
For a step from state $s$ to state $s'$ with action length $L \in \{1, \sqrt{2}\}$:
- **Absorbed Step Dose:**
  $$\Delta D = I(s') \cdot L$$
- **Cumulative Dose (arbitrary units, a.u.):**
  $$D = \sum_{t=1}^{T} I(s_t) \cdot L_t$$
- **Radiation-Aware Step Cost:**
  $$c_{\text{aware}}(s, s') = L \cdot \left(1.0 + 2.5 \cdot I(s')\right)$$

### 3.3 Reinforcement Learning Objective (Bellman Update)
Tabular Q-learning agents optimize state-action values $Q(s, a)$ over discrete state transitions:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a' \in \mathcal{A}} Q(s', a') - Q(s, a) \right]$$

where:
- Learning rate $\alpha = 0.15$, discount factor $\gamma = 0.98$
- Step reward: $R(s, a) = -L \cdot \left(1.0 + 2.5 \cdot I(s')\right)$
- Terminal Goal reward: $R_{\text{goal}} = +1000.0$
- Terminal Collision penalty: $R_{\text{collision}} = -1000.0$
- Exploration $\epsilon$ decays geometrically from $1.0$ down to $0.02$ across 50,000 episodes.

---

## 4. Key Experimental Results

All numbers reported below were measured directly from real simulation runs (`main.py`).

### 4.1 Benchmark Evaluation on 50 Fixed Start Cells
Evaluated across 50 fixed starts with free-space geodesic distance $\ge 20$ cells to the goal $(47, 47)$:

| Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Planning / Training Time * |
|---|---|---|---|---|
| **A\* Shortest** | 50/50 (100%) | 36.18 | 119.15 | 0.451 s total |
| **A\* Aware** | 50/50 (100%) | 45.04 | 54.42 | 2.082 s total |
| **Tabular Q-Learning (5 seeds)** | 49.8 ± 0.4/50 (99.6%) | 45.01 ± 0.70 | 65.18 ± 1.05 | 80.37 ± 10.70 s / seed |

*\* Note: Planning and training times were measured on a 2-core laptop CPU with jobs running one after another.*

### 4.2 Canonical Route: $(2, 2) \to (47, 47)$

| Method | Outcome | Path Steps | Cumulative Dose (a.u.) |
|---|---|---|---|
| A* Shortest | Goal | 56 | 281.88 |
| A* Aware | Goal | 80 | 58.15 |
| Q-Learning (Seed 0) | Goal | 74 | 66.10 |
| Q-Learning (Seed 1) | Goal | 75 | 68.71 |
| Q-Learning (Seed 2) | Goal | 80 | 65.81 |
| Q-Learning (Seed 3) | Goal | 80 | 64.10 |
| Q-Learning (Seed 4) | Goal | 73 | 69.34 |

### 4.3 Paired Statistical Comparison
On the 50 start pairs where both classical planners succeeded:
- **A\* Aware vs. A\* Shortest ($N=50$):**
  - Mean dose reduction: **$-64.73$ a.u. ($-54.3\%$)**
  - Mean step increase: **$+8.86$ steps ($+24.5\%$)**
- **Q-Learning (Seed 0) vs. A\* Shortest ($N=50$):**
  - Mean dose reduction: **$-53.80$ a.u. ($-45.2\%$)**
  - Mean step increase: **$+7.94$ steps ($+21.9\%$)**

### 4.4 Radiation Weight Sweep ($\lambda$) and Pareto Trade-Off
To systematically characterize the Pareto frontier between absorbed dose and path length, $\lambda$ was varied across $\{0.0, 0.5, 1.0, 2.5, 5.0, 10.0\}$ using step cost $L \cdot (1 + \lambda \cdot I)$. Results on the 50 evaluation starts and canonical $(2, 2) \to (47, 47)$ route across A* Aware and Tabular Q-Learning (3 seeds each):

| Lambda ($\lambda$) | Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Canonical Steps | Canonical Dose (a.u.) | Training Time * |
|---|---|---|---|---|---|---|---|
| **0.0** | A* Aware (Shortest) | 50/50 (100%) | 36.18 | 119.15 | 56 | 281.88 | 0.45 s total |
| 0.0 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100%) | 38.67 ± 1.06 | 207.43 ± 64.17 | 60.0 | 377.34 | 110.5 ± 3.3 s |
| **0.5** | A* Aware | 50/50 (100%) | 40.86 | 59.11 | 70 | 69.64 | 1.82 s total |
| 0.5 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100%) | 42.29 ± 0.51 | 73.27 ± 1.11 | 69.0 | 80.53 | 128.6 ± 13.8 s |
| **1.0** | A* Aware | 50/50 (100%) | 42.90 | 55.68 | 76 | 59.73 | 1.95 s total |
| 1.0 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100%) | 43.69 ± 0.63 | 69.02 ± 2.11 | 75.0 | 71.13 | 138.4 ± 10.5 s |
| **2.5** | A* Aware | 50/50 (100%) | 45.04 | 54.42 | 80 | 58.15 | 2.08 s total |
| 2.5 | Q-Learning (3 seeds) | 49.7 ± 0.5/50 (99.3%) | 45.22 ± 0.82 | 64.89 ± 0.35 | 76.3 | 66.87 | 97.4 ± 15.7 s |
| **5.0** | A* Aware | 50/50 (100%) | 46.12 | 54.16 | 82 | 57.80 | 2.15 s total |
| 5.0 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100%) | 45.97 ± 0.95 | 64.64 ± 1.73 | 75.7 | 66.78 | 95.9 ± 3.0 s |
| **10.0** | A* Aware | 50/50 (100%) | 46.78 | 54.09 | 83 | 57.70 | 2.22 s total |
| 10.0 | Q-Learning (3 seeds) | 48.3 ± 1.2/50 (96.7%) | 46.35 ± 0.77 | 63.25 ± 2.04 | 78.7 | 65.08 | 108.5 ± 7.9 s |

*\* Note: Training times were measured on a 2-core laptop CPU with jobs running one after another.*

![Pareto Trade-off](results/pareto_dose_vs_length.png)
*Pareto trade-off curve of mean absorbed radiation dose versus path length across radiation weights $\lambda \in [0, 10]$ on 50 fixed evaluation starts.*

### 4.5 On-Policy vs. Off-Policy Comparison: Tabular Q-Learning vs. SARSA ($\lambda = 2.5$)
Empirical comparison of off-policy Q-Learning versus on-policy SARSA across 3 independent training seeds ($50,000$ episodes each) on the radiation-aware navigation benchmark:

| Algorithm | Type | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Canonical Steps | Canonical Dose (a.u.) | Training Time / Seed * |
|---|---|---|---|---|---|---|---|
| **Q-Learning** | Off-policy | 49.7 ± 0.5/50 (99.3%) | 45.22 ± 0.82 | 64.89 ± 0.35 | 76.3 | 66.87 | 97.4 ± 15.7 s |
| **SARSA** | On-policy | 49.0 ± 0.8/50 (98.0%) | 46.83 ± 0.69 | 67.01 ± 1.24 | 78.3 | 75.20 | 160.6 ± 5.6 s |

- **Policy Safety Trade-off (Hypothesis)**: Because SARSA is on-policy and takes exploratory actions into account during updates, it tends to learn conservative paths with wider clearance margins around obstacle walls and radiation sources.
- **Sample Efficiency**: Both algorithms converge within 50,000 episodes on this discrete grid space. Training times were measured on a 2-core laptop CPU with jobs running one after another.

### 4.6 Exact Optimality Gap Analysis (Dijkstra Reference)
Comparison of navigation trajectories against the exact global optimum computed by Dijkstra search on 50 evaluation starts using the radiation-aware objective ($\lambda = 2.5$):

| Method | Objective λ | Success Count | Mean Optimality Gap (%) | Std Gap (%) | Median Gap (%) | Exactly Optimal Starts | Exact Share (%) |
|---|---|---|---|---|---|---|---|
| **Tabular Q-Learning (Pre-trained)** | 2.5 | 50/50 | **18.58%** | ±7.02% | 18.46% | 0/50 | **0.0%** |
| **A* Radiation-Aware (Admissible Heuristic)** | 2.5 | 50/50 | **0.00%** | ±0.00% | 0.00% | 50/50 | **100.0%** |
| **A* Shortest (Radiation-Blind Baseline)** | 2.5 | 50/50 | **70.27%** | ±68.11% | 44.09% | 0/50 | **0.0%** |

- **A* Aware Admissibility**: A* Aware achieves **0.00% gap** (100% exact match) with Dijkstra across all 50 starts, verifying that the octile distance heuristic is strictly admissible for the additive radiation cost function.
- **Q-Learning Closeness**: Tabular Q-learning achieves an average gap of **18.58%** relative to the exact global optimum without prior access to the map.
- **Radiation-Blind Penalty**: A* Shortest incurs an average **70.27% cost penalty** under the radiation-aware objective due to routing directly through intense source cores.

### 4.7 Multi-Layout Generalization Study (L1)
To test whether radiation-aware navigation generalizes across environments, 10 procedural layouts (seeds 100–109) were evaluated with 4–7 random rectangular walls (10–25% wall area) and 3 random point sources ($I_0 \in [120, 200]$), evaluated on 50 starts per layout ($N = 500$ evaluation pairs). In addition to classical search, Tabular Q-Learning (50,000 episodes, $\lambda = 2.5$) was trained on each layout:

| Layout Seed | Wall Frac | Q-Learning Success | Q-Learning Steps | Q-Learning Dose (a.u.) | A* Aware Success | A* Aware Steps | A* Aware Dose (a.u.) | Dose Gap (a.u.) | Dose Gap (%) | Q Train Time * |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 17.0% | 49/50 (98%) | 43.59 | 64.31 | 50/50 | 41.84 | 54.61 | +9.69 | +17.7% | 112.8s |
| 101 | 14.1% | 46/50 (92%) | 72.24 | 152.90 | 50/50 | 57.44 | 138.34 | +14.55 | +10.5% | 175.7s |
| 102 | 11.7% | 49/50 (98%) | 38.04 | 45.70 | 50/50 | 45.06 | 33.89 | +11.81 | +34.9% | 105.7s |
| 103 | 14.5% | 50/50 (100%) | 49.04 | 45.13 | 50/50 | 47.00 | 35.29 | +9.84 | +27.9% | 80.1s |
| 104 | 13.0% | 50/50 (100%) | 44.48 | 64.35 | 50/50 | 43.68 | 53.56 | +10.79 | +20.1% | 112.2s |
| 105 | 17.7% | 50/50 (100%) | 46.30 | 83.27 | 50/50 | 47.80 | 67.71 | +15.56 | +23.0% | 114.0s |
| 106 | 15.7% | 50/50 (100%) | 48.20 | 95.40 | 50/50 | 49.44 | 67.85 | +27.54 | +40.6% | 109.1s |
| 107 | 11.8% | 50/50 (100%) | 39.74 | 38.84 | 50/50 | 37.00 | 31.14 | +7.70 | +24.7% | 90.8s |
| 108 | 18.8% | 50/50 (100%) | 42.50 | 98.95 | 50/50 | 39.46 | 86.55 | +12.41 | +14.3% | 107.4s |
| 109 | 13.0% | 50/50 (100%) | 62.88 | 54.69 | 50/50 | 65.54 | 42.28 | +12.42 | +29.4% | 138.6s |
| **AVERAGE** | **14.7%** | **98.8%** | **48.70** | **74.35** | **100.0%** | **47.43** | **61.12** | **+13.23** | **+24.3%** | **114.7s** |

*\* Note: Training times were measured on a 2-core laptop CPU with jobs running one after another.*

- **A* Aware Pooled Benchmark**: Across all $N=500$ evaluation pairs, A* Aware achieved a pooled dose reduction of **57.8%** (mean difference: **-83.63 a.u.**, 95% bootstrap CI: `[-94.53, -73.18]`) relative to A* Shortest.
- **Q-Learning Generalization**: Tabular Q-learning achieved an average evaluation success rate of **98.8%** across all 10 procedural layouts with a mean dose gap of **+13.23 a.u. (+24.3%)** relative to optimal A* Aware.

![Layouts Overview](results/layouts_overview.png)
*First 4 procedural layouts (seeds 100–103) with radiation heatmap, wall obstacles, and planned trajectories from start (2,2) to goal (47,47).*

### 4.8 Map-Error Robustness Study (L2)
Robustness of A* Aware planning ($\lambda = 2.5$) when the prior radiation map has errors in source position (shifts $\Delta_{\text{pos}} \in \{1, 2, 3\}$ cells) and source intensity ($\pm 20\%$). Paths planned on perturbed fields were evaluated on the true radiation field across 50 starts (10 random perturbations per level):

| Perturbation Level | Max Pos Shift | Intensity Scale | Mean True Dose (a.u.) | Dose Penalty vs Oracle | Dose Savings vs Shortest |
|---|---|---|---|---|---|
| Oracle (Ground Truth) | 0 cells | 1.00 (exact) | 54.42 | 0.00 (ref) | 54.3% |
| Level 1 | ±1 cell | [0.8, 1.2] | 54.56 ± 0.23 | +0.14 a.u. (+0.3%) | 54.2% |
| Level 2 | ±2 cells | [0.8, 1.2] | 55.85 ± 1.08 | +1.43 a.u. (+2.6%) | 53.1% |
| Level 3 | ±3 cells | [0.8, 1.2] | 59.31 ± 3.18 | +4.89 a.u. (+9.0%) | 50.2% |
| Baseline (Shortest) | N/A | N/A | 119.15 | +64.73 a.u. (+118.9%) | 0.0% (ref) |

![Map Error Robustness](results/robustness.png)
*True absorbed dose on ground truth radiation field across perturbation levels (10 runs per level with mean ± 1 std), compared against Oracle Aware and Shortest baselines.*

---

## 5. Visualizations

### Trajectory and Dose Profile Comparison
![Trajectory Comparison](results/trajectory_comparison.png)
*Left: Radiation field heatmap (`inferno`) with structural obstacle walls (grey) and agent trajectories. Right: Cumulative dose accumulation as a function of path progression.*

### Reinforcement Learning Convergence
![RL Convergence](results/rl_convergence.png)
*Moving average (window = 1000) of episode rewards and absorbed dose across 50,000 training episodes for Seed 0 and the 5-seed ensemble average.*

### Greedy Success Rate Convergence
![Greedy Success Rate](results/convergence_greedy_success.png)
*Periodic greedy evaluation ($N=50$ starts, $\epsilon=0$) every 2,500 training episodes for Tabular Q-learning ($\lambda=2.5$, Seed 0). The greedy policy reaches 100% navigation success by episode 22,500 and remains stable.*

---

## 6. Scientific Observations & Hypotheses

> [!NOTE]
> Explanations of observed performance are scientific hypotheses and should not be construed as definitive proof.

1. **Trade-off Mechanism (Hypothesis):** The A* Shortest path navigates between the central obstacle and source hotspots $(18,20)$ and $(32,30)$, experiencing peak field intensities up to $\sim 70-100$ a.u. Radiation-Aware A* adopts an outer detour along lower-gradient perimeter corridors, trading a 24.5% step overhead for a 54.3% dose reduction.
2. **Q-Learning Detour Discovery (Hypothesis):** Despite having no prior map and relying entirely on scalar rewards, the Tabular Q-Learning agent successfully converges to a low-radiation perimeter policy. Its mean path length (45.01 steps) closely mirrors A* Aware (45.04 steps), with absorbed dose (65.18 a.u.) within 20% of the full-information optimal planner.

---

## 7. Limitations

1. **Toy 2D Simulation:** The environment is an idealized 50x50 grid; it does not model continuous robot dynamics, wheel slippage, or 3D terrain.
2. **Simplified Physics:** Radiation is modeled as static, softened inverse-square point sources without scattering, material-dependent wall shielding/attenuation, or time-varying decay.
3. **Tabular State Space:** The Q-learning agent uses tabular state indices $(r, c)$ for a single goal and memorizes optimal paths for this specific static map; it possesses no onboard perception sensors or generalizability to unseen layouts.
4. **Deterministic Sensors:** Perception and movement transitions are noise-free.

---

## 8. How to Run

### Installation
Clone the repository and run with Python 3.10+ (standard library, numpy, matplotlib, pillow):
```powershell
git clone https://github.com/muhammad-umair-aideveloper/radiation-aware-robot-navigation.git
cd radiation-aware-robot-navigation

# Run unit tests
python tests/test_core.py

# Run full evaluation benchmark and regenerate plots
python main.py
```

### Quick Inference (Pre-trained Weights)
Load the best saved Q-table and run a zero-training-time greedy rollout from `(2,2)` to `(47,47)`:
```powershell
python evaluate.py
```

---

## 9. License
MIT License. See [LICENSE](LICENSE) for details.

---

## 10. Development Notes

This project was developed with AI coding assistance (Antigravity / Gemini). The experiment specification, parameter values, and evaluation protocol were provided by the author; code implementation, testing, and result verification were carried out by the AI assistant.
