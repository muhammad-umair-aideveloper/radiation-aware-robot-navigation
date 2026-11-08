# Radiation-Aware Robot Navigation: Path Planning & Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-numpy%20%7C%20matplotlib%20%7C%20pillow-green.svg)](#)

A simulation study comparing classical graph search (shortest vs. radiation-aware A*) and Tabular Q-Learning for autonomous mobile robot navigation across a 2D environment containing radiation hotspots and structural obstacles.

![Radiation Navigation Animation](results/radiation_navigation.gif)

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

| Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Planning / Training Time |
|---|---|---|---|---|
| **A\* Shortest** | 50/50 (100%) | 36.18 | 119.15 | 0.451 s total |
| **A\* Aware** | 50/50 (100%) | 45.04 | 54.42 | 2.082 s total |
| **Tabular Q-Learning (5 seeds)** | 49.8 ± 0.4/50 (99.6%) | 45.01 ± 0.70 | 65.18 ± 1.05 | 80.37 ± 10.70 s / seed |

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

---

## 5. Visualizations

### Trajectory and Dose Profile Comparison
![Trajectory Comparison](results/trajectory_comparison.png)
*Left: Radiation field heatmap (`inferno`) with structural obstacle walls (grey) and agent trajectories. Right: Cumulative dose accumulation as a function of path progression.*

### Reinforcement Learning Convergence
![RL Convergence](results/rl_convergence.png)
*Moving average (window = 1000) of episode rewards and absorbed dose across 50,000 training episodes for Seed 0 and the 5-seed ensemble average.*

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
