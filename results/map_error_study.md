# Map-Error Robustness Study Results (L2)

Evaluation of **A* Aware** ($\lambda = 2.5$) planning under imperfect radiation field estimates on the canonical layout. Paths planned on perturbed source fields are evaluated on the true radiation field across 50 starts ($10$ runs per perturbation level).

### Baseline References (Ground Truth)
- **A* Shortest**: Mean dose = **119.15 a.u.**, Path length = **36.18 steps**
- **Oracle A* Aware**: Mean dose = **54.42 a.u.**, Path length = **45.04 steps**

### Robustness Across Perturbation Levels

| Perturbation Level (Δ_pos) | Max Shift | Intensity Error | Mean True Dose (a.u.) | Std Dose | Mean Steps | Overhead vs Oracle (a.u.) | Dose Savings vs Shortest (%) |
|---|---|---|---|---|---|---|---|
| Level 1 | ±1 cells | ±20% | 54.56 | ±0.23 | 45.35 | +0.14 (+0.3%) | **54.2%** |
| Level 2 | ±2 cells | ±20% | 55.85 | ±1.08 | 45.04 | +1.43 (+2.6%) | **53.1%** |
| Level 3 | ±3 cells | ±20% | 59.31 | ±3.18 | 45.52 | +4.89 (+9.0%) | **50.2%** |

### Scientific Insights (Hypothesis)
- **Graceful Degradation**: Even under significant mapping errors (Level 3: source positions shifted by up to 3 cells and intensity scaled by ±20%), the radiation-aware planner preserves over **40% dose reduction** relative to the radiation-blind shortest path.
- **Safety Detour Persistence**: Because obstacles and boundary corridors naturally force navigation outside the core high-intensity zone, small localized shifts in source centroids do not destroy the global topology of the detour.
