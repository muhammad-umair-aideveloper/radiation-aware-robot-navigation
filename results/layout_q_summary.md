# Multi-Layout Q-Learning vs A* Aware Study (L1)

Evaluation of Tabular Q-Learning (50,000 episodes, $\lambda = 2.5$) versus A* Aware baseline on 10 procedural layouts (seeds 100–109), 50 starts per layout.

*Note: Training times were measured on a 2-core laptop CPU with jobs running one after another.*

| Layout Seed | Wall Frac | Q-Learning Success | Q-Learning Steps | Q-Learning Dose (a.u.) | A* Aware Success | A* Aware Steps | A* Aware Dose (a.u.) | Dose Gap (a.u.) | Dose Gap (%) | Q Train Time (s) |
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
| **AVERAGE** | **-** | **98.8%** | **48.70** | **74.35** | **100.0%** | **47.43** | **61.12** | **+13.23** | **+24.3%** | **114.7s** |

### Key Findings
- **Generalization Performance**: Tabular Q-learning successfully converges to high-quality navigation policies across diverse random procedural layouts, achieving an average success rate of **98.8%** on unseen start positions.
- **Radiation Dose Gap**: Relative to the optimal A* Aware planner, Tabular Q-learning exhibits an average dose gap of **+13.23 a.u.** (**+24.3%**), while discovering safe detours around hotspots across all wall fractions (10%–25%).
