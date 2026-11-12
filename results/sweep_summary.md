# Radiation Weight Sweep Summary (A* vs. Tabular Q-Learning)

Aggregated benchmark results across radiation weights $\lambda \in \{0.0, 0.5, 1.0, 2.5, 5.0, 10.0\}$.
Evaluated on the fixed 50 evaluation starts and canonical route $(2, 2) \to (47, 47)$.
Q-learning metrics are reported as mean $\pm$ std over 3 independent training seeds ($50,000$ episodes each).

| Lambda (λ) | Method | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Canonical Steps | Canonical Dose (a.u.) | Training / Planning Time |
|---|---|---|---|---|---|---|---|
| 0.0 | A* Aware | 50/50 (100%) | 36.18 | 119.15 | 56 | 281.88 | deterministic planner |
| 0.0 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100.0%) | 38.67 ± 1.06 | 207.43 ± 64.17 | 60.0 | 377.34 | 110.5 ± 3.3 s |
| 0.5 | A* Aware | 50/50 (100%) | 40.86 | 59.11 | 70 | 69.64 | deterministic planner |
| 0.5 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100.0%) | 42.29 ± 0.51 | 73.27 ± 1.11 | 69.0 | 80.53 | 128.6 ± 13.8 s |
| 1.0 | A* Aware | 50/50 (100%) | 42.90 | 55.68 | 76 | 59.73 | deterministic planner |
| 1.0 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100.0%) | 43.69 ± 0.63 | 69.02 ± 2.11 | 75.0 | 71.13 | 138.4 ± 10.5 s |
| 2.5 | A* Aware | 50/50 (100%) | 45.04 | 54.42 | 80 | 58.15 | deterministic planner |
| 2.5 | Q-Learning (3 seeds) | 49.7 ± 0.5/50 (99.3%) | 45.22 ± 0.82 | 64.89 ± 0.35 | 76.3 | 66.87 | 97.4 ± 15.7 s |
| 5.0 | A* Aware | 50/50 (100%) | 46.12 | 54.16 | 82 | 57.80 | deterministic planner |
| 5.0 | Q-Learning (3 seeds) | 50.0 ± 0.0/50 (100.0%) | 45.97 ± 0.95 | 64.64 ± 1.73 | 75.7 | 66.78 | 95.9 ± 3.0 s |
| 10.0 | A* Aware | 50/50 (100%) | 46.78 | 54.09 | 83 | 57.70 | deterministic planner |
| 10.0 | Q-Learning (3 seeds) | 48.3 ± 1.2/50 (96.7%) | 46.35 ± 0.77 | 63.25 ± 2.04 | 78.7 | 65.08 | 108.5 ± 7.9 s |

### Unfinished or Failed Runs
- Q-learning λ=2.5 seed=1: success=49/50
- Q-learning λ=10.0 seed=0: success=47/50
- Q-learning λ=10.0 seed=1: success=48/50
- SARSA runs not executed (--sarsa option was not present in sweep_lambda.py)
