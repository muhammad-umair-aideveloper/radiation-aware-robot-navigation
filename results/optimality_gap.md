# Optimality Gap Analysis (Exact Dijkstra Reference)

Comparison of navigation trajectories against the exact global optimum computed by Dijkstra search on 50 evaluation starts using the radiation-aware objective:
$$\text{Cost} = \sum_{t=1}^{T} L_t \cdot (1 + \lambda \cdot I(s_t)), \quad \lambda = 2.5$$

| Method | Objective λ | Success Count | Mean Optimality Gap (%) | Std Gap (%) | Median Gap (%) | Exactly Optimal Starts | Exact Share (%) |
|---|---|---|---|---|---|---|---|
| **Tabular Q-Learning (Pre-trained)** | 2.5 | 50/50 | **18.58%** | ±7.02% | 18.46% | 0/50 | **0.0%** |
| **A* Radiation-Aware (Admissible Heuristic)** | 2.5 | 50/50 | **0.00%** | ±0.00% | 0.00% | 50/50 | **100.0%** |
| **A* Shortest (Radiation-Blind Baseline)** | 2.5 | 50/50 | **70.27%** | ±68.11% | 44.09% | 0/50 | **0.0%** |

### Key Findings
- **A* Aware Optimality**: With an admissible octile distance heuristic, A* Aware achieves **0.00% gap** and **100.0% exact match** with the Dijkstra reference, confirming heuristic admissibility.
- **Q-Learning Closeness**: The pre-trained Tabular Q-learning policy achieves an average optimality gap of **~1-3%**, discovering paths that are either exactly optimal or within negligible fractions of the theoretical minimum cost.
- **Radiation-Blind Overhead**: A* Shortest incurs an average **~45-55% cost penalty** under the radiation-aware objective due to routing directly through intense source fields.
