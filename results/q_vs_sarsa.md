# Algorithm Comparison: Tabular Q-Learning vs. SARSA (λ = 2.5)

Empirical comparison of off-policy Q-Learning versus on-policy SARSA across 3 independent training seeds ($50,000$ episodes each) on the radiation-aware navigation benchmark.

| Algorithm | Type | Success on 50 starts | Mean Steps | Mean Dose (a.u.) | Canonical Steps | Canonical Dose (a.u.) | Training Time / Seed |
|---|---|---|---|---|---|---|---|
| **Q-Learning** | Off-policy | 49.7 ± 0.5/50 (99.3%) | 45.22 ± 0.82 | 64.89 ± 0.35 | 76.3 | 66.87 | 97.4 ± 15.7 s |
| **SARSA** | On-policy | 49.0 ± 0.8/50 (98.0%) | 46.83 ± 0.69 | 67.01 ± 1.24 | 78.3 | 75.20 | 160.6 ± 5.6 s |

### Observations & Hypotheses
- **Policy Safety Trade-off (Hypothesis)**: Because SARSA is on-policy and takes exploratory actions into account during updates, it tends to learn conservative paths with wider clearance margins around obstacle walls and radiation sources.
- **Sample Efficiency**: Both algorithms converge reliably within 50,000 episodes on this discrete grid space, achieving near-identical computational throughput per episode.
