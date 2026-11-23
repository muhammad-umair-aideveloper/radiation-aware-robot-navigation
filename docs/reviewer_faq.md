# Reviewer FAQ

This document addresses common technical and methodological questions that a scholarship
committee or technical reviewer may ask about the Radiation-Aware Robot Navigation project.
All responses are grounded in existing result files and pre-declared experimental protocols.

## Questions and Answers

### 1. What is the main result of this project and how strong is the evidence?

Across multiple studies, radiation-aware graph search (A* aware) reliably identifies lower-dose
corridors around hotspots compared to shortest-path baselines, reducing true path dose substantially
with modest increases in path length. In procedural fresh-layout replications across 20 layouts (S13),
radiation-aware planning with true field access saved a mean of 52.959% dose versus shortest paths,
demonstrating that spatial radiation avoidance is structurally robust across diverse wall and source
configurations.

Evidence: [Technical Report](../TECHNICAL_REPORT.md), [S13 replication summary](../results/s13_summary.md), and [Sweep summary](../results/sweep_summary.json).

### 2. How does radiation-aware planning compare to shortest-path navigation in the primary layout?

In the fixed primary layout (S1–S2), varying the radiation penalty weight $\lambda$ establishes a clear
trade-off curve where increasing $\lambda$ lowers accumulated dose at the cost of modest path detour steps.
At the benchmark weight $\lambda = 2.5$, A* aware reduces mean dose from 214.48 arbitrary units (shortest path)
to 59.79 arbitrary units on fixed starts.

Evidence: [Sweep summary](../results/sweep_summary.md) and [Pareto plot](../results/pareto_dose_vs_length.png).

### 3. What is the relationship between the tabular RL policies and the exact optimal planner?

Dijkstra dynamic programming acts as an exact reference solver, confirming zero optimality gap for
A* aware across benchmark starts. Tabular Q-learning and SARSA successfully converge to near-optimal
policies on the fixed layout, but require extensive episode training and lack zero-shot transferability
to unseen layouts.

Evidence: [Optimality gap study](../results/optimality_gap.md) and [DP reference implementation](../src/dp_reference.py).

### 4. Why did the sensor-based DQN agent fail on held-out goal locations (S5–S6)?

The map-free DQN agent trained with 16-ray LiDAR and local Geiger sensors achieved 0% to 1% success on
held-out goal pairs, suffering from 88% to 100% timeouts. Diagnostic analysis revealed that the agent
failed to make net progress toward distant goals, trapped in local revisit loops rather than colliding
with obstacles.

Evidence: [Sensor diagnosis](../results/sensor_diagnosis.md), [Sensor agent documentation](sensor_agent.md), and [S6 selection](../results/sensor_s6_selection.md).

### 5. Did pre-declared improvements (frame stacking, curriculum, goal pool) help the sensor agent?

No. In study S6, four pre-declared structural variants (frame stacking, distance curriculum, combined
features, and enlarged training goal pool) were trained for 400000 steps each and evaluated against a
pre-declared 10% validation success gate. All four variants achieved 0.0% validation success, triggering
the pre-declared rule to halt Stage B training and report the negative result honestly.

Evidence: [S6 selection summary](../results/sensor_s6_selection.md) and [EXPERIMENTS.md](../EXPERIMENTS.md).

### 6. Is a pre-goal exploration survey worth its radiation dose (S12)?

No; in study S12, allocating an exploratory survey budget ($B = 30$ or $B = 60$ steps) before navigating
to the goal consistently increased total cumulative dose compared to direct goal-directed replanning
($B = 0$). While pre-goal surveys increased the count of distinct measured cells, the extra radiation
absorbed during the survey exceeded the downstream path-planning savings.

Evidence: [S12 survey summary](../results/s12_summary.md) and [Field estimation documentation](field_estimation.md).

### 7. How well do spatial field estimators extrapolate beyond visited trajectories (S7–S10)?

Non-parametric spatial estimators (kernel regression and inverse distance weighting) fit sampled path
points accurately but extrapolate poorly to unvisited grid regions, frequently underestimating distant
hotspots and leading online replanners into high-radiation zones. In contrast, the parametric OMP3
estimator leverages physical inverse-square modeling to infer global hotspot locations from sparse
samples, reducing dose regret substantially in offline benchmarks.

Evidence: [Field estimation documentation](field_estimation.md) and [Parametric study](../results/param_summary.md).

### 8. Does wall shielding affect radiation field estimation and planning (S11)?

Yes; wall attenuation creates sharp radiation shadows that violate unshielded line-of-sight assumptions.
Study S11 demonstrated that when walls attenuate radiation, unshielded estimators suffer elevated regret,
whereas shielding-aware models (OMP3S) that incorporate straight-line wall transmission recover accurate
field geometry.

Evidence: [Shielding study summary](../results/shield_summary.md) and [Field estimation documentation](field_estimation.md).

### 9. Did the physics-informed estimator (OMP3) beat kernel regression in the fresh-layout replication (S13)?

In the pre-declared confirmatory protocol across 20 unseen layouts, OMP3 beat kernel online in 14 of 20
layouts, which fell one layout short of the pre-declared threshold of 15 layouts; therefore, hypothesis H1
was formally not supported. However, kernel online regret exceeded 50% in 16 of 20 layouts (supporting H3),
confirming that unguided spatial regression repeatedly fails during online replanning on fresh layouts.

Evidence: [S13 replication summary](../results/s13_summary.md) and [EXPERIMENTS.md](../EXPERIMENTS.md).

### 10. What does the supplementary post-hoc analysis show for S13?

A supplementary post-hoc analysis of paired layout differences showed that OMP3 had lower online regret
than kernel with a paired difference mean of 25.038% and a 95% bootstrap interval of [9.110, 44.686]%.
The exact two-sided binomial sign test yielded $p = 0.0118$ offline (16/20 wins, 0 ties) and $p = 0.1153$
online (14/20 wins, 0 ties). Because this analysis was decided after observing results, it is strictly
descriptive and does not alter the formal "not supported" verdict for H1.

Evidence: [S13 summary](../results/s13_summary.md) and [Design decisions](design_decisions.md).

### 11. What parts of the problem are physically simulated versus idealized toy assumptions?

Radiation attenuation strictly obeys physical inverse-square geometry with an explicit softening
parameter, Geiger counters sample Poisson-distributed arrival counts governed by dwell time, and walls
produce exponential line-of-sight shielding. Conversely, the environment assumes idealized 2D grid
movement, zero actuation noise, zero measurement time or dose cost, and no radiation scattering or
secondary emissions.

Evidence: [Technical Report](../TECHNICAL_REPORT.md) and [Radiation physics module](../src/radiation_physics.py).

### 12. What aspects of the current findings do not generalize?

Findings do not directly generalize to continuous 3D environments, mobile robots with non-holonomic
kinematics, complex scattering environments, or real-time dynamics with moving radiation sources.
Furthermore, the strong performance of OMP3 relies on assuming that the number of point sources and the
algebraic inverse-square form are known a priori.

Evidence: [Technical Report](../TECHNICAL_REPORT.md) and [GLOSSARY.md](../GLOSSARY.md).

### 13. How were experimental pre-declarations and negative controls enforced throughout the project?

All research questions, parameter grids, evaluation seeds, hypothesis thresholds, and termination rules
were committed to EXPERIMENTS.md prior to running experiments. Pre-study negative controls (such as
true-field consistency and zero-field equivalence) were executed and verified via automated test suites
before experimental data generation was permitted.

Evidence: [EXPERIMENTS.md](../EXPERIMENTS.md) and [S13 study controls](../tests/test_s13_study.py).

### 14. What was the role of AI coding assistance and what did the human author decide and verify?

AI coding assistance was used to implement, refactor, and document code. The author reviews generated
diffs, chooses conventional commit boundaries, verifies tests and exact result values, and decides
whether scientific claims are supported. No generated result file was edited or deleted during
consolidation.

Evidence: [Technical Report](../TECHNICAL_REPORT.md) and [Commit history](../CHANGELOG.md).

### 15. How can an external reviewer independently reproduce the results and verify code quality?

Reviewers can execute the automated quality gate (`scripts/quality_gate.ps1`), which runs Ruff linting,
the full pytest suite (including golden reference checks), the numeric claim audit (`scripts/audit_numbers.py`),
and Markdown link validation (`scripts/check_links.py`). Individual experiment runner scripts and resulting
data files are comprehensively cataloged in RESULTS_INDEX.md.

Evidence: [Quality gate](../scripts/quality_gate.ps1), [Documentation Index](../DOCUMENTATION_INDEX.md), and [RESULTS_INDEX.md](../RESULTS_INDEX.md).
