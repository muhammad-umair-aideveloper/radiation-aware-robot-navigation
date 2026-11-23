# Radiation-Aware Robot Navigation: Technical Report

## Abstract

This technical report synthesizes numerical simulation studies of radiation-aware
navigation on a 50x50 grid. It compares shortest-path and radiation-aware A*,
tabular reinforcement learning, sensor-based learning, field estimators, online
replanning, shielding, and surveys. The model is toy physics; it is not a
facility-safety certificate.

## Motivation and Physics Context

Particle accelerator facilities, beamlines, target stations, and activated
equipment can create spatially non-uniform radiation fields. A robot inspection
platform should reduce total exposure while maintaining route completion.
The ALARA principle means keeping radiation exposure as low as reasonably
achievable. These studies use arbitrary dose units, simplified inverse-square
fields, and idealized sensing; they do not represent regulatory dosimetry.

## Environment and Radiation Model

The environment is an 8-connected 50x50 grid with structural obstacles and point
radiation sources. Inverse-square intensity is softened to avoid singularities.
A trajectory dose is the sum of source-field intensity times step length.
Dose and shielding are reported in arbitrary units, not millisieverts.

| Quantity | Model meaning |
|---|---|
| Grid | 50x50 cells with walls and free space |
| Motion | 8-connected steps with corner-cutting constraints |
| Radiation field | Sum of softened inverse-square source fields |
| Dose | Sum of intensity times step length in a.u. |

## Methods

A* was used with shortest and radiation-aware objectives. Tabular Q-learning and
SARSA learned from episodic rewards on the same grid. Sensor agents used LiDAR
and local Geiger observations. Offline estimators used kernel regression,
inverse-distance weighting, and parametric source fitting. Online agents sampled
along the route, re-estimated, and replanned. The survey study added a fixed tour
before goal-directed replanning. Shield-aware fitting incorporated known walls
and attenuation.

## Experimental Protocol

Each study used pre-declared controls, fixed seeds, fixed start sets, and paired
comparisons on mutually successful episodes. Where bootstrap intervals are shown
they are two-sided 95% intervals from 10,000 resamples with fixed random seeds.
Failed episodes were not inserted into dose averages. S12 used an additional
control that compared B=0 execution against earlier S8 and S10 results.

## Results

### S1: Radiation Weight and Classical/RL Benchmark

| Metric | A* aware lambda 2.5 | A* shortest reference |
|---|---:|---:|
| Mean dose (a.u.) | 54.42 | 119.15 |
| Mean steps | 45.04 | 36.18 |

Key finding: radiation-aware A* reduced the measured mean dose at the cost of
additional mean steps. The paired A* aware/shortest dose difference was
-64.73 a.u.

### S2: Multi-Layout Generalization

| Metric | Pooled result |
|---|---:|
| Mean dose difference (a.u.) | -83.63 |
| Bootstrap low | -94.53 |
| Bootstrap high | -73.18 |

Key finding: the paired pooled dose difference remained negative across layouts.

### S3: Map-Error Robustness

| Reference | Mean true dose (a.u.) |
|---|---:|
| Oracle | 54.42 |
| Perturbation level 2 | 55.85 |
| Shortest | 119.15 |

Key finding: stronger map perturbations increased executed dose in the reported
one-layout robustness run.

### S4: Exact Optimality Gap

| Method | Mean gap (%) | Exact starts |
|---|---:|---:|
| A* aware | 0.00 | 50/50 |
| Tabular Q-learning | 18.58 | 0/50 |
| A* shortest | 70.27 | 0/50 |

Key finding: A* aware matched the Dijkstra consistency reference exactly.

### S5: Sensor-Based DQN

Key finding: the map-free DQN variants did not provide the held-out navigation
success of classical graph search; the final guarded test set showed near-zero
success, overwhelmingly from timeouts.

| Baseline | Test success | Success count |
|---|---:|---:|
| A* shortest | 100% | 100/100 |
| A* aware | 100% | 100/100 |

### S6: Sensor-Agent Improvements

Key finding: the pre-declared improvements did not exceed the 10% validation
usability threshold. Frame stacking was the selected negative-result comparison
only because it had the lowest timeout count among variants.

### S7: Offline Measure-Estimate-Plan

Key finding: offline measurement density and dwell time affected estimator dose
regret. The existing estimation summary records 27 estimator/sample/dwell results
plus a cross-estimator summary.

### S8: Online Measure-Estimate-Replan

| Reference / configuration | Mean dose (a.u.) | Dose reduction (%) |
|---|---:|---:|
| Oracle | 56.22 | 48.03 |
| A* shortest | 108.19 | 0.00 |
| Kernel, 3x3, dwell 1 | 106.37 | 1.68 |

Key finding: online local sensing retained only part of oracle dose saving.

### S9: Sensor Range and Unexplored-Cell Inflation

| Diagnostic | Result |
|---|---:|
| True/zero controls | 0.0 / 0.0 max difference |
| Kernel disc6 noise-free dose regret (%) | 6.55 |
| Kernel disc6 noise-free distinct cells | 489.48 |

Key finding: larger sensor patterns improved noise-free coverage and reduced regret
in the reported range study.

### S10: Parametric Field Estimation

| Metric | OMP3 online 3x3 dwell 1 |
|---|---:|
| Dose regret (%) | 30.81 |

Key finding: the known-source parametric model was tested against kernel and IDW
and its strong assumptions must be read alongside its best-case source fit.

### S11: Wall Shielding

| Metric | Shielded result at mu 2.0 |
|---|---:|
| Oracle dose (a.u.) | 37.91 |
| Wall-blind regret (%) | 3.45 |
| OMP3s N50 dwell 10 regret (%) | 1.52 |

Key finding: shield-aware OMP3S reduced the tested wall-shielded planning regret.

### S12: Survey Dose

| Estimator/dwell/B | Total dose (a.u.) | Total steps |
|---|---:|---:|
| kernel, dwell 10, B 0 | 93.66 | 36.14 |
| kernel, dwell 10, B 30 | 242.58 | 84.66 |
| kernel, dwell 10, B 60 | 260.52 | 112.27 |
| kernel, dwell 1, B 0 | 106.37 | 37.89 |
| kernel, dwell 1, B 30 | 256.98 | 86.98 |
| kernel, dwell 1, B 60 | 292.37 | 115.90 |
| omp3, dwell 10, B 0 | 63.31 | 41.69 |
| omp3, dwell 10, B 30 | 179.08 | 92.05 |
| omp3, dwell 10, B 60 | 235.59 | 120.70 |
| omp3, dwell 1, B 0 | 73.55 | 42.64 |
| omp3, dwell 1, B 30 | 182.15 | 91.22 |
| omp3, dwell 1, B 60 | 248.61 | 121.82 |

Key finding: fixed surveys increased total steps and dose for all 12
reported B=30 and B=60 cases; B=0 remained the lower-dose surveyed comparison.

### S13: Fresh-Layout Online Replication

| Metric | Pooled mean [95% layout interval] |
|---|---:|
| Oracle saving versus shortest (%) | 52.96 [45.83, 59.55] |
| Kernel online dose regret (%) | 81.91 [62.18, 105.17] |
| OMP3 online dose regret (%) | 56.87 [46.73, 67.20] |

| Paired threshold | Layout count |
|---|---:|
| OMP3 beats kernel offline | 16/20 |
| OMP3 beats kernel online | 14/20 |
| Oracle saving at least 10% | 19/20 |
| Kernel online regret over 50% | 16/20 |

| Hypothesis | Verdict |
|---|---|
| H1 | not supported |
| H2 | supported |
| H3 | supported |

Supplementary post-hoc analysis (decided after S13 results were seen; descriptive, p-values unadjusted):

| Mode | OMP3 wins vs kernel | Ties | Exact p-value | Paired regret diff mean [%] | 95% bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Offline | 16/20 | 0 | 0.0118 | 18.30 | [5.17, 32.99] |
| Online | 14/20 | 0 | 0.1153 | 25.04 | [9.11, 44.69] |

Key finding: the pooled online regret intervals for kernel and OMP3 overlap;
H1 is not supported by the 14/20 paired layout count. H2 and H3 are supported.
The supplementary paired difference interval excludes zero [9.11, 44.69]%, but the
exact sign test (p = 0.1153) and layout count remain below confirmatory thresholds.
Detailed limits and layout values are in the S13 documentation and summary.

## Negative and Null Results

- Sensor DQN held-out success was effectively zero despite low collision rates.
- S6 variants did not pass the pre-declared 10% validation gate.
- S12 surveys did not lower total measured dose in the reported B=30 and B=60 cases.
- The frame-stacking, curriculum, combined, and goal-pool changes therefore provide
  negative rather than positive deployment evidence.

## Threats to Validity and Limitations

The grid is synthetic and one-layout effects can dominate estimator performance.
Movement is noise-free, wall geometry is known, and readings incur no travel time
or dose. Sensor studies use a fixed 300-step guard and limited seeds. Shielding
uses straight-line wall attenuation without scattering. Bootstrap intervals describe
seed resampling uncertainty, not all physical or implementation uncertainty.

## Verification and Development Process

The project is implemented and maintained with plain-assert pytest tests, golden
reference recomputation tests, Ruff lint, coverage measurement, a number audit,
and a documentation-link check. The CI workflow runs the Python version matrix.
AI coding assistance was used to implement, refactor, and document code. The author
reviews generated diffs, chooses conventional commit boundaries, verifies tests
and exact result values, and decides whether scientific claims are supported.
No generated result file was edited or deleted during consolidation.

## Reproducibility

Source files and result files are indexed in RESULTS_INDEX.md. The quality gate
runs lint, tests, golden references, number audit, and link checking. Generated
tables in this report are produced by this report script, not transcribed manually.
Runtime varies by machine and the test suite includes a resumable S12 control.

## Future Work

Future work could add uncertainty-aware plans, repeated layouts, physical dose
units, movement and measurement costs, scattering-aware shielding, information
gain for surveys, and memory-based sensor agents. It should predeclare stopping
rules and preserve the existing negative-result discipline.

## Artifact References

- [Experimental protocol](EXPERIMENTS.md)
- [Results index](RESULTS_INDEX.md)
- [Field estimation documentation](docs/field_estimation.md)
- [Sensor-agent documentation](docs/sensor_agent.md)
