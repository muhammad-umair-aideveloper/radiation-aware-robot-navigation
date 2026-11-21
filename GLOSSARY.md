# Reviewer Glossary

This glossary defines terms as used in this repository. It contains no measured
claims or study counts.

## A* Aware

A graph-search planner that minimizes a step objective combining path length and
radiation cost. In this project it receives the estimated or true radiation field
available to that study. See [Technical Report](TECHNICAL_REPORT.md).

## A* Shortest

The radiation-blind baseline A* objective. It minimizes geometric step length only
and is used to quantify the trade-off against radiation-aware planning.

## Arbitrary Units (a.u.)

The project’s simulated dose scale. A dose in a.u. is a sum of field intensity times
step length; it is not mSv, Sv, or any calibrated physical dose.

## ALARA

As Low As Reasonably Achievable. The project uses ALARA as the practical motivation
for limiting dose, while the environment remains a simplified toy model.

## Bootstrap Interval

A percentile-style uncertainty interval produced by resampling study seed means or
paired observations and re-computing the statistic. The project reports fixed-seed
two-sided intervals rather than treating them as physical confidence guarantees.

## Dwell Time

The counter sensitivity parameter for a simulated Geiger reading. Counts are sampled
as Poisson observations proportional to dwell time and true local intensity, and
the measured intensity is counts divided by dwell time.

## Dose Reduction

The percentage reduction in mean true dose relative to a specified baseline:
`(baseline dose - method dose) / baseline dose * 100`. Positive values favor the
method; negative values indicate worse dose than the baseline.

## Dose Regret

The percentage excess in mean true dose relative to an oracle:
`(method dose - oracle dose) / oracle dose * 100`. Lower regret is better.

## DQN

A neural-network deep Q-learning agent implemented in NumPy. In sensor studies, it
maps local LiDAR, Geiger, and goal-relative observations to discrete actions.

## Dwell

The short name used in result tables for dwell time.

## Estimator

An interpolation or parametric model that constructs a radiation field from
measurements. The project primarily uses kernel regression, inverse-distance
weighting, parametric source fitting, and shield-aware parametric fitting.

## Held-Out Set

A pair or goal set excluded from model selection and reserved for a final guarded
test. It prevents reuse of validation evidence during the test decision.

## Hypothesis

A labelled explanation of why an observed result might have occurred. Hypotheses
are not claims of proof and are marked as such in the project documents.

## IDW

Inverse-distance weighting. Its field estimate is a power-weighted average of
nearby point measurements.

## Kernel Estimator

Gaussian Nadaraya-Watson kernel regression with a bandwidth selected by
leave-one-out cross-validation on measured cells.

## Measurement Pattern

The rule defining which free cells a robot measures. Examples include the current
cell, a Moore neighborhood, and discs of a named radius.

## Mutual Success

A paired comparison subset containing episodes in which both compared methods
successfully reached the goal. Failed episodes are excluded from paired dose
differences.

## Negative Result

A pre-declared result where a method or variant does not pass its decision rule.
Negative results are reported without re-labeling them as success.

## Null Result

An outcome that does not show a pre-declared statistically or practically
meaningful difference. The project treats no result as a win by relabeling absence
of evidence.

## Oracle

A planning reference that executes the relevant A* calculation on the true radiation
field. An oracle is an experimental reference, not an implementable online policy.

## OMP

Orthogonal matching pursuit over candidate source positions. OMP3 assumes three
sources, while OMP3S additionally assumes a known shielding model.

## Paired Comparison

A comparison of two methods on the same starts or episodes, restricted to mutual
successes. It preserves per-start pairing rather than comparing independent
aggregates.

## Parametric Estimator

A source-model estimator that represents the field as weighted inverse-square
source kernels. It differs from a generic spatial interpolator because it assumes
a physical source form.

## Radiation-Aware

Planning or learning that includes radiation intensity in the cost or reward.

## Replay Buffer

The DQN transition store used for sampled gradient updates.

## Shielding Coefficient

The attenuation parameter mu used in wall-shielded fields. It controls exponential
loss along straight segments passing through wall cells.

## Sensor Agent

A map-free navigation agent that observes local geometry, radiation, and
goal-relative features instead of an absolute global radiation map.

## Survey Budget

The maximum number of extra steps allocated to a pre-goal survey tour. It is
separate from the maximum goal-directed episode length.

## Survey Pattern

A fixed route over coarse-grid waypoints selected before the robot is sent to the
goal. The route is nearest-neighbour ordered in a predefined free-cell grid.

## Success

Reaching the goal within the episode step guard without a wall collision.

## Tabular Reinforcement Learning

An episodic state-action learner that stores values for grid states and actions,
including tabular Q-learning and SARSA in this project.

## Wall-Blind Oracle

A reference planned on the free-space field with true sources but without
shielding attenuation, then executed on the shielded true field.

## Wall-Masking

The rule that sets wall-cell estimates or radiation values to zero where the
estimator is required to exclude obstacles.

## Cross-References

- [Experimental protocol](EXPERIMENTS.md)
- [Technical report](TECHNICAL_REPORT.md)
- [Field-estimation documentation](docs/field_estimation.md)
- [Sensor-agent documentation](docs/sensor_agent.md)
- [Results index](RESULTS_INDEX.md)
