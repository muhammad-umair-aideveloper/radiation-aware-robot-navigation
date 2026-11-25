# S13 and Project Design Decisions

This record explains stable reviewer-facing decisions using existing plans, implementation
conventions, result files, and commit history. Status names mean implemented, fixed-by-plan,
or completed; they do not predict future performance.

## 1. Keep the grid and motion model stable

- **Context:** Classical planning, reinforcement learning, and field-estimation studies share a
  synthetic environment.
- **Decision:** Use a 50x50 8-connected grid with corner-cutting constraints and the existing
  radiation field model.
- **Alternatives considered:** A continuous workspace or a 4-connected grid would be less
  comparable with existing results.
- **Evidence:** [Experimental protocol](../EXPERIMENTS.md), [the environment implementation](../src/environment.py),
  and commits `5512ff5` and `be81dd0`.
- **Status:** Implemented and reused across studies.

## 2. Predeclare study protocols before execution

- **Context:** Several studies could otherwise be changed to fit observed outcomes.
- **Decision:** Fix objectives, seeds, configurations, metrics, thresholds, controls, and runtime
  rules in EXPERIMENTS.md before running code.
- **Alternatives considered:** Post-hoc tuning, unreported checkpoint selection, and adaptive
  configuration changes were rejected.
- **Evidence:** [the fixed S13 plan](../EXPERIMENTS.md), especially the S13 section,
  and commit `18631be`.
- **Status:** Completed for S13 and applied to the existing study sequence.

## 3. Use layouts as the replication unit

- **Context:** S13 tests whether earlier online conclusions generalize beyond a single layout.
- **Decision:** Generate fresh layouts, report per-layout values, and bootstrap over layouts.
- **Alternatives considered:** Pooling trajectories before layout-level comparison would obscure
  layout effects.
- **Evidence:** [S13 plan](../EXPERIMENTS.md), [layout set](../results/s13_layouts.json),
  [layout results](../results/s13_summary.md), and commit `807a520`.
- **Status:** Implemented.

## 4. Separate offline and online measurement assumptions

- **Context:** Offline and online methods receive measurements in different ways.
- **Decision:** Keep offline random 50-cell measurement and online 3x3 path measurement as
  separate protocols in S13.
- **Alternatives considered:** Calling the methods equivalent without controlling their
  measurement access would mix extrapolation and coverage effects.
- **Evidence:** [S13 plan](../EXPERIMENTS.md), [S13 runner](../src/s13_study.py),
  and [S13 summary](../results/s13_summary.json).
- **Status:** Implemented.

## 5. Use OMP3 as a strong but limited physics-informed estimator

- **Context:** Earlier kernel and IDW estimators can miss unobserved hotspots.
- **Decision:** Compare them with a parametric softened inverse-square source fit when S13
  declares three sources known.
- **Alternatives considered:** An estimator with unknown source counts or shielding was not
  added to S13 after seeing results.
- **Evidence:** [S13 plan](../EXPERIMENTS.md), [S13 runner](../src/s13_study.py),
  [S10 protocol](../EXPERIMENTS.md), and commit `175c3cc`.
- **Status:** Implemented with a stated strong-assumption limitation.

## 6. Judge paths by true-field dose

- **Context:** Estimated-field planning can look good without improving executed safety.
- **Decision:** Plan on estimated fields but evaluate the resulting path on the true field.
- **Alternatives considered:** Reporting estimated dose only would be self-confirming.
- **Evidence:** [S13 runner](../src/s13_study.py), [S13 layout results](../results/s13_summary.md),
  and the protocol in [EXPERIMENTS.md](../EXPERIMENTS.md).
- **Status:** Implemented.

## 7. Use bootstrap intervals and paired layout counts

- **Context:** Layout means can hide frequency effects and point estimates are insufficient.
- **Decision:** Use layout-level bootstrap intervals and explicit layouts-won counts for S13
  comparisons.
- **Alternatives considered:** A single pooled point estimate or a seed-pooled significance test
  would not answer the fixed layout replication thresholds.
- **Evidence:** [S13 summary script](../src/s13_summary.py),
  [S13 summary JSON](../results/s13_summary.json), and commit `34efcf3`.
- **Status:** Implemented.

## 8. Fail controls before treating replication data as valid

- **Context:** Online replanning could contain a control or path-cost bug.
- **Decision:** Require true-field and zero-field online controls on layouts 200 through 202
  before accepting the S13 design.
- **Alternatives considered:** Reporting study values despite control failure was rejected.
- **Evidence:** [S13 controls](../results/s13_controls.json),
  [S13 runner](../src/s13_study.py), and the S13 plan.
- **Status:** Controls passed.

## 9. Report hypotheses without retroactive claims

- **Context:** The S13 data may support, fail, or leave hypotheses ambiguous.
- **Decision:** Assign supported or not-supported verdicts exactly from the predeclared layout
  thresholds and state intervals where claims depend on separation.
- **Alternatives considered:** Describing every favorable direction as a success was rejected.
- **Evidence:** [S13 plan](../EXPERIMENTS.md),
  [S13 summary](../results/s13_summary.md), and commit `159647d`.
- **Status:** Completed.

## 10. Trace claims to generated artifacts and audited documentation

- **Context:** Reviewer documents include many result-derived numbers and links.
- **Decision:** Generate summaries and reports from result JSON, run the number audit and link
  check, and commit scripts together with generated artifacts.
- **Alternatives considered:** Hand-maintained tables would create drift and require untestable
  transcription checks.
- **Evidence:** [technical report generator](../scripts/generate_technical_report.py),
  [number audit](../scripts/audit_numbers.py), [link check](../scripts/check_links.py),
  and [results index](../RESULTS_INDEX.md).
- **Status:** Implemented.

## 11. Separate exploratory S14 analysis from S13 verdicts

- **Context:** S13's confirmatory online comparison produced a result below its pre-declared
  layout-count threshold, motivating a follow-up examination of the layouts where OMP3 lost.
- **Decision:** Keep the S13 H1 verdict fixed as not supported. The S14 question was post hoc
  because it was selected after S13 results were seen; before S14 analysis code and diagnostics
  ran, the plan fixed the six features, association procedures, diagnostic measures, and H4/H5
  thresholds. Label the follow-up post hoc and exploratory, report it as descriptive, and prefix
  any proposed explanation with "Hypothesis:".
- **Alternatives considered:** Reinterpreting the S13 outcome or changing H1's threshold in
  response to exploratory results would blur the distinction between confirmatory and
  hypothesis-generating analyses.
- **Evidence:** [S14 plan](../EXPERIMENTS.md#s14-exploratory-failure-analysis-of-online-estimation-post-hoc-exploratory),
  [S13 summary](../results/s13_summary.md), [S14 results](../results/s14_analysis.md), and
  [project changelog](../CHANGELOG.md). In repository history, the S14 plan commit
  `895cb30` precedes its feature, diagnostics, and analysis implementation commits.
- **Status:** S14 is reported as post hoc and exploratory; it does not change H1.

## What I would do differently

Given the completed work, I would add machine-readable protocol metadata to every summary
before increasing study breadth; this would make estimator assumptions, seeds, starts, and
controls easier to audit. I would also retain layout-level raw outcomes in a consistent schema
across S13 and earlier studies, rather than reconstructing comparable fields from separate
study-specific result structures. These changes would improve reviewability, but they would
not be evidence for or against the current S13 results.
