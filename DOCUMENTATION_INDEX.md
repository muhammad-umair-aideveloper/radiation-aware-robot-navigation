# Reviewer Documentation Index

Use this index to choose a reading depth. It also identifies generated results,
protocol sources, and verification artifacts.

## 5-Minute Path

1. Read [README.md](README.md) for the project overview and headline findings.
2. Read [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for the synthesis, limitations,
   and negative results.
3. Consult [docs/reviewer_faq.md](docs/reviewer_faq.md) for concise answers to 18 key
   technical and methodological evaluation questions.
4. Use [GLOSSARY.md](GLOSSARY.md) when a term such as oracle or dose regret needs
   a precise meaning.

## 30-Minute Path

1. Start with [README.md](README.md) for scope and the benchmark at a glance.
2. Read [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for methods, results, threats,
   verification, and future work.
3. Read [EXPERIMENTS.md](EXPERIMENTS.md) for pre-declared plans and controls.
4. Read [docs/field_estimation.md](docs/field_estimation.md) for S7-S14 estimation
   and online-replanning evidence, including the post hoc S14 failure analysis.
   The S13 fresh-layout replication and [S14 summary](results/s14_analysis.md) are
   also indexed in that page and in
   [RESULTS_INDEX.md](RESULTS_INDEX.md).
5. Read [docs/sensor_agent.md](docs/sensor_agent.md) for S5-S6 sensor-agent behavior.
6. Consult [RESULTS_INDEX.md](RESULTS_INDEX.md) to locate producer scripts and
   result provenance.
7. Inspect [COVERAGE.md](COVERAGE.md) and the [number audit](scripts/audit_numbers.py)
   to understand automated verification.

## S14 Exploratory Failure Analysis

Read [S14 field analysis](docs/field_estimation.md) for the design, fidelity control,
per-layout and association tables, figures, verdicts, and limitations. See the
[generated S14 results](results/s14_analysis.md) and the
[exploratory-analysis decision](docs/design_decisions.md#11-separate-exploratory-s14-analysis-from-s13-verdicts)
for source data and interpretation rules.

## Where to Start

- New reviewer: 5-minute path.
- Implementer: technical report, protocol, and source index.
- Test reviewer: quality gate, golden references, coverage, and result index.
- Scientific reviewer: protocol first, then report and result-specific documents.

## Core Documents

| Document | Purpose |
|---|---|
| [README.md](README.md) | Project overview and quick results |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Consolidated technical synthesis |
| [GLOSSARY.md](GLOSSARY.md) | Precise project terminology |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Pre-declared experimental plans |
| [RESULTS_INDEX.md](RESULTS_INDEX.md) | Result producer and provenance index |
| [docs/field_estimation.md](docs/field_estimation.md) | S7-S14 estimator and replanning studies |
| [docs/design_decisions.md](docs/design_decisions.md) | Reviewer-facing implementation decisions and alternatives |
| [docs/reviewer_faq.md](docs/reviewer_faq.md) | Answers to 18 key reviewer questions grounded in study evidence |
| [COVERAGE.md](COVERAGE.md) | Core-module coverage summary |
