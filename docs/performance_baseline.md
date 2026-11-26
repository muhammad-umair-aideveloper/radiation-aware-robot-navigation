# Performance baseline

Run `python scripts/performance_baseline.py` to regenerate the committed
machine-specific timing record in `results/performance_baseline.json`. It records
median timings for the canonical planners, estimators, an online episode, and the
shielded estimator, together with the machine and library versions.

These measurements describe this machine and software environment only. They are a
baseline for comparison, not a performance guarantee. The bounded benchmark is
designed to finish in less than two minutes and does not add timing assertions to
the test suite.
