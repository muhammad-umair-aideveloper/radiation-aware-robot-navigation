# S13 Fresh-Layout Replication Summary

## Per-layout results

| Layout | Oracle saving (%) | Offline kernel regret (%) | Offline IDW regret (%) | Offline omp3 regret (%) | Online kernel regret (%) | Online omp3 regret (%) |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 48.9 | 43.7 | 48.7 | 33.5 | 38.9 | 45.8 |
| 201 | 79.0 | 10.2 | 16.8 | -0.2 | 241.1 | 80.2 |
| 202 | 56.8 | 63.8 | 37.5 | 3.4 | 66.1 | 70.3 |
| 203 | 54.5 | 25.0 | 53.8 | 36.6 | 32.2 | 25.0 |
| 204 | 55.0 | 8.6 | 16.4 | 22.8 | 53.2 | 72.6 |
| 205 | 60.7 | 41.1 | 58.8 | 12.6 | 89.0 | 55.6 |
| 206 | 53.2 | 41.5 | 92.1 | 9.5 | 66.4 | 41.2 |
| 207 | 70.0 | 128.8 | 63.5 | 9.7 | 137.3 | 49.2 |
| 208 | 53.4 | 38.6 | 13.6 | 0.3 | 97.4 | 53.6 |
| 209 | 64.0 | 15.4 | 26.4 | 11.3 | 128.7 | 93.1 |
| 210 | 19.0 | 9.9 | 9.5 | 9.1 | 18.8 | 9.2 |
| 211 | 9.7 | 10.9 | 9.0 | 3.5 | 15.9 | 28.0 |
| 212 | 46.0 | 41.1 | 54.6 | 46.3 | 54.8 | 62.3 |
| 213 | 59.5 | 31.8 | 27.2 | 0.4 | 74.4 | 43.4 |
| 214 | 45.0 | 66.3 | 73.0 | 107.5 | 62.5 | 34.0 |
| 215 | 59.0 | 56.1 | 23.9 | 10.4 | 58.0 | 69.0 |
| 216 | 69.2 | 30.8 | 53.4 | 25.7 | 108.1 | 90.3 |
| 217 | 45.7 | 50.4 | 49.3 | 43.7 | 95.8 | 79.2 |
| 218 | 64.5 | 45.5 | 99.9 | 36.7 | 98.9 | 95.0 |
| 219 | 46.0 | 29.5 | 30.8 | 0.3 | 100.7 | 40.4 |

## Pooled layout means

| Metric | Mean [95% bootstrap interval] |
|---|---:|
| oracle_saving_percent | 53.0 [45.8, 59.6] |
| offline_kernel_regret_percent | 39.4 [28.8, 52.1] |
| offline_idw_regret_percent | 42.9 [31.9, 54.5] |
| offline_omp3_regret_percent | 21.1 [11.6, 33.2] |
| online_kernel_regret_percent | 81.9 [62.2, 105.2] |
| online_omp3_regret_percent | 56.9 [46.7, 67.2] |
| offline_kernel_saving_percent | 35.6 [27.7, 43.7] |
| offline_omp3_saving_percent | 42.7 [32.6, 52.2] |
| online_kernel_saving_percent | 20.0 [13.6, 26.0] |
| online_omp3_saving_percent | 28.6 [21.2, 35.9] |

## Paired layout counts

| Comparison or threshold | Layout count |
|---|---:|
| omp3 offline beats kernel offline | 16/20 |
| omp3 online beats kernel online | 14/20 |
| Oracle saving at least 10% | 19/20 |
| Kernel online regret over 50% | 16/20 |

## Hypothesis verdicts

| Hypothesis | Verdict |
|---|---|
| H1: omp3 online beats kernel in at least 15 layouts | not supported |
| H2: oracle saving is at least 10% in at least 15 layouts | supported |
| H3: kernel online regret exceeds 50% in at least 15 layouts | supported |

## Supplementary post-hoc analysis

Status: post hoc (decided after S13 results were seen). Note: 2 planned comparisons; p-values are unadjusted and descriptive. Pre-declared verdicts (H1: not supported, H2: supported, H3: supported) remain unchanged under their pre-declared thresholds.

### Exact two-sided binomial sign test (omp3 vs kernel)

| Mode | Wins (omp3) | Losses (kernel) | Ties | n | Two-sided p-value |
|---|---:|---:|---:|---:|---:|
| offline | 16 | 4 | 0 | 20 | 0.0118 |
| online | 14 | 6 | 0 | 20 | 0.1153 |

### Paired dose regret difference (kernel minus omp3, %)

| Mode | Mean diff [%] | Median diff [%] | 95% bootstrap interval |
|---|---:|---:|---:|
| offline | 18.3 | 9.5 | [5.2, 33.0] |
| online | 25.0 | 17.2 | [9.1, 44.7] |
