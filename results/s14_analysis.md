# S14 Exploratory Failure Analysis Results

Status: post hoc, exploratory. All p-values are descriptive. Six features compared without multiplicity adjustment.

## Per-Layout Table

| Layout | Group | Paired diff (%) | Oracle saving (%) | Matched source error | Relative L2 error | Distinct cells |
|---:|---|---:|---:|---:|---:|---:|
| 200 | omp3 loses | -6.92 | 48.9 | 21.84 | 1.3247 | 150.6 |
| 201 | omp3 wins | 160.89 | 79.0 | 18.35 | 1.3422 | 168.1 |
| 202 | omp3 loses | -4.18 | 56.8 | 17.02 | 1.3891 | 177.5 |
| 203 | omp3 wins | 7.22 | 54.5 | 20.94 | 0.8664 | 177.4 |
| 204 | omp3 loses | -19.44 | 55.0 | 15.18 | 1.4127 | 179.4 |
| 205 | omp3 wins | 33.38 | 60.7 | 19.34 | 1.5014 | 156.9 |
| 206 | omp3 wins | 25.24 | 53.2 | 19.66 | 1.3319 | 156.2 |
| 207 | omp3 wins | 88.08 | 70.0 | 13.76 | 1.3261 | 153.5 |
| 208 | omp3 wins | 43.81 | 53.4 | 26.45 | 1.0032 | 145.1 |
| 209 | omp3 wins | 35.56 | 64.0 | 10.75 | 1.1371 | 220.1 |
| 210 | omp3 wins | 9.58 | 19.0 | 23.75 | 1.2499 | 158.2 |
| 211 | omp3 loses | -12.15 | 9.7 | 24.61 | 1.3809 | 178.3 |
| 212 | omp3 loses | -7.45 | 46.0 | 16.53 | 1.1166 | 192.3 |
| 213 | omp3 wins | 30.98 | 59.5 | 15.89 | 1.1134 | 160.6 |
| 214 | omp3 wins | 28.56 | 45.0 | 14.40 | 1.3958 | 168.1 |
| 215 | omp3 loses | -11.05 | 59.0 | 16.27 | 1.5821 | 214.4 |
| 216 | omp3 wins | 17.82 | 69.2 | 18.04 | 1.5328 | 169.5 |
| 217 | omp3 wins | 16.65 | 45.7 | 21.84 | 1.1526 | 168.5 |
| 218 | omp3 wins | 3.88 | 64.5 | 19.69 | 1.2214 | 154.0 |
| 219 | omp3 wins | 60.29 | 46.0 | 16.18 | 0.9947 | 172.0 |

## Feature-Difference Spearman Correlations

| Feature | Spearman rho | 95% CI | Permutation p |
|---|---:|---:|---:|
| oracle_saving_percent | 0.395 | [-0.074, 0.749] | 0.0849 |
| wall_fraction | -0.032 | [-0.510, 0.457] | 0.8917 |
| min_source_distance | -0.534 | [-0.782, -0.083] | 0.0166 |
| mean_source_to_shortest_dist | -0.283 | [-0.656, 0.230] | 0.2201 |
| corridor_intensity_share | 0.239 | [-0.211, 0.666] | 0.3104 |
| mean_shortest_path_length | -0.414 | [-0.716, 0.014] | 0.0711 |

## Group Comparison: Matched Source Position Error

- Losing group (n=6): median = 16.78 cells
- Winning group (n=14): median = 18.85 cells
- Difference of medians (loses minus wins): -2.07 [-4.05, 4.87]

## Verdicts

- H4: **not supported**. Loses median 16.78 vs wins median 18.85; diff of medians -2.07 [-4.05, 4.87].
- H5: **not supported**. Oracle saving Spearman rho = 0.395 [-0.074, 0.749].
- H1: **not supported** (unchanged from S13).
