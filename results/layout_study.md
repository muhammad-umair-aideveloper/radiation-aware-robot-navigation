# Multi-Layout Generalization Study Results (L1)

Comparison of **A* Shortest** vs. **A* Aware** ($\lambda = 2.5$) across 10 random procedural layouts (seeds 100–109), evaluated on 50 starts per layout (total $N = 500$ evaluation pairs).

| Layout Seed | Wall Frac | Shortest Dose (a.u.) | Aware Dose (a.u.) | Dose Diff (a.u.) | 95% Bootstrap CI | Dose Red (%) | Step Diff (cells) | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|---|
| 100 | 17.0% | 127.21 | 54.61 | -72.60 | [-104.44, -44.30] | -57.1% | +3.92 | [2.06, 6.06] |
| 101 | 14.1% | 302.18 | 138.34 | -163.83 | [-202.08, -129.96] | -54.2% | +23.16 | [17.92, 28.80] |
| 102 | 11.7% | 60.34 | 33.89 | -26.45 | [-42.31, -15.35] | -43.8% | +10.54 | [6.82, 14.88] |
| 103 | 14.5% | 106.87 | 35.29 | -71.59 | [-104.83, -42.83] | -67.0% | +9.70 | [6.24, 13.46] |
| 104 | 13.0% | 89.64 | 53.56 | -36.09 | [-61.74, -17.39] | -40.3% | +6.42 | [3.68, 9.72] |
| 105 | 17.7% | 151.26 | 67.71 | -83.55 | [-115.11, -54.39] | -55.2% | +8.76 | [6.02, 11.76] |
| 106 | 15.7% | 222.97 | 67.85 | -155.12 | [-208.72, -105.37] | -69.6% | +12.08 | [8.84, 15.54] |
| 107 | 11.8% | 44.74 | 31.14 | -13.60 | [-22.65, -5.83] | -30.4% | +1.84 | [0.72, 3.24] |
| 108 | 18.8% | 180.82 | 86.55 | -94.27 | [-135.81, -59.71] | -52.1% | +5.16 | [4.20, 6.24] |
| 109 | 13.0% | 161.46 | 42.28 | -119.19 | [-150.09, -88.40] | -73.8% | +27.92 | [20.94, 35.28] |
| **POOLED** | **N=500** | **144.75** | **61.12** | **-83.63** | **[-94.53, -73.18]** | **-57.8%** | **+10.95** | **[9.57, 12.36]** |

### Key Findings
- Across all 10 independent random layouts, radiation-aware planning consistently lowered absorbed dose (pooled reduction: **57.8%**).
- The pooled paired dose reduction is **-83.63 a.u.** (95% CI: [-94.53, -73.18]), achieved with a modest step overhead of **10.95 cells** (95% CI: [9.57, 12.36]).
- The 95% bootstrap confidence interval strictly excludes zero in all 10 tested layouts.
