# Test Set Evaluation Usage Log

Guarded audit log recording every evaluation on the final held-out test set (`results/sensor_test_pairs.json`).

| Timestamp (UTC) | Models Evaluated | Command | Forced |
|---|---|---|---|
| 2026-11-14 06:38:21 UTC | 7 runs: dqn_shaped_lam2.5_seed0..dqn_s6_m1_lam2.5_seed0 | `src/evaluate_sensor.py --set test --runs dqn_shaped_lam2.5_seed0 dqn_shaped_lam2.5_seed1 dqn_shaped_lam2.5_seed2 dqn_shaped_lam0.0_seed0 dqn_shaped_lam0.0_seed1 dqn_shaped_lam0.0_seed2 dqn_s6_m1_lam2.5_seed0 --out-md results/sensor_s6_final.md --out-json results/sensor_s6_final.json --no-canonical` | False |
