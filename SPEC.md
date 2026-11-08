# SPEC.md: Project Specification

Repo: `radiation-aware-robot-navigation`

This file holds the full specification (goal, README outline, constants,
protocol, tests, common mistakes, limitations). The rules for working in this
repo (git, credits, reporting) are in `AGENTS.md`.

Read ONLY the section named in your current task. Do not re-read the whole
file for every task. All constants in Section 8 are declared before any
experiment and live in `src/config.py`; do not change them after seeing
results.

---

## 1. Project goal and scope

A 2D grid world (50 x 50) with walls and invisible radiation hotspots. A
robot with a LiDAR and a Geiger-counter-style sensor must go from a start
cell to a goal cell without hitting walls while absorbing as little
radiation dose as possible.

Four methods are compared on the same fixed evaluation set:

| Method | Knows the map? | What it optimises |
|---|---|---|
| `astar_shortest` | yes | path length only (ignores radiation) |
| `astar_aware` | yes | path length + radiation dose (same objective as the RL reward) |
| `rl_normal` | no (sensors only) | PPO with radiation weight 0 |
| `rl_aware` | no (sensors only) | PPO with radiation weight lambda = 5 |

`astar_aware` is the full-information reference. The RL agents are NOT
expected to beat it. The scientific question is: does the radiation term in
the reward make the PPO robot absorb less dose than the PPO robot trained
without it, and how much longer is its path?

This is a toy simulation study. It is not a validated radiation-transport or
real-robot result, and the README must never present it as one.

---

---

## 7. README requirements

Sections, in this order:
1. Title and one-line description
2. Key results at the top: a small table of the real numbers for the four
   methods (success rate, mean path length, mean dose) plus the GIF, and one
   honest sentence stating what the study does and does not show
3. Problem statement (2 to 3 lines, with physics context: robots replacing
   humans in high-radiation zones, dose minimisation, ALARA idea)
4. Environment: grid, walls, sources, the softened inverse-square field,
   sensors, actions, observation, reward (with a formula), termination
5. Methods: the four methods, the shared objective, the potential-based
   shaping, PPO settings
6. Experimental protocol: pre-declared constants, seeds, training budget,
   validation and evaluation pairs, how the pairs were generated
7. Results: the tables from `results/metrics.md`, learning curves, paths
   overview, the paired comparison with confidence intervals
8. How the GIF pair was chosen (the rule from Section 8.8)
9. Project structure
10. How to run (copy-paste Windows PowerShell commands you have actually run)
11. Limitations and future work (Section 10)
12. License (MIT)

Never write a number in the README that is not in a results file.

---

---

## 8. SPECIFICATION

All constants below are declared before any experiment and live in
`src/config.py`. Do not change them after seeing results (Golden rule 4).

### 8.1 Layout (`src/layout.py`)

- Grid 50 x 50, cell size 1. Index arrays as `[row, col]`; `x = col`,
  `y = row`. A cell is free (0) or a wall (1).
- The layout is a fixed, hand-designed function (no randomness): outer area
  open, 2 to 3 internal wall segments with gaps (doors) and several block
  obstacles. Avoid deep dead-end mazes. Wall fraction between 10% and 25%.
- All free cells must be connected (tested).
- 3 radiation sources (cell positions and strengths in config), placed on
  free cells so that direct routes between typical start and goal regions pass
  near at least two of them, while a longer low-dose detour exists.
- Layout validity check (`src/validate_layout.py`), run BEFORE any RL
  training on the 200 VALIDATION pairs (never the evaluation pairs): mean
  dose of `astar_shortest` must be at least 1.5 times the mean dose of
  `astar_aware` (`LAYOUT_MIN_DOSE_RATIO` in config). If not, adjust the layout
  (one commit per version, with the ratio in the message body), then re-run.
  This is an environment-design check, not result tuning, and it must be
  documented in `results/layout_validation.md` with every attempt.

### 8.2 Radiation field (`src/radiation.py`)

- Softened inverse-square law from each source i with strength `S_i` in
  (0, 1]: `I_i(c) = S_i * r_core^2 / (d_i(c)^2 + r_core^2)`, where `d_i` is the
  distance (in cells) from the cell centre to the source centre and
  `r_core = 1.5` cells. Total `I(c) = min(1, sum_i I_i(c))`.
- Far from a source (d >> r_core) this follows `I ~ 1/d^2`; the soft core
  removes the singularity at d = 0. Document this in the docstring.
- Walls do NOT shield in this version (limitation).
- The field is computed once for the whole grid and stored as a float32 array.
- Dose of one step = `I(new cell) * step_length`, with step_length 1 for
  orthogonal and sqrt(2) for diagonal moves. Cumulative dose is the sum over
  all steps of an episode. The robot has no dose limit and is never "burned"
  or terminated by dose; dose is only measured and penalised.

### 8.3 Sensors (`src/sensors.py`)

- LiDAR: 16 rays at angles k * 22.5 degrees (k = 0..15), maximum range 15
  cells, returning distance to the first wall or grid border, normalised to
  [0, 1] by the maximum range. Implement with an efficient method (for
  example a precomputed set of sample offsets per ray or a grid DDA) and
  test it against hand-computed geometry.
- Geiger readings: the intensity `I` at the robot's current cell and at its 8
  neighbouring cells (a 3 x 3 neighbourhood; wall or outside cells read 0).
  This neighbourhood is an assumed local-gradient sensor and must be stated
  as an assumption in the README.
- Sensors are noise-free in this version (limitation).

### 8.4 Environment (`src/env.py`)

- `gymnasium.Env`, class `RadiationNavEnv`, fully seeded through
  `reset(seed=...)`, returns `(obs, info)`, and `step` returns
  `(obs, reward, terminated, truncated, info)` with the Gymnasium API.
- Actions: `Discrete(8)` (N, NE, E, SE, S, SW, W, NW).
- Observation: `Box(float32)` of 28 numbers: 16 LiDAR + 9 Geiger + goal
  vector `(dx, dy) / GRID_SIZE` (2) + Euclidean goal distance
  `/ (GRID_SIZE * sqrt(2))` (1).
- Collision: the target cell is a wall or outside the grid, or the move is
  diagonal and either orthogonal neighbour is a wall (no corner cutting).
  On collision the episode terminates with reward -100 and the robot stays.
- Goal: robot enters the goal cell, the episode terminates with reward +100.
- Time limit: `MAX_STEPS = 300`, reported as `truncated = True` (not
  `terminated`).
- Per-step reward (non-terminal part):
  `r = -STEP_PENALTY * L - LAMBDA * I(new) * L + F`, with `L` the step length,
  `STEP_PENALTY = 0.1`, `LAMBDA = 5` (aware) or `0` (normal), and `F` the
  shaping term below. Terminal bonus or penalty is added on goal or
  collision.
- Potential-based shaping: `F = GAMMA * Phi(s') - Phi(s)` with
  `Phi(s) = -SHAPING_SCALE * d_free(s)`, where `d_free` is the 8-connected
  free-space shortest-path length (step lengths 1 and sqrt(2), radiation
  ignored) from the cell to the goal, from a BFS/Dijkstra map computed once
  per episode. `SHAPING_SCALE = 1.0`, `GAMMA = 0.99`. `Phi` of a terminal
  state is 0. A test must show that the shaping terms telescope along a
  trajectory.
- `info` contains: `dose_step`, `dose_total`, `path_length`, `outcome`
  (`goal`, `collision` or `timeout`), and `radiation_at_cell`.
- Start and goal sampling: free cells with `I < 0.05` and free-space path
  length between them of at least 20 cells. Training pairs are sampled online
  from the training RNG and must never equal a validation or evaluation pair.

### 8.5 Planners (`src/planners.py`)

- A* on the same 8-connected grid with the same collision rules (no corner
  cutting) and an octile-distance heuristic.
- `astar_shortest`: cost per step `L`.
- `astar_aware`: cost per step `L * (1 + KAPPA * I(new))` with
  `KAPPA = LAMBDA / STEP_PENALTY = 50`, i.e. the same per-step objective as
  the RL reward (without shaping and terminal terms).
- Tests: shortest path length on an empty grid matches the known octile
  value; paths never cross walls; `astar_aware` makes a detour (lower dose,
  longer or equal path) on a small hand-made grid with one hotspot.

### 8.6 Training protocol (`src/train_agent.py`)

- Stable-Baselines3 `PPO`, `MlpPolicy`, 4 `DummyVecEnv` environments, each
  wrapped in `Monitor`, CPU only.
- Hyperparameters (declared, not tuned on the evaluation pairs):
  `learning_rate 3e-4`, `n_steps 256`, `batch_size 256`, `n_epochs 10`,
  `gamma 0.99`, `gae_lambda 0.95`, `ent_coef 0.01`, `clip_range 0.2`,
  `net_arch [64, 64]`.
- Variants: `normal` (`LAMBDA = 0`) and `aware` (`LAMBDA = 5`). Seeds 0, 1, 2
  for each variant. Same hyperparameters and the same number of timesteps for
  both variants.
- `TOTAL_TIMESTEPS` is chosen once, after a smoke test that measures the real
  steps per second, so that each run takes about 10 to 12 minutes on the
  user's laptop. Record it in `config.py` in its own commit BEFORE the full
  runs and write the measured speed to `results/training_speed.md`.
- Use the FINAL model of each run. No checkpoint selection on evaluation
  pairs. Validation pairs may be used for learning curves only.
- CLI: `--variant {normal,aware}`, `--seed`, `--timesteps` (default from
  config), `--out-dir` (default: the repo folders; smoke tests use a temp
  directory).
- Save per run: `models/ppo_<variant>_seed<k>.zip`, a training log CSV and
  the real wall-clock time and steps per second in
  `results/runs/<variant>_seed<k>/`.
- If a run is unstable, crashes or looks wrong, keep its outputs and report it
  (Golden rule 5). Do not silently re-run until it looks good.
- Tuning: if the user asks for tuning, use only the validation pairs, try at
  most 3 configurations, keep a table of ALL of them in
  `results/experiments/`, one commit per attempt.

### 8.7 Evaluation protocol (`src/evaluate_agents.py`, `src/pairs.py`)

- Generate once and save: `results/val_pairs.json` (200 pairs) and
  `results/eval_pairs.json` (200 pairs), with fixed seeds from config,
  disjoint from each other. Commit them before any RL training.
- Evaluate each method on the 200 evaluation pairs, deterministic policy for
  RL (`deterministic=True`). The evaluation pairs are used for final
  measurement only (a failed or partial run must be reported, not hidden).
- Metrics per method (and per seed for RL): success rate, collision rate,
  timeout rate; and for SUCCESSFUL episodes: mean path length and mean
  cumulative dose.
- Paired comparison: for each comparison (for example `rl_aware` vs
  `rl_normal` of the same seed, `rl_aware` vs `astar_shortest`, `rl_aware` vs
  `astar_aware`), use only pairs where BOTH methods succeed, report the number
  of pairs used, the mean difference in dose and in path length, and a 95%
  bootstrap confidence interval (10,000 resamples, seed from config).
  Never compare doses on different sets of pairs, and never include failed
  episodes in dose averages without saying so.
- RL results are reported per seed and as mean and standard deviation across
  the 3 seeds.
- The primary pre-declared question is the paired dose difference
  `rl_aware - rl_normal`. Report the result whether or not it is favourable.
- Write `results/metrics.md` and `results/metrics.json` from the run, never
  by hand.

### 8.8 Visualisation (`src/visualize.py`)

- `results/radiation_navigation.gif`: 4 panels for ONE evaluation pair:
  `astar_shortest`, `astar_aware`, `rl_normal` (seed 0), `rl_aware` (seed 0).
  Each panel: radiation heatmap (`inferno`), walls in black, start and goal
  markers, the trail so far, the robot, a running cumulative-dose label, the
  method name and the outcome. Same colour scale in all panels. Keep a
  finished robot on its last frame until all panels end.
- Pair choice rule, declared now: among evaluation pairs solved by all four
  methods, take the pair whose `astar_shortest` dose is the median of those
  pairs. No cherry-picking; state the rule in the README. If the RL agents do
  not solve it, show that honestly.
- Also create `results/paths_overview.png` (the first 5 evaluation pairs, all
  four methods, final paths only) and `results/learning_curves.png` (mean
  episode reward and mean episode dose against timesteps for both variants,
  mean and spread over seeds, from the logs).
- GIF at most 10 MB; use matplotlib with `PillowWriter` or imageio.

### 8.9 Tests (`tests/`, run with `python -m pytest -v`)

- `test_radiation.py`: peak value at a source equals its strength; far-field
  ratio `I(2d) / I(d)` is about 1/4 within 10% (for d >> r_core, single
  source, ignore the cap); values in [0, 1].
- `test_sensors.py`: LiDAR distances on a hand-built grid (wall at a known
  distance in four directions, border, diagonal ray); Geiger 3 x 3 layout.
- `test_layout.py`: grid size, free cells connected, wall fraction in range,
  sources on free cells.
- `test_env.py`: `gymnasium.utils.env_checker.check_env` passes; same seed
  gives the same episode; collision gives -100 and terminated; goal gives
  +100 and terminated; timeout gives `truncated`; radiation penalty is
  proportional to `LAMBDA`; shaping telescopes; observation shape and range.
- `test_planners.py`: as in Section 8.5.
- `test_pairs.py`: pair lists have 200 entries, start and goal are valid,
  distance rule holds, validation and evaluation lists are disjoint.
- `test_evaluate.py`: bootstrap interval, paired-subset logic and the
  "failed episodes are excluded from dose" rule on synthetic data.

---

---

## 9. Common mistakes to avoid

- Using evaluation pairs for training, tuning, checkpoint selection or the
  layout check.
- Changing the layout, constants or metrics after seeing results.
- Claiming that RL beats A*. `astar_aware` has the full map and is the
  reference; report the gap honestly.
- Comparing doses on different sets of pairs, or including failed episodes in
  dose averages.
- Shaping that is not potential-based, or `Phi` of a terminal state not 0.
- Mixing up `terminated` and `truncated` (time-limit truncation must be
  handled as truncation).
- Forgetting `deterministic=True` at evaluation, or forgetting to seed the
  environment, the action space and PyTorch.
- Wrapping environments without `Monitor`, so logs are missing.
- Recomputing the radiation field or distance maps at every step (training
  becomes far too slow).
- Using `SubprocVecEnv` on Windows, or `pygame`.
- Cherry-picking the GIF pair, or showing only successful runs.
- Hiding a failed or unstable run, or overwriting earlier results.
- Describing a simulated 2D grid as a validated radiation-protection tool.

---

---

## 10. Limitations that the README must state honestly

- Toy 2D grid world with a fixed layout. Generalisation to unseen layouts is
  NOT tested or claimed.
- Simplified radiation: softened inverse-square point sources, static, no
  shielding by walls, no scattering, no time dependence, no dose-rate or
  dose-limit model.
- Noise-free LiDAR and Geiger sensors; the 3 x 3 Geiger neighbourhood is an
  assumed local-gradient sensor.
- The RL agent has no memory and no map; it is compared with A* planners that
  know the full map.
- Results depend on 3 training seeds, one layout and one set of 200
  evaluation pairs. Any explanation of behaviour is a hypothesis.
- Not a validated safety system; no sim-to-real claims.

---
