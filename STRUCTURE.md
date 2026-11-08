# Project Structure: `radiation-aware-robot-navigation`

A 2D grid world with walls and invisible radiation hotspots. A robot with a
LiDAR and a Geiger-counter-style sensor goes from a start cell to a goal cell
while avoiding collisions and minimising absorbed radiation dose. Four methods
are compared on the same fixed evaluation set: shortest-path A*,
radiation-aware A*, PPO without a radiation penalty, and PPO with a radiation
penalty.

## Folder layout

```
radiation-aware-robot-navigation/
├── README.md                  # Results, method, protocol, how to run
├── AGENTS.md                  # Short rules for the AI coding agent (git, credits, reporting)
├── SPEC.md                    # Full specification (constants, protocol, tests)
├── STRUCTURE.md               # This file
├── requirements.txt           # Python dependencies
├── .gitignore
├── LICENSE                    # MIT
├── src/
│   ├── config.py              # All constants (declared before experiments)
│   ├── layout.py              # Fixed 50x50 map: walls and radiation sources
│   ├── radiation.py           # Softened inverse-square field, dose helpers
│   ├── sensors.py             # LiDAR (16 rays) and Geiger 3x3 readings
│   ├── env.py                 # Gymnasium environment RadiationNavEnv
│   ├── planners.py            # A* shortest and A* radiation-aware
│   ├── pairs.py               # Start/goal pair sampling, validation/eval lists
│   ├── validate_layout.py     # Layout check (dose ratio on validation pairs)
│   ├── train_agent.py         # PPO training (--variant, --seed)
│   ├── evaluate_agents.py     # Fixed protocol, paired comparison, bootstrap CI
│   └── visualize.py           # GIF, paths overview, learning curves
├── tests/
│   ├── test_radiation.py
│   ├── test_sensors.py
│   ├── test_layout.py
│   ├── test_env.py
│   ├── test_planners.py
│   ├── test_pairs.py
│   └── test_evaluate.py
├── models/
│   └── ppo_<variant>_seed<k>.zip      # Trained PPO models (small)
└── results/
    ├── val_pairs.json                 # 200 validation pairs (committed before training)
    ├── eval_pairs.json                # 200 evaluation pairs (committed before training)
    ├── layout_validation.md           # Layout check, every attempt documented
    ├── training_speed.md              # Measured steps/second and wall-clock times
    ├── runs/<variant>_seed<k>/        # Training logs of every run (all kept)
    ├── metrics.md                     # Final tables (real numbers only)
    ├── metrics.json
    ├── learning_curves.png
    ├── paths_overview.png
    └── radiation_navigation.gif       # 4-panel comparison on one evaluation pair
```

## Pipeline

```
config.py ─► layout.py ─► radiation.py ─► sensors.py ─► env.py
                                  │                        │
                                  └──────► planners.py     └─► train_agent.py ─► models/*.zip
                                                │                                   │
                              pairs.py ─► val_pairs / eval_pairs                    │
                                                │                                   │
                                                └──────────► evaluate_agents.py ◄───┘
                                                                   │
                                                  metrics.md ◄─────┴─────► visualize.py ─► GIF, plots
```

## Methods compared

| Method | Knows the map? | Objective |
|---|---|---|
| `astar_shortest` | yes | path length only |
| `astar_aware` | yes | path length + dose (same per-step objective as the RL reward) |
| `rl_normal` | no | PPO, radiation weight 0 |
| `rl_aware` | no | PPO, radiation weight 5 |

`astar_aware` is the full-information reference. The RL agents use only their
sensors: 16 LiDAR rays, the Geiger reading at the current cell and at its 8
neighbours, and the relative goal position.

## Prerequisites (Windows)

1. Python 3.10+ and Git. No C++ compiler is needed for this project.
2. Everything runs on CPU (Intel i5, no GPU). Expect about 10 to 12 minutes
   per training run and 6 runs in total (2 variants x 3 seeds), so plan for
   about 1 to 1.5 hours of training. Keep the laptop on charger power and do
   not let it sleep.

## How to run (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m pytest -v

python src/pairs.py                      # writes results/val_pairs.json and eval_pairs.json
python src/validate_layout.py            # layout check on validation pairs

python src/train_agent.py --variant normal --seed 0
python src/train_agent.py --variant aware  --seed 0
# repeat with --seed 1 and --seed 2

python src/evaluate_agents.py            # metrics.md / metrics.json
python src/visualize.py                  # GIF, paths overview, learning curves
```

The exact verified commands are documented in the README after they have
been run.

## Rules for this structure

- Do not add files or folders outside this layout without a reason.
- Keep paths relative to the repo root (`pathlib.Path`, no hardcoded `C:\...`).
- Commit small results, trained models and logs; keep every run.
- Rendering uses matplotlib only (no pygame).
- Every change to any file gets its own conventional commit (see `AGENTS.md`).
- Training runs are started by the person in a terminal, not by the agent (saves agent credits).
