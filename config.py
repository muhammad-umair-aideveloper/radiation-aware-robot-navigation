"""Central configuration for radiation-aware-robot-navigation.

Every constant of the experimental protocol lives here and is declared BEFORE
any experiment (see SPEC.md, Section 8). Do not change these values after
seeing results. If a change is unavoidable, make it in its own commit and
explain why.

Placeholders filled in by dedicated commits:
    SOURCE_POSITIONS, SOURCE_STRENGTHS   (when the layout is designed)
    TOTAL_TIMESTEPS                      (after the training-speed smoke test)
"""

from __future__ import annotations

import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root, never hardcoded absolute paths)
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = REPO_ROOT / "src"
MODELS_DIR: Path = REPO_ROOT / "models"
RESULTS_DIR: Path = REPO_ROOT / "results"
RUNS_DIR: Path = RESULTS_DIR / "runs"
EXPERIMENTS_DIR: Path = RESULTS_DIR / "experiments"

VAL_PAIRS_PATH: Path = RESULTS_DIR / "val_pairs.json"
EVAL_PAIRS_PATH: Path = RESULTS_DIR / "eval_pairs.json"
LAYOUT_VALIDATION_PATH: Path = RESULTS_DIR / "layout_validation.md"
TRAINING_SPEED_PATH: Path = RESULTS_DIR / "training_speed.md"
METRICS_MD_PATH: Path = RESULTS_DIR / "metrics.md"
METRICS_JSON_PATH: Path = RESULTS_DIR / "metrics.json"
GIF_PATH: Path = RESULTS_DIR / "radiation_navigation.gif"
PATHS_OVERVIEW_PATH: Path = RESULTS_DIR / "paths_overview.png"
LEARNING_CURVES_PATH: Path = RESULTS_DIR / "learning_curves.png"

# ---------------------------------------------------------------------------
# Grid and layout
# ---------------------------------------------------------------------------
GRID_SIZE: int = 50
WALL_FRACTION_MIN: float = 0.10
WALL_FRACTION_MAX: float = 0.25

# Radiation sources: (row, col) cell positions and strengths in (0, 1].
# Left empty until the layout is designed (3 sources, see SPEC.md 8.1).
NUM_SOURCES: int = 3
SOURCE_POSITIONS: tuple[tuple[int, int], ...] = ()
SOURCE_STRENGTHS: tuple[float, ...] = ()

# ---------------------------------------------------------------------------
# Radiation field: I_i(c) = S_i * r_core^2 / (d^2 + r_core^2), total capped at 1
# ---------------------------------------------------------------------------
R_CORE: float = 1.5
INTENSITY_CAP: float = 1.0

# ---------------------------------------------------------------------------
# Sensors, actions, observation
# ---------------------------------------------------------------------------
LIDAR_NUM_RAYS: int = 16
LIDAR_MAX_RANGE: float = 15.0
GEIGER_NEIGHBORHOOD: int = 3  # 3 x 3 cells around the robot
GEIGER_NUM_VALUES: int = GEIGER_NEIGHBORHOOD * GEIGER_NEIGHBORHOOD
GOAL_VECTOR_SIZE: int = 3  # dx, dy, euclidean distance (all normalised)
OBS_DIM: int = LIDAR_NUM_RAYS + GEIGER_NUM_VALUES + GOAL_VECTOR_SIZE  # 28

NUM_ACTIONS: int = 8
# (d_row, d_col) for N, NE, E, SE, S, SW, W, NW. Row 0 is the top of the grid.
ACTION_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
)
DIAGONAL_STEP_LENGTH: float = math.sqrt(2.0)

# ---------------------------------------------------------------------------
# Episode and reward
# ---------------------------------------------------------------------------
MAX_STEPS: int = 300
STEP_PENALTY: float = 0.1
LAMBDA_NORMAL: float = 0.0
LAMBDA_AWARE: float = 5.0
VARIANT_LAMBDA: dict[str, float] = {"normal": LAMBDA_NORMAL, "aware": LAMBDA_AWARE}
KAPPA_AWARE: float = LAMBDA_AWARE / STEP_PENALTY  # A* radiation-aware weight (= 50)
GOAL_REWARD: float = 100.0
COLLISION_REWARD: float = -100.0
GAMMA: float = 0.99
SHAPING_SCALE: float = 1.0

# ---------------------------------------------------------------------------
# Start / goal pairs
# ---------------------------------------------------------------------------
PAIR_MAX_INTENSITY: float = 0.05  # start and goal must be low-radiation cells
PAIR_MIN_PATH_LENGTH: float = 20.0  # free-space path length (cells)
NUM_VAL_PAIRS: int = 200
NUM_EVAL_PAIRS: int = 200
VAL_PAIRS_SEED: int = 1001
EVAL_PAIRS_SEED: int = 2002

# Layout validity check, on VALIDATION pairs only (SPEC.md 8.1)
LAYOUT_MIN_DOSE_RATIO: float = 1.5

# ---------------------------------------------------------------------------
# PPO training (declared, not tuned on evaluation pairs)
# ---------------------------------------------------------------------------
VARIANTS: tuple[str, ...] = ("normal", "aware")
TRAIN_SEEDS: tuple[int, ...] = (0, 1, 2)
NUM_ENVS: int = 4
PPO_PARAMS: dict[str, object] = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": GAMMA,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
}
POLICY_NET_ARCH: tuple[int, ...] = (64, 64)

# Chosen once, after the training-speed smoke test, in its own commit
# (target: about 10 to 12 minutes per run on the user's laptop).
TOTAL_TIMESTEPS: int | None = None

# ---------------------------------------------------------------------------
# Evaluation and plots
# ---------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES: int = 10_000
BOOTSTRAP_SEED: int = 12345
CONFIDENCE_LEVEL: float = 0.95
PATHS_OVERVIEW_PAIRS: int = 5
GIF_MAX_MB: float = 10.0
