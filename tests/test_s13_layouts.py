"""Tests for the fixed S13 layout and start set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.layouts import check_connectivity, generate_random_layout
from src.s13_layouts import (
    EARLIER_LAYOUT_SEEDS,
    LAYOUT_SEEDS,
    STARTS_PER_LAYOUT,
    generate_s13_layout_set,
)


ROOT = Path(__file__).resolve().parents[1]


def test_s13_layout_set_properties() -> None:
    result = generate_s13_layout_set()
    assert result["validity"]["passed"]
    assert set(LAYOUT_SEEDS) == set(range(200, 220))
    assert set(LAYOUT_SEEDS).isdisjoint(EARLIER_LAYOUT_SEEDS)

    for seed in LAYOUT_SEEDS:
        layout = generate_random_layout(seed)
        record = result["layouts"][str(seed)]
        assert len(record["starts"]) == STARTS_PER_LAYOUT
        assert check_connectivity(layout.walls, layout.start, layout.goal)
        assert 0.10 <= layout.wall_fraction <= 0.25
        assert len(layout.sources) == 3
        for position, intensity in layout.sources:
            assert not layout.walls[position]
            assert position != layout.start and position != layout.goal
            assert 120.0 <= intensity <= 200.0
        env = layout.to_env()
        from src.s13_layouts import generate_free_space_distances
        distances = generate_free_space_distances(env)
        for row, col in record["starts"]:
            assert not layout.walls[row, col]
            assert distances[row, col] >= 20.0


def test_s13_layout_set_deterministic_and_committed() -> None:
    first = generate_s13_layout_set()
    second = generate_s13_layout_set()
    assert first == second

    stored = json.loads((ROOT / "results" / "s13_layouts.json").read_text(encoding="utf-8"))
    assert stored == first


def test_s13_layout_walls_deterministic() -> None:
    for seed in LAYOUT_SEEDS:
        first = generate_random_layout(seed)
        second = generate_random_layout(seed)
        assert np.array_equal(first.walls, second.walls)
        assert first.sources == second.sources
