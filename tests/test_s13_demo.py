"""Smoke test for the fast S13 canonical demo."""

from __future__ import annotations

from src.s13_demo import run_demo


def test_s13_demo_fast_smoke() -> None:
    result = run_demo(fast=True)
    assert result["elapsed_seconds"] < 30.0
    assert len(result["rows"]) == 4
    assert all(row["starts"] >= 2 for row in result["rows"])
    assert result["audit"]["passed"]
