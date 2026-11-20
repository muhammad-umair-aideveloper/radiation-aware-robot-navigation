"""Plain-assert test for the S12 B=0 integration control."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.online_controls_s12 import run_control


def test_s12_budget_zero_control() -> None:
    result = run_control()
    assert result["passed"] is True
    assert len(result["checks"]) == 4
    assert all(check["passed"] for check in result["checks"])
    assert all(check["max_abs_difference"] <= 1e-9 for check in result["checks"])


if __name__ == "__main__":
    test_s12_budget_zero_control()
    print("S12 control test passed")
