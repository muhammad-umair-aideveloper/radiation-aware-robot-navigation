"""Generate a short Markdown coverage summary for core modules."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = [
    "src.environment",
    "src.planners",
    "src.estimators",
    "src.online_sim",
    "src.measurement_patterns",
    "src.survey",
    "src.shielding",
]


def main() -> int:
    coverage_file = ROOT / "coverage.json"
    if not coverage_file.exists():
        print("coverage.json is missing; run pytest --cov")
        return 1
    data = json.loads(coverage_file.read_text(encoding="utf-8"))
    lines = [
        "# Core Module Coverage",
        "",
        "| Module | Line coverage | Target |",
        "|---|---:|---|",
    ]
    for module in CORE_MODULES:
        relative = module.replace(".", "/") + ".py"
        record = next(
            (
                value
                for key, value in data["files"].items()
                if key.replace("\\", "/") == relative
            ),
            None,
        )
        percent = record["summary"]["percent_covered"] if record else 0.0
        lines.append(
            f"| `{module}` | {percent:.1f}% | {'PASS' if percent >= 85.0 else 'BELOW'} |"
        )
    (ROOT / "COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote COVERAGE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
