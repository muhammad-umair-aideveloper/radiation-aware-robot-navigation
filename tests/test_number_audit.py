import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_numbers import (
    audit,
    collect_result_numbers,
    extract_numbers,
    load_allowlist,
    matches_displayed,
)


def test_exact_result_match(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary.json").write_text(
        json.dumps({"dose": 54.422}), encoding="utf-8"
    )
    assert "54.422" in collect_result_numbers(tmp_path / "results")


def test_displayed_rounding_matches(tmp_path: Path) -> None:
    assert matches_displayed("54.42", "54.422")
    assert not matches_displayed("54.41", "54.422")


def test_allowlist(tmp_path: Path) -> None:
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps({"entries": [{"value": "50", "justification": "grid size"}]}),
        encoding="utf-8",
    )
    assert load_allowlist(allowlist) == {"50": "grid size"}


def test_ignore_rules() -> None:
    text = """
```python
x = 12345
```
https://example.com/page/123
2026-11-21
v3.11.2
3afe1b2
1. First item
![plot](image.png)
"""
    assert extract_numbers(text) == []


def test_unmatched_detection(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary.md").write_text("Known 12.34\n")
    (tmp_path / "allowlist.json").write_text(
        json.dumps({"entries": []}), encoding="utf-8"
    )
    doc = tmp_path / "README.md"
    doc.write_text("Measured 12.34 and unknown 987.65\n")
    unmatched = audit(tmp_path, tmp_path / "allowlist.json", [doc])
    assert unmatched == [("README.md", 1, "987.65")]
