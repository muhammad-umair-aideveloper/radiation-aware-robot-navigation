"""Generate CHANGELOG.md from conventional commit subjects."""

import subprocess
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TYPE_ORDER = OrderedDict(
    [
        ("feat", "Features"),
        ("fix", "Fixes"),
        ("docs", "Documentation"),
        ("test", "Tests"),
        ("ci", "Continuous Integration"),
        ("build", "Build and Tooling"),
        ("refactor", "Refactoring"),
        ("chore", "Chores"),
        ("other", "Other"),
    ]
)


def git_log() -> list[tuple[str, str]]:
    output = subprocess.check_output(
        ["git", "log", "--pretty=format:%h%x09%s"],
        cwd=ROOT,
        text=True,
    )
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        commit, subject = line.split("\t", 1)
        if "backup" in subject.lower() or "day12-backup" in subject.lower():
            continue
        rows.append((commit, subject))
    return rows


def main() -> int:
    grouped: dict[str, list[tuple[str, str]]] = {
        key: [] for key in TYPE_ORDER
    }
    for commit, subject in git_log():
        kind = "other"
        if subject.startswith("feat"):
            kind = "feat"
        elif subject.startswith("fix"):
            kind = "fix"
        elif subject.startswith("docs"):
            kind = "docs"
        elif subject.startswith("test"):
            kind = "test"
        elif subject.startswith("ci"):
            kind = "ci"
        elif subject.startswith("build"):
            kind = "build"
        elif subject.startswith("refactor"):
            kind = "refactor"
        elif subject.startswith("chore"):
            kind = "chore"
        grouped[kind].append((commit, subject))

    lines = ["# Changelog", "", "Generated from conventional commit subjects.", ""]
    for kind, heading in TYPE_ORDER.items():
        rows = grouped[kind]
        if not rows:
            continue
        lines.append(f"## {heading}")
        lines.append("")
        for commit, subject in rows:
            lines.append(f"- `{commit}` {subject}")
        lines.append("")
    (ROOT / "CHANGELOG.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote CHANGELOG.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
