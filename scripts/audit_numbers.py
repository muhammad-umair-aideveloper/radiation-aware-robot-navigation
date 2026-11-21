"""Audit numeric claims in project documentation against generated results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUMBER_RE = re.compile(r"(?<![\w./-])[-+]?(?:\d+\.\d+|\d+)(?![\w./-])")
URL_RE = re.compile(r"https?://\S+")
VERSION_RE = re.compile(
    r"\b(?:Python\s+\d+(?:\.\d+)*|v\d+(?:\.\d+)+(?:[A-Za-z0-9.+-]*)?)\b"
)
COMMIT_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")


def strip_ignored(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = URL_RE.sub("", text)
    text = COMMIT_RE.sub("", text)
    text = VERSION_RE.sub("", text)
    text = DATE_RE.sub("", text)
    text = YEAR_RE.sub("", text)
    text = LIST_RE.sub("", text)
    text = re.sub(r"\b\d+\s*[xX]\s*\d+\b", "", text)
    return text


def extract_numbers(
    text: str,
    *,
    ignore_document_rules: bool = True,
) -> list[tuple[str, int]]:
    findings: list[tuple[str, int]] = []
    cleaned = strip_ignored(text) if ignore_document_rules else text
    for line_number, line in enumerate(cleaned.splitlines(), 1):
        for match in NUMBER_RE.finditer(line):
            findings.append((match.group(0), line_number))
    return findings


def decimal_precision(token: str) -> int:
    token = token.lower()
    if "e" in token:
        return max(0, len(token.split(".")[-1].split("e")[0]) if "." in token else 0)
    return len(token.split(".", 1)[1]) if "." in token else 0


def matches_displayed(documented: str, available: str) -> bool:
    try:
        doc = Decimal(documented)
        value = Decimal(available)
    except InvalidOperation:
        return False
    precision = decimal_precision(documented)
    quantum = Decimal(1).scaleb(-precision)
    return value.quantize(quantum) == doc.quantize(quantum)


def load_allowlist(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("entries", [])
    return {str(entry["value"]): str(entry["justification"]) for entry in entries}


def collect_result_numbers(results_dir: Path) -> set[str]:
    values: set[str] = set()
    if not results_dir.exists():
        return values
    for path in results_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token, _ in extract_numbers(text, ignore_document_rules=False):
            values.add(token)
    return values


def audit(
    root: Path,
    allowlist_path: Path,
    document_paths: list[Path],
) -> list[tuple[str, int, str]]:
    result_numbers = collect_result_numbers(root / "results")
    allowlist = load_allowlist(allowlist_path)
    unmatched: list[tuple[str, int, str]] = []
    for path in document_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for token, line_number in extract_numbers(text):
            if token in allowlist:
                continue
            if any(matches_displayed(token, candidate) for candidate in result_numbers):
                continue
            unmatched.append((str(path.relative_to(root)), line_number, token))
    return unmatched


def default_documents(root: Path) -> list[Path]:
    paths = [root / "README.md", root / "EXPERIMENTS.md"]
    paths.extend(sorted((root / "docs").glob("*.md")))
    report = root / "TECHNICAL_REPORT.md"
    if report.exists():
        paths.append(report)
    return [path for path in paths if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", default="config/number_allowlist.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    allowlist = root / args.allowlist
    unmatched = audit(root, allowlist, default_documents(root))
    if unmatched:
        for file_name, line_number, token in unmatched:
            print(f"{file_name}:{line_number}: unmatched number {token}")
        print(f"Number audit failed: {len(unmatched)} unmatched numbers")
        return 1
    print("Number audit passed: every documented number is in results or the allowlist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
