"""Create and verify SHA-256 hashes for committed result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config" / "results_integrity.json"


class IntegrityError(ValueError):
    """Raised when the result artifact set or content differs from its manifest."""


def _relative_files(root: Path) -> set[str]:
    result_dir = root / "results"
    if not result_dir.is_dir():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in result_dir.rglob("*")
        if path.is_file()
    }


def tracked_result_files(root: Path = ROOT) -> list[Path]:
    """Return repository-tracked paths beneath results/."""
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "results"], cwd=root
    )
    return [root / path.decode("utf-8") for path in output.split(b"\0") if path]


def create_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Build a deterministic manifest from tracked result files."""
    files = {}
    for path in tracked_result_files(root):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.relative_to(root).as_posix()] = digest
    return {"algorithm": "sha256", "files": dict(sorted(files.items()))}


def verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Raise IntegrityError for missing, extra, or changed result artifacts."""
    if manifest.get("algorithm") != "sha256" or not isinstance(manifest.get("files"), dict):
        raise IntegrityError("manifest must define a SHA-256 files mapping")
    expected = manifest["files"]
    actual_paths = _relative_files(root)
    expected_paths = set(expected)
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"extra: {', '.join(extra)}")
        raise IntegrityError("result file set mismatch (" + "; ".join(details) + ")")
    changed = [
        path for path, digest in expected.items()
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest
    ]
    if changed:
        raise IntegrityError("SHA-256 mismatch: " + ", ".join(sorted(changed)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--write", action="store_true", help="write the current manifest")
    operation.add_argument("--check", action="store_true", help="verify the committed manifest")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    if args.write:
        args.manifest.write_text(
            json.dumps(create_manifest(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.manifest.relative_to(ROOT).as_posix()}")
        return 0

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_manifest(ROOT, manifest)
    print(f"Verified {len(manifest['files'])} result artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
