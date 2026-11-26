"""Tests for result manifest change, deletion, and addition detection."""

import hashlib

import pytest

from scripts.results_integrity import IntegrityError, verify_manifest


def _root_with_manifest(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    summary = results / "summary.json"
    figure = results / "layout.png"
    summary.write_bytes(b'{"ok": true}\n')
    figure.write_bytes(b"image-bytes")
    return tmp_path, {
        "algorithm": "sha256",
        "files": {
            "results/summary.json": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "results/layout.png": hashlib.sha256(figure.read_bytes()).hexdigest(),
        },
    }


def test_manifest_detects_changed_bytes(tmp_path) -> None:
    root, manifest = _root_with_manifest(tmp_path)
    (root / "results" / "summary.json").write_bytes(b'{"ok": false}\n')
    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        verify_manifest(root, manifest)


def test_manifest_detects_missing_file(tmp_path) -> None:
    root, manifest = _root_with_manifest(tmp_path)
    (root / "results" / "layout.png").unlink()
    with pytest.raises(IntegrityError, match="missing"):
        verify_manifest(root, manifest)


def test_manifest_detects_extra_file(tmp_path) -> None:
    root, manifest = _root_with_manifest(tmp_path)
    (root / "results" / "untracked.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IntegrityError, match="extra"):
        verify_manifest(root, manifest)
