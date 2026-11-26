"""Portability checks for source, scripts, and tests."""

import ast
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
BACKSLASH_PATH = re.compile(r"[A-Za-z0-9_.-]+\\[A-Za-z0-9_.-]+")


def _calls_without_utf8(tree: ast.AST, path: Path) -> list[str]:
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"read_text", "write_text"}:
            if not any(keyword.arg == "encoding" for keyword in node.keywords):
                findings.append(f"{path}:{node.lineno}: {node.func.attr} without encoding")
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            mode = "r"
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value
            encoding = any(keyword.arg == "encoding" for keyword in node.keywords)
            if isinstance(mode, str) and "b" not in mode and not encoding:
                findings.append(f"{path}:{node.lineno}: text open without encoding")
    return findings


def test_source_has_no_hard_coded_absolute_or_windows_paths() -> None:
    findings = []
    for base in SCAN_ROOTS:
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {
                child: node
                for node in ast.walk(tree)
                for child in ast.iter_child_nodes(node)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                value = node.value
                parent = parents.get(node)
                is_regex = (
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Attribute)
                    and parent.func.attr in {"compile", "sub", "search", "match", "fullmatch"}
                    and isinstance(parent.func.value, ast.Name)
                    and parent.func.value.id == "re"
                )
                posix_absolute = (
                    PurePosixPath(value).is_absolute()
                    and value.startswith("/")
                    and not re.fullmatch(r"/(?:\d+|t|b)?", value)
                    and not re.match(r"/\d+(?:\s|\)|\|)", value)
                )
                windows_path = PureWindowsPath(value)
                windows_absolute = (
                    windows_path.is_absolute()
                    and bool(windows_path.drive.strip("\\/"))
                )
                if not is_regex and (posix_absolute or windows_absolute):
                    findings.append(f"{path}:{node.lineno}: absolute path literal")
                if not is_regex and BACKSLASH_PATH.search(value):
                    findings.append(f"{path}:{node.lineno}: backslash path separator")
    assert not findings, "\n".join(findings)


def test_text_file_operations_specify_utf8() -> None:
    findings = []
    for base in SCAN_ROOTS:
        for path in base.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            findings.extend(_calls_without_utf8(tree, path))
    assert not findings, "\n".join(findings)
