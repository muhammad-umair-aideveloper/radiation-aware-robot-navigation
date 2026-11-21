"""Check relative Markdown links and image references."""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def main() -> int:
    missing: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if ".venv" in path.parts or ".git" in path.parts or "tmp" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for target in LINK_RE.findall(text):
            target = target.strip().split()[0].strip("<>")
            if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    if missing:
        for item in missing:
            print(item)
        print(f"Link check failed: {len(missing)} broken links")
        return 1
    print("Link check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
