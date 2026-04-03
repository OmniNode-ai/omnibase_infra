#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate that no localhost fallbacks exist in src/ Python files.

Scans src/**/*.py for patterns like:
  - os.environ.get("...", "localhost...")
  - default="localhost..."
  - default="http://localhost..."
  - default="bolt://localhost..."
  - default="redis://localhost..."
  - default="postgresql://localhost..."

Exits non-zero if any violations are found.
Skips test files (tests/, __tests__/).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VIOLATIONS_PATTERNS = [
    re.compile(
        r'os\.environ\.get\([^)]*,\s*"(localhost|http://localhost|bolt://localhost|redis://localhost|postgresql://localhost)'
    ),
    re.compile(
        r'default="(localhost|http://localhost|bolt://localhost|redis://localhost|postgresql://localhost)'
    ),
]

SKIP_DIRS = {"tests", "__tests__", "test", "__pycache__"}


def scan(root: Path) -> list[tuple[str, int, str]]:
    violations: list[tuple[str, int, str]] = []
    for py_file in sorted(root.rglob("*.py")):
        # Skip test directories
        if any(part in SKIP_DIRS for part in py_file.parts):
            continue
        try:
            lines = py_file.read_text().splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, start=1):
            for pattern in VIOLATIONS_PATTERNS:
                if pattern.search(line):
                    violations.append((str(py_file), lineno, line.strip()))
                    break  # one violation per line is enough
    return violations


def main() -> int:
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if not src_dir.is_dir():
        print(f"ERROR: src directory not found at {src_dir}", file=sys.stderr)
        return 1

    violations = scan(src_dir)
    if violations:
        print(f"Found {len(violations)} localhost fallback(s):\n")
        for filepath, lineno, line in violations:
            print(f"  {filepath}:{lineno}: {line}")
        print("\nAll localhost/default fallbacks must be removed from src/.")
        return 1

    print("OK: No localhost fallbacks found in src/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
