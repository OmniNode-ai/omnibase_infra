#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""CI guard: EnumMessageCategory may only be defined in omnibase_core.

Usage: python scripts/check_shared_enum_ownership.py [src_root]
Exit 0: clean. Exit 1: duplicate found.
"""

import ast
import sys
from pathlib import Path

CANONICAL_ALLOWLIST = {"omnibase_core"}
GUARDED_NAMES = {"EnumMessageCategory"}


def check(root: Path) -> list[str]:
    violations = []
    for py_file in root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in GUARDED_NAMES:
                pkg = next((p for p in py_file.parts if p in CANONICAL_ALLOWLIST), None)
                if pkg is None:
                    violations.append(
                        f"DUPLICATE SHARED ENUM DETECTED\n"
                        f"  File: {py_file}\n"
                        f"  Class: {node.name} (line {node.lineno})\n"
                        f"  Fix: remove this class and import from omnibase_core instead.\n"
                        f"  Canonical owner: omnibase_core"
                    )
    return violations


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src")
    violations = check(root)
    if violations:
        print("\n".join(violations), file=sys.stderr)
        sys.exit(1)
    print("Shared enum ownership check: PASS")
