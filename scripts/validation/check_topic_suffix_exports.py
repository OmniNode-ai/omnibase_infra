#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate that all SUFFIX_* constants in platform_topic_suffixes.py are
re-exported from topics/__init__.py.

This CI check prevents the recurring failure where a new SUFFIX_* constant is
added to platform_topic_suffixes.py but not imported/exported in __init__.py,
breaking downstream consumers that import from ``omnibase_infra.topics``.

Usage:
    python scripts/validation/check_topic_suffix_exports.py

Exit codes:
    0  all SUFFIX_* constants are exported
    1  one or more SUFFIX_* constants are missing from __init__.py
    2  runtime error

NOTE: stdlib only — no third-party dependencies.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _extract_suffix_names(source: str) -> set[str]:
    """Extract all module-level SUFFIX_* variable names from source."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("SUFFIX_"):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.startswith("SUFFIX_"):
                names.add(node.target.id)
    return names


def _extract_imported_names(source: str) -> set[str]:
    """Extract all names imported in ``from ... import (...)`` statements."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                real = alias.asname if alias.asname else alias.name
                if real.startswith("SUFFIX_"):
                    names.add(real)
    return names


def _extract_all_list(source: str) -> set[str]:
    """Extract SUFFIX_* names from the ``__all__`` list.

    Handles both ``__all__ = [...]`` (Assign) and ``__all__: list[str] = [...]``
    (AnnAssign) forms.
    """
    tree = ast.parse(source)
    names: set[str] = set()

    def _collect_from_list(value: ast.expr) -> None:
        if isinstance(value, ast.List):
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if elt.value.startswith("SUFFIX_"):
                        names.add(elt.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    _collect_from_list(node.value)
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
                and node.value is not None
            ):
                _collect_from_list(node.value)
    return names


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    suffixes_file = (
        repo_root / "src" / "omnibase_infra" / "topics" / "platform_topic_suffixes.py"
    )
    init_file = repo_root / "src" / "omnibase_infra" / "topics" / "__init__.py"

    if not suffixes_file.exists():
        print(f"ERROR: {suffixes_file} not found", file=sys.stderr)
        return 2
    if not init_file.exists():
        print(f"ERROR: {init_file} not found", file=sys.stderr)
        return 2

    # Get all SUFFIX_* defined in platform_topic_suffixes.py
    defined = _extract_suffix_names(suffixes_file.read_text(encoding="utf-8"))

    # Get all SUFFIX_* imported + listed in __all__ in __init__.py
    init_source = init_file.read_text(encoding="utf-8")
    imported = _extract_imported_names(init_source)
    in_all = _extract_all_list(init_source)

    # A constant is properly exported if it is both imported AND in __all__
    missing_import = defined - imported
    missing_all = (defined & imported) - in_all

    errors: list[str] = []
    if missing_import:
        errors.append(
            f"{len(missing_import)} SUFFIX_* constant(s) not imported in __init__.py:\n"
            + "\n".join(f"  - {s}" for s in sorted(missing_import))
        )
    if missing_all:
        errors.append(
            f"{len(missing_all)} SUFFIX_* constant(s) imported but missing from __all__:\n"
            + "\n".join(f"  - {s}" for s in sorted(missing_all))
        )

    if errors:
        print("FAIL: Topic suffix export validation\n")
        for err in errors:
            print(err)
        print(
            "\nFix: Add missing SUFFIX_* constants to the import block and __all__ "
            "list in src/omnibase_infra/topics/__init__.py"
        )
        return 1

    print(f"PASS: All {len(defined)} SUFFIX_* constants are properly exported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
