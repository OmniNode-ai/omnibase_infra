#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""
CI gate: any PR touching a writer_postgres.py or handler_*_postgres.py file
must either include a new migration file OR have a '# no-migration: <reason>'
comment in the changed file.

Usage (CI — compare against base branch):
    python scripts/check_migration_required.py --ci

Usage (pre-commit — check staged files):
    python scripts/check_migration_required.py --pre-commit
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

WRITER_PATTERNS = [
    re.compile(r"writer_postgres\.py$"),
    re.compile(r"handler_\w+_postgres\.py$"),
]

MIGRATION_DIRS = [
    "docker/migrations/forward/",
    "src/omnibase_infra/migrations/forward/",
]

BYPASS_RE = re.compile(r"#\s*no-migration\s*:", re.IGNORECASE)


def is_writer_file(path: str) -> bool:
    return any(p.search(path) for p in WRITER_PATTERNS)


def has_bypass_comment(content: str) -> bool:
    return bool(BYPASS_RE.search(content))


def _get_merge_base() -> str:
    """Return the merge-base SHA between HEAD and origin/main."""
    try:
        return subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/main"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        print(
            "WARNING: could not determine merge-base with origin/main", file=sys.stderr
        )
        return ""


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    """Remove docstring nodes from module/class/function bodies for comparison."""
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return tree


def is_cosmetic_only(path: str, merge_base: str) -> bool:
    """True if the diff for this file is comments/docstrings/whitespace only."""
    if not merge_base:
        return False

    try:
        base_src = subprocess.check_output(
            ["git", "show", f"{merge_base}:{path}"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return False  # new file

    current_src = Path(path).read_text(encoding="utf-8")

    try:
        base_tree = _strip_docstrings(ast.parse(base_src))
        current_tree = _strip_docstrings(ast.parse(current_src))
    except SyntaxError:
        return False

    return ast.dump(base_tree, include_attributes=False) == ast.dump(
        current_tree, include_attributes=False
    )


def get_changed_files(ci: bool) -> list[str]:
    if ci:
        base = subprocess.check_output(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            text=True,
        )
    else:
        base = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"],
            text=True,
        )
    return [f.strip() for f in base.splitlines() if f.strip()]


def find_violations(
    changed_files: list[str],
    merge_base: str = "",
) -> list[str]:
    """Return list of writer files that lack a corresponding migration."""
    writer_files = [f for f in changed_files if is_writer_file(f)]
    if not writer_files:
        return []

    has_migration = any(
        any(f.startswith(d) and f.endswith(".sql") for d in MIGRATION_DIRS)
        for f in changed_files
    )
    if has_migration:
        return []

    violations = []
    for wf in writer_files:
        if merge_base and is_cosmetic_only(wf, merge_base):
            continue
        path = Path(wf)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            if has_bypass_comment(content):
                continue
        violations.append(wf)

    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--pre-commit", action="store_true")
    args = parser.parse_args()

    merge_base = _get_merge_base()
    changed = get_changed_files(ci=args.ci)
    violations = find_violations(changed, merge_base=merge_base)

    if violations:
        print("ERROR: Writer file(s) changed without a migration file in this PR:")
        for v in violations:
            print(f"  {v}")
        print()
        print("Either:")
        print(
            "  1. Add a migration file to docker/migrations/forward/"
            " or src/omnibase_infra/migrations/forward/"
        )
        print(
            "  2. Add '# no-migration: <reason>' to the writer file"
            " if no schema change is needed"
        )
        sys.exit(1)

    print("OK: writer-migration coupling check passed.")


if __name__ == "__main__":
    main()
