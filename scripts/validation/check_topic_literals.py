#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
#
# Topic Literal Check (OMN-3343).
#
# Scans src/**/*.py for raw topic literal strings of the form
# `onex.(evt|cmd).<producer>.<event>.v<n>` outside the canonical topic
# definition files. All topic references in production code must go through
# the canonical topics.py / topic constants instead.
#
# Exclusions:
#   - tests/, fixtures/, docs/            (not production code)
#   - generated/                          (auto-generated enum files)
#   - topics.py, topic_constants.py       (canonical topic definition files)
#   - platform_topic_suffixes.py          (canonical platform topic registry)
#   - contract.yaml                       (contract definitions, not Python)
#
# Pre-existing violations that were present before OMN-3343 are suppressed
# via `scripts/validation/topic_literal_baseline.txt`. New violations added
# after that baseline are blocked.
#
# Usage:
#   uv run python scripts/validation/check_topic_literals.py
#   uv run python scripts/validation/check_topic_literals.py src/
#   uv run python scripts/validation/check_topic_literals.py --baseline scripts/validation/topic_literal_baseline.txt
#
# Exit codes:
#   0  no new violations (suppressed violations are logged with SUPPRESSED)
#   1  one or more new violations found
#

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Pattern: full topic literal of the form onex.(evt|cmd).<producer>.<event>.v<n>
# Only match strings that START with the ONEX prefix (no env prefix like dev./prod.)
_TOPIC_PATTERN = re.compile(r"^onex\.(evt|cmd)\.[a-z][a-z0-9._-]*$")

# Files that ARE canonical topic definitions — raw literals are legitimate here
_EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "topics.py",
        "topic_constants.py",
        "platform_topic_suffixes.py",
        "contract.yaml",
    }
)

# Directory segments to skip — these are not production code
_EXCLUDED_DIR_SEGMENTS: frozenset[str] = frozenset(
    {
        "tests",
        "fixtures",
        "docs",
        "generated",
    }
)


# ---------------------------------------------------------------------------
# AST-based literal detection
# ---------------------------------------------------------------------------


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Return a mapping from node id → parent node."""
    parent_map: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[id(child)] = node
    return parent_map


def _is_docstring(
    node: ast.Constant,
    parent_map: dict[int, ast.AST],
) -> bool:
    """Return True if *node* is the docstring of its parent scope."""
    parent = parent_map.get(id(node))
    if not isinstance(parent, ast.Expr):  # docstrings are always bare Expr nodes
        return False
    grandparent = parent_map.get(id(parent))
    if not isinstance(
        grandparent,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module),
    ):
        return False
    # The bare Expr must be the FIRST statement in the scope body
    return bool(
        hasattr(grandparent, "body")
        and grandparent.body
        and grandparent.body[0] is parent
    )


def find_topic_literals(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, topic_value) for each raw topic literal in *path*.

    Docstrings are excluded because they document topic names rather than
    using them as runtime values.
    """
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        # Treat parse failures as a single violation so CI is clearly notified
        return [(0, f"SyntaxError: {exc}")]

    parent_map = _build_parent_map(tree)
    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if not isinstance(node.value, str):
            continue
        val = node.value.strip()
        if not _TOPIC_PATTERN.match(val):
            continue
        if _is_docstring(node, parent_map):
            continue
        hits.append((node.lineno, val))

    return hits


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _should_skip(path: Path, src_root: Path) -> bool:
    """Return True if *path* should be excluded from the scan."""
    if path.name in _EXCLUDED_FILENAMES:
        return True
    try:
        rel_parts = set(path.relative_to(src_root).parts)
    except ValueError:
        rel_parts = set(path.parts)
    return bool(_EXCLUDED_DIR_SEGMENTS & rel_parts)


def collect_files(src_root: Path) -> list[Path]:
    """Return sorted list of Python files to scan under *src_root*."""
    return sorted(p for p in src_root.rglob("*.py") if not _should_skip(p, src_root))


# ---------------------------------------------------------------------------
# Baseline (suppression list)
# ---------------------------------------------------------------------------


def load_baseline(baseline_path: Path) -> frozenset[str]:
    """Load suppressed violations from *baseline_path*.

    Each non-comment, non-blank line must be of the form ``path:lineno``.
    Paths are stored as-is (relative to repo root) for comparison.
    """
    if not baseline_path.exists():
        return frozenset()
    suppressed: set[str] = set()
    for raw in baseline_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        suppressed.add(line)
    return frozenset(suppressed)


def violation_key(path: Path, lineno: int, repo_root: Path) -> str:
    """Return the suppression key for a violation.

    Format: ``<repo_root_relative_path>:<lineno>``
    e.g.    ``omnibase_infra/src/omnibase_infra/cli/foo.py:42``
    """
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path
    return f"{rel}:{lineno}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(src_dir: Path | None = None, baseline_path: Path | None = None) -> int:
    """Scan *src_dir* for raw topic literals; suppress known-good violations.

    Args:
        src_dir: Path to the ``src/`` directory to scan.  When *None*, the
                 function auto-discovers it by walking up from the script.
        baseline_path: Path to the suppression list.  When *None*, defaults
                       to ``scripts/validation/topic_literal_baseline.txt``
                       next to the project root.

    Returns:
        0 when no new violations are found, 1 otherwise.
    """
    # --- auto-discover project root -----------------------------------------
    if src_dir is None:
        root = Path(__file__).resolve().parent
        for _ in range(10):
            if (root / "pyproject.toml").exists() or (root / "src").exists():
                break
            root = root.parent
        src_dir = root / "src"

    repo_root = src_dir.parent  # src/ lives directly under repo root

    if not src_dir.exists():
        print(f"WARNING: src/ directory not found at {src_dir}; skipping check")
        return 0

    # --- load baseline -------------------------------------------------------
    if baseline_path is None:
        baseline_path = (
            repo_root / "scripts" / "validation" / "topic_literal_baseline.txt"
        )
    suppressed = load_baseline(baseline_path)

    # --- scan ----------------------------------------------------------------
    files = collect_files(src_dir)
    new_violations: list[str] = []
    suppressed_count = 0

    for path in files:
        hits = find_topic_literals(path)
        for lineno, topic_value in hits:
            key = violation_key(path, lineno, repo_root)
            if key in suppressed:
                suppressed_count += 1
            else:
                new_violations.append(
                    f"{path}:{lineno}: raw topic literal: {topic_value!r}"
                )

    # --- report --------------------------------------------------------------
    if not new_violations:
        print(
            f"OK: No new raw topic literals found in {len(files)} file(s)"
            f" ({suppressed_count} pre-existing violations suppressed)"
        )
        return 0

    print(
        f"FAIL: Found {len(new_violations)} new raw topic literal(s)"
        f" (outside the suppression baseline):\n"
    )
    for msg in sorted(new_violations):
        print(f"  {msg}")
    print(
        f"\nRaw topic literals bypass the contract system and create invisible coupling."
        f"\nAll topic references must go through the canonical topics.py constants."
        f"\nTo suppress a pre-existing violation, add it to:"
        f"\n  {baseline_path}"
        f"\nDO NOT add new violations to the baseline; fix them instead."
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scan src/ for raw ONEX topic literal strings (OMN-3343)."
    )
    parser.add_argument(
        "src_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the src/ directory (auto-discovered if omitted).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        dest="baseline_path",
        help="Path to suppression baseline file.",
    )
    args = parser.parse_args()

    target: Path | None = args.src_dir
    if target is not None and not target.exists():
        print(f"ERROR: path does not exist: {target}", file=sys.stderr)
        sys.exit(1)
    if target is not None:
        candidate = target / "src"
        if candidate.is_dir():
            target = candidate

    sys.exit(main(src_dir=target, baseline_path=args.baseline_path))
