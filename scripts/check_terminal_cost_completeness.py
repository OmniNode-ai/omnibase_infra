#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fresh-deploy fitness gate: terminal cost completeness (OMN-13412).

A terminal LLM-call-completed / delegation-completed event that reports a
hardcoded ``cost_usd=0.0`` silently clobbers the real token-derived cost on the
success path. The cost projection then reads ``$0.00`` for a paid call, the
savings estimate is overstated, and the no-provenance gap stays invisible. This
is the static side of the OMN-13408 token-clobber work: this gate forbids NEW
hardcoded ``cost_usd=0.0`` literals on terminal *success* emit paths.

What it scans
-------------
Python source under ``src/`` for a literal ``cost_usd=0.0`` (or
``cost_usd: 0.0`` mapping form, or ``"cost_usd": 0.0``) assignment.

A hardcoded zero is LEGITIMATE on two kinds of path and is allowed via an inline
annotation rather than a separate allowlist file (Rule 7a: no allowlists that
launder debt; the annotation is co-located with the code so it is reviewed in
the same diff):

  * a genuinely-zero local-model / budget-rejection / error path, annotated with
    a trailing ``# cost-zero-ok: <reason>`` comment on the same logical line, OR
  * the immediately-following line carrying the annotation when the literal sits
    inside a multi-line constructor call.

The annotation makes the human decision explicit and greppable. Any other
hardcoded ``cost_usd=0.0`` fails the gate.

The currently-known legitimate occurrences (error path, budget-rejection path,
local-model default) are annotated by this PR. OMN-13408's in-flight token-clobber
fix owns the *computation* on the success path; this gate guards against
regressions and does not modify any cost computation.

Usage::

    uv run python scripts/check_terminal_cost_completeness.py
    uv run python scripts/check_terminal_cost_completeness.py --root src
    uv run python scripts/check_terminal_cost_completeness.py path/to/file.py ...

Exit codes:
    0 — no un-annotated hardcoded cost_usd=0.0 found
    1 — one or more un-annotated hardcoded zero-cost terminal literals found
    2 — configuration error (root missing)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "src"

# Matches a hardcoded zero cost in any of the constructor / mapping forms:
#   cost_usd=0.0   cost_usd =0.0   "cost_usd": 0.0   'cost_usd': 0.0   cost_usd: 0.0
# The fractional-zero variants (0.0, 0.00, 0.0e0) and a bare integer 0 are caught;
# a non-zero literal or an expression (e.g. self._estimate_cost(...)) is NOT a match.
_ZERO_COST_RE = re.compile(
    r"""(?<![\w.])(?P<key>['"]?cost_usd['"]?)\s*[:=]\s*"""
    r"""(?P<val>0(?:\.0+)?(?:[eE][+-]?\d+)?)\b"""
)

# Inline human decision marker. Co-located with the literal (same line) OR on the
# line immediately preceding the literal (multi-line constructor).
_ANNOTATION_RE = re.compile(r"#\s*cost-zero-ok:\s*\S")


def _is_annotated(lines: list[str], idx: int) -> bool:
    """Return True if the literal on line ``idx`` (0-based) is annotated."""
    if _ANNOTATION_RE.search(lines[idx]):
        return True
    # Allow the annotation on the immediately-preceding line for multi-line calls.
    if idx > 0 and _ANNOTATION_RE.search(lines[idx - 1]):
        return True
    return False


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, line)] of un-annotated hardcoded cost_usd=0.0 literals."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    findings: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Skip comment-only lines and docstring-ish prose lines.
        if stripped.startswith("#"):
            continue
        m = _ZERO_COST_RE.search(line)
        if m is None:
            continue
        # Only the non-zero suppression matters; a real value never matches the regex.
        if _is_annotated(lines, i):
            continue
        findings.append((i + 1, line.strip()))
    return findings


def _iter_targets(root: Path, explicit: list[Path]) -> list[Path]:
    if explicit:
        return [p for p in explicit if p.suffix == ".py" and p.is_file()]
    return sorted(root.rglob("*.py"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Explicit files to scan (pre-commit passes staged files). "
        "If omitted, scans --root recursively.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Directory to scan when no explicit files are given (default: src/).",
    )
    args = parser.parse_args(argv)

    if not args.files and not args.root.exists():
        print(f"ERROR: scan root not found: {args.root}", file=sys.stderr)
        return 2

    targets = _iter_targets(args.root, args.files)
    violations: list[tuple[Path, int, str]] = []
    for path in targets:
        for lineno, line in scan_file(path):
            violations.append((path, lineno, line))

    if violations:
        print(
            "FAIL: hardcoded cost_usd=0.0 on a terminal cost path "
            "(OMN-13412 terminal-cost-completeness gate).",
            file=sys.stderr,
        )
        print(
            "A hardcoded zero clobbers token-derived cost on the success path. "
            "If the zero is genuinely correct (local model / budget-rejection / "
            "error path), add a trailing or preceding "
            "'# cost-zero-ok: <reason>' annotation on that line.",
            file=sys.stderr,
        )
        for path, lineno, line in violations:
            rel = (
                path.relative_to(_REPO_ROOT)
                if path.is_relative_to(_REPO_ROOT)
                else path
            )
            print(f"  {rel}:{lineno}: {line}", file=sys.stderr)
        return 1

    print(f"OK: no un-annotated hardcoded cost_usd=0.0 in {len(targets)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
