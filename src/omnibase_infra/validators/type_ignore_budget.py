# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""type-ignore budget gate — arg-type and union-attr suppressions (OMN-10822).

Counts ``# type: ignore[arg-type]`` and ``# type: ignore[union-attr]`` across
the codebase and fails if the count exceeds the configured budget.

These two suppression categories are the highest-risk mypy silencers:
  - arg-type  (120 historical) — type mismatches at call sites
  - union-attr (18 historical) — None-dereference warnings suppressed

Budget strategy: set at current count when wired (prevents new additions
without fixing an existing one). Decrement the budget in subsequent PRs as
violations are remediated. Long-term target: 0.

Escape hatch: ``# substrate-allow: <reason>`` on the same line exempts that
occurrence from the count. This makes the justification explicit and auditable.

Usage:
    uv run python -m omnibase_infra.validators.type_ignore_budget \\
        --max-violations 67 src/omnibase_infra

    # Zero tolerance (future target):
    uv run python -m omnibase_infra.validators.type_ignore_budget \\
        --max-violations 0 src/omnibase_infra
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")

# Why: Runtime wiring validates and narrows this payload shape before use.
# Matches: # type: ignore[arg-type]  OR  # type: ignore[union-attr]
# Why: Runtime wiring validates and narrows this payload shape before use.
# Allows multiple codes in the bracket: # type: ignore[arg-type, misc]
_TARGET_RE = re.compile(r"#\s*type:\s*ignore\[[^\]]*\b(arg-type|union-attr)\b[^\]]*\]")

# Exemption: # substrate-allow: <reason>  or  # ONEX_EXCLUDE: ...  or # ai-slop-ok
_ALLOW_RE = re.compile(
    r"#\s*(substrate-allow:|ONEX_EXCLUDE:|ai-slop-ok)",
    re.IGNORECASE,
)

# Initial budget when gate was added (2026-05-22). Decrement target: 0.
# Track history here to make regressions visible in git blame.
# 2026-05-22: 74 (57 arg-type + 17 union-attr) — initial baseline
DEFAULT_BUDGET = 74


@dataclass(frozen=True, slots=True)
class TypeIgnoreFinding:
    path: Path
    line: int
    code: str
    text: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: [{self.code}] {self.text.strip()}"


def scan_file(path: Path) -> list[TypeIgnoreFinding]:
    # Exclude this file itself — the docstring contains the patterns for documentation.
    if path.name == "type_ignore_budget.py" and "validators" in path.parts:
        return []

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    if "type: ignore" not in source:
        return []

    findings: list[TypeIgnoreFinding] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = _TARGET_RE.search(line)
        if m is None:
            continue
        if _ALLOW_RE.search(line):
            continue
        findings.append(
            TypeIgnoreFinding(
                path=path,
                line=lineno,
                code=m.group(1),
                text=line,
            )
        )
    return findings


def scan_paths(paths: Sequence[Path]) -> list[TypeIgnoreFinding]:
    findings: list[TypeIgnoreFinding] = []
    for path in _iter_python_files(paths):
        findings.extend(scan_file(path))
    return findings


def _iter_python_files(paths: Sequence[Path]) -> Iterator[Path]:
    scan_paths = paths or (DEFAULT_SCAN_ROOT,)
    for path in scan_paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(
                p for p in path.rglob("*.py") if "__pycache__" not in p.parts
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Budget gate for # type: ignore[arg-type] and # type: ignore[union-attr]. "
            "Fails when suppression count exceeds --max-violations."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_SCAN_ROOT],
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=DEFAULT_BUDGET,
        metavar="N",
        help=(
            f"Maximum allowed suppressions (default: {DEFAULT_BUDGET} — current baseline). "
            "Decrement as violations are fixed."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all matching lines (verbose).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else list(argv))
    findings = scan_paths(args.paths)
    count = len(findings)

    if args.list:
        for f in findings:
            sys.stdout.write(f"  {f.format()}\n")  # print-ok: CLI reporting

    arg_type_count = sum(1 for f in findings if f.code == "arg-type")
    union_attr_count = sum(1 for f in findings if f.code == "union-attr")

    sys.stderr.write(
        f"[type-ignore-budget] {count} suppressions "
        f"(arg-type: {arg_type_count}, union-attr: {union_attr_count}), "
        f"budget: {args.max_violations}\n"
    )

    if count > args.max_violations:
        sys.stderr.write(
            f"[type-ignore-budget] FAIL: {count} suppressions exceed budget {args.max_violations}.\n"
            "  Fix the type error or add '# substrate-allow: <reason>' to exempt the line.\n"
            "  Do NOT raise --max-violations without fixing an equivalent number of violations.\n"
        )
        return 1

    if count == args.max_violations and args.max_violations > 0:
        sys.stderr.write(
            f"[type-ignore-budget] WARN: at budget ceiling ({count}/{args.max_violations}). "
            "Reduce budget in .pre-commit-config.yaml as violations are fixed.\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
