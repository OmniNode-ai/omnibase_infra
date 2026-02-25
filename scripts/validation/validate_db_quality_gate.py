#!/usr/bin/env python3
"""CI quality gate to forbid new domain-specific DB adapters.

Enforces the contract-driven database architecture by detecting and blocking:

1. **Adapter class patterns** - Classes matching ``*Adapter*Postgres`` or
   ``*Postgres*Adapter`` in non-infra code indicate domain-specific DB coupling.
2. **Direct SQL in domain code** - Raw SQL keywords (``SELECT ... FROM``,
   ``INSERT``, ``UPDATE``, ``DELETE ... FROM``) outside of contracts and infra
   layers indicate schema coupling.
3. **Direct connection calls** - ``psycopg.connect`` / ``asyncpg.connect`` usage
   outside infra indicates bypass of the connection pool and circuit breaker.

Escape hatches:
    - ``# db-adapter-ok`` comment on the same line suppresses adapter class checks.
    - ``# sql-ok`` comment on the same line suppresses direct SQL checks.
    - Files under ``omnibase_infra/`` are exempt (infra owns DB access).
    - Test files (under ``tests/``) are exempt.

Usage:
    python scripts/validation/validate_db_quality_gate.py
    python scripts/validation/validate_db_quality_gate.py --verbose
    python scripts/validation/validate_db_quality_gate.py --check-paths src/omniclaude src/omnimemory

Exit Codes:
    0 - No violations found.
    1 - Violations detected.

Ticket: OMN-1785
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Forbidden patterns
# ---------------------------------------------------------------------------

# Adapter class patterns that indicate domain-specific DB coupling
_ADAPTER_CLASS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"class\s+\w*Adapter\w*Postgres"),
    re.compile(r"class\s+\w*Postgres\w*Adapter"),
    re.compile(r"class\s+\w*Adapter\w*Asyncpg"),
    re.compile(r"class\s+\w*Asyncpg\w*Adapter"),
]

# Direct SQL patterns (case-insensitive)
_DIRECT_SQL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(SELECT)\s+.*\bFROM\b", re.IGNORECASE),
    re.compile(r"\b(INSERT)\s+INTO\b", re.IGNORECASE),
    re.compile(r"\b(UPDATE)\s+\w+\s+SET\b", re.IGNORECASE),
    re.compile(r"\b(DELETE)\s+FROM\b", re.IGNORECASE),
]

# Direct DB connection patterns
_DIRECT_CONNECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"psycopg[23]?\.connect"),
    re.compile(r"asyncpg\.connect"),
    re.compile(r"psycopg[23]?\.AsyncConnection\.connect"),
]

# Escape-hatch comment markers
_ADAPTER_ESCAPE = "# db-adapter-ok"
_SQL_ESCAPE = "# sql-ok"

# Directories that are exempt from all checks (infra owns DB access)
_EXEMPT_DIR_SEGMENTS: frozenset[str] = frozenset(
    {
        "omnibase_infra",
        "tests",
        "test",
        "__pycache__",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "migrations",
    }
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DbQualityViolation:
    """A single DB quality gate violation.

    Attributes:
        file_path: Absolute or relative path to the offending file.
        line_number: 1-based line number.
        category: One of ``adapter_class``, ``direct_sql``, ``direct_connect``.
        matched_text: The portion of the line that matched.
        message: Human-readable remediation guidance.
    """

    file_path: str
    line_number: int
    category: str
    matched_text: str
    message: str

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}: [{self.category}] {self.message}"


@dataclass
class DbQualityGateResult:
    """Aggregated result of the DB quality gate scan.

    Attributes:
        violations: All violations discovered.
        files_checked: Total Python files scanned.
        files_skipped: Files skipped due to exemptions.
    """

    violations: list[DbQualityViolation] = field(default_factory=list)
    files_checked: int = 0
    files_skipped: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.violations) == 0

    @property
    def adapter_violations(self) -> list[DbQualityViolation]:
        return [v for v in self.violations if v.category == "adapter_class"]

    @property
    def sql_violations(self) -> list[DbQualityViolation]:
        return [v for v in self.violations if v.category == "direct_sql"]

    @property
    def connect_violations(self) -> list[DbQualityViolation]:
        return [v for v in self.violations if v.category == "direct_connect"]


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------


def _is_exempt(file_path: Path) -> bool:
    """Check whether a file resides in an exempt directory."""
    return any(part in _EXEMPT_DIR_SEGMENTS for part in file_path.parts)


def _is_in_string_or_comment(line: str, match_start: int) -> bool:
    """Rough heuristic: skip matches inside triple-quoted strings or comments.

    This is intentionally simple -- it catches the common cases of
    ``# comment`` and ``\"\"\"docstring lines\"\"\"`` without requiring a full
    AST parse of every file.
    """
    stripped = line.lstrip()
    # Entire line is a comment
    if stripped.startswith("#"):
        return True
    # Match is after a ``#`` on the same line (inline comment containing the pattern)
    hash_pos = line.find("#")
    if 0 <= hash_pos < match_start:
        return True
    return False


def scan_file(file_path: Path) -> list[DbQualityViolation]:
    """Scan a single Python file for DB quality gate violations.

    Args:
        file_path: Path to the Python file.

    Returns:
        List of violations found in the file.
    """
    violations: list[DbQualityViolation] = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return violations

    lines = content.splitlines()

    for line_no_0, line in enumerate(lines):
        line_no = line_no_0 + 1

        # --- Adapter class checks ---
        if _ADAPTER_ESCAPE not in line:
            for pattern in _ADAPTER_CLASS_PATTERNS:
                m = pattern.search(line)
                if m and not _is_in_string_or_comment(line, m.start()):
                    violations.append(
                        DbQualityViolation(
                            file_path=str(file_path),
                            line_number=line_no,
                            category="adapter_class",
                            matched_text=m.group(0),
                            message=(
                                "Domain-specific DB adapter class is forbidden. "
                                "Use repository contracts instead. "
                                "Add '# db-adapter-ok' to suppress if intentional."
                            ),
                        )
                    )

        # --- Direct SQL checks ---
        if _SQL_ESCAPE not in line:
            for pattern in _DIRECT_SQL_PATTERNS:
                m = pattern.search(line)
                if m and not _is_in_string_or_comment(line, m.start()):
                    violations.append(
                        DbQualityViolation(
                            file_path=str(file_path),
                            line_number=line_no,
                            category="direct_sql",
                            matched_text=m.group(0),
                            message=(
                                "Direct SQL in domain code is forbidden. "
                                "Use repository contracts instead. "
                                "Add '# sql-ok' to suppress if intentional."
                            ),
                        )
                    )

        # --- Direct connect checks ---
        if _SQL_ESCAPE not in line and _ADAPTER_ESCAPE not in line:
            for pattern in _DIRECT_CONNECT_PATTERNS:
                m = pattern.search(line)
                if m and not _is_in_string_or_comment(line, m.start()):
                    violations.append(
                        DbQualityViolation(
                            file_path=str(file_path),
                            line_number=line_no,
                            category="direct_connect",
                            matched_text=m.group(0),
                            message=(
                                "Direct DB connection is forbidden outside infra. "
                                "Use the connection pool via omnibase_infra. "
                                "Add '# db-adapter-ok' to suppress if intentional."
                            ),
                        )
                    )

    return violations


def validate_db_quality_gate(
    check_paths: list[Path] | None = None,
    *,
    verbose: bool = False,
) -> DbQualityGateResult:
    """Run the DB quality gate scan across one or more source trees.

    Args:
        check_paths: Directories to scan. Defaults to ``[Path("src/")]`` which
            relies on the exemption logic to skip ``omnibase_infra/`` and
            ``tests/`` automatically.
        verbose: If True, log each file being scanned.

    Returns:
        Aggregated scan result.
    """
    if check_paths is None:
        check_paths = [Path("src/")]

    result = DbQualityGateResult()

    for root in check_paths:
        if not root.exists():
            if verbose:
                print(f"  Skipping non-existent path: {root}")
            continue

        for py_file in sorted(root.rglob("*.py")):
            if _is_exempt(py_file):
                result.files_skipped += 1
                continue

            result.files_checked += 1
            if verbose:
                print(f"  Scanning: {py_file}")

            file_violations = scan_file(py_file)
            result.violations.extend(file_violations)

    return result


def generate_report(result: DbQualityGateResult) -> str:
    """Generate a human-readable report from the scan result.

    Args:
        result: The scan result to report on.

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("DB Quality Gate (OMN-1785)")
    lines.append("=" * 60)

    if result.is_valid:
        lines.append(f"PASS - {result.files_checked} files checked, 0 violations")
    else:
        lines.append(
            f"FAIL - {len(result.violations)} violation(s) "
            f"in {result.files_checked} files"
        )
        lines.append("")

        if result.adapter_violations:
            lines.append(
                f"  Adapter class violations ({len(result.adapter_violations)}):"
            )
            for v in result.adapter_violations:
                lines.append(f"    {v}")
            lines.append("")

        if result.sql_violations:
            lines.append(f"  Direct SQL violations ({len(result.sql_violations)}):")
            for v in result.sql_violations:
                lines.append(f"    {v}")
            lines.append("")

        if result.connect_violations:
            lines.append(
                f"  Direct connect violations ({len(result.connect_violations)}):"
            )
            for v in result.connect_violations:
                lines.append(f"    {v}")
            lines.append("")

        lines.append("Remediation:")
        lines.append("  - Use repository contracts for data access")
        lines.append("  - DB adapters belong in omnibase_infra only")
        lines.append(
            "  - Add '# db-adapter-ok' or '# sql-ok' to suppress false positives"
        )

    lines.append(
        f"\nFiles checked: {result.files_checked}, skipped: {result.files_skipped}"
    )
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CI quality gate: forbid domain-specific DB adapters (OMN-1785)"
    )
    parser.add_argument(
        "--check-paths",
        nargs="*",
        default=None,
        help="Directories to scan (default: src/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show each file being scanned",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in args.check_paths] if args.check_paths else None
    result = validate_db_quality_gate(paths, verbose=args.verbose)
    report = generate_report(result)
    print(report)

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
