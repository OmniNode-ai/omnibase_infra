"""Architecture compliance tests for ONEX infrastructure.

These tests verify that architectural boundaries are maintained between
omnibase_core (pure, no I/O) and omnibase_infra (infrastructure, owns all I/O).

The core principle: omnibase_core must not contain any infrastructure-specific
imports such as kafka, httpx, or asyncpg. These belong exclusively in omnibase_infra.

Ticket: OMN-255
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class ArchitectureViolation:
    """Represents a single architecture violation.

    Attributes:
        file_path: Path to the file containing the violation.
        line_number: Line number where the violation occurs (1-indexed).
        line_content: The actual line content containing the violation.
        import_pattern: The forbidden import pattern that was matched.
    """

    file_path: Path
    line_number: int
    line_content: str
    import_pattern: str

    def __str__(self) -> str:
        """Format violation for display."""
        return f"  - {self.file_path}:{self.line_number}: {self.line_content.strip()}"


def _get_package_source_path(package_name: str) -> Path | None:
    """Locate the source directory for an installed package.

    Uses importlib to find the package spec and extract the source path.

    Args:
        package_name: The name of the package to locate.

    Returns:
        Path to the package source directory, or None if not found.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return None

    # spec.origin is the __init__.py path
    origin_path = Path(spec.origin)
    return origin_path.parent


def _find_python_files(directory: Path) -> list[Path]:
    """Find all Python files in a directory recursively.

    Args:
        directory: Root directory to search.

    Returns:
        List of paths to Python files (*.py).
    """
    if not directory.exists():
        return []
    return list(directory.rglob("*.py"))


def _is_requirements_file(file_path: Path) -> bool:
    """Check if a file is a requirements or configuration file.

    These files are allowed to mention infrastructure packages
    as dependencies.

    Args:
        file_path: Path to check.

    Returns:
        True if the file is a requirements/config file.
    """
    excluded_patterns = {
        "requirements",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
    }
    file_name = file_path.name.lower()
    return any(pattern in file_name for pattern in excluded_patterns)


def _scan_file_for_imports(
    file_path: Path,
    forbidden_patterns: list[str],
) -> list[ArchitectureViolation]:
    """Scan a Python file for forbidden import patterns.

    Detects both `import X` and `from X import Y` patterns.
    Properly handles multiline docstrings (both triple-quoted variants).

    Args:
        file_path: Path to the Python file to scan.
        forbidden_patterns: List of import patterns to detect.

    Returns:
        List of violations found in the file.
    """
    violations: list[ArchitectureViolation] = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Skip files that can't be read
        return violations

    lines = content.splitlines()

    for pattern in forbidden_patterns:
        # Regex patterns for detecting imports:
        # 1. `import kafka` or `import kafka.something`
        # 2. `from kafka import ...` or `from kafka.something import ...`
        import_regex = re.compile(
            rf"^\s*(import\s+{re.escape(pattern)}(?:\.\w+)*"
            rf"|from\s+{re.escape(pattern)}(?:\.\w+)*\s+import)",
            re.MULTILINE,
        )

        # Track multiline string state for this pattern scan
        in_multiline_string = False
        multiline_delimiter: str | None = None

        for line_num, line in enumerate(lines, start=1):
            stripped = line.lstrip()

            # Skip comment lines (only if not in multiline string)
            if not in_multiline_string and stripped.startswith("#"):
                continue

            # Handle multiline string tracking
            # Check for docstring delimiters (""" or ''')
            # Also handles raw strings (r""", r''') since we look for the quotes
            for delimiter in ('"""', "'''"):
                count = line.count(delimiter)
                if count > 0:
                    if not in_multiline_string:
                        # Not currently in a multiline string
                        if count >= 2:
                            # Single-line string (opens and closes on same line)
                            # Don't change state, but skip this line for import checking
                            pass
                        else:
                            # Opening a multiline string (odd count = 1)
                            in_multiline_string = True
                            multiline_delimiter = delimiter
                    elif delimiter == multiline_delimiter:
                        # We're in a multiline string with this delimiter
                        # Odd count means we're closing it
                        if count % 2 == 1:
                            in_multiline_string = False
                            multiline_delimiter = None

            # Skip lines inside multiline strings (docstrings)
            if in_multiline_string:
                continue

            # Skip lines that are single-line docstrings
            # (check after multiline handling to avoid double-processing)
            if stripped.startswith(('"""', "'''", 'r"""', "r'''")):
                # Check if it's a complete single-line string
                for delimiter in ('"""', "'''"):
                    if delimiter in stripped:
                        # Count occurrences after the first one
                        first_pos = stripped.find(delimiter)
                        rest = stripped[first_pos + 3 :]
                        if delimiter in rest:
                            # Single-line docstring, skip
                            continue

            if import_regex.match(line):
                violations.append(
                    ArchitectureViolation(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line,
                        import_pattern=pattern,
                    )
                )

    return violations


def _scan_package_for_forbidden_imports(
    package_name: str,
    forbidden_patterns: list[str],
    skip_requirements: bool = True,
) -> list[ArchitectureViolation]:
    """Scan an entire package for forbidden import patterns.

    Args:
        package_name: Name of the package to scan.
        forbidden_patterns: List of import patterns to detect.
        skip_requirements: If True, skip requirements/config files.

    Returns:
        List of all violations found in the package.

    Raises:
        ValueError: If the package cannot be located.
    """
    package_path = _get_package_source_path(package_name)
    if package_path is None:
        raise ValueError(f"Could not locate package: {package_name}")

    python_files = _find_python_files(package_path)
    all_violations: list[ArchitectureViolation] = []

    for file_path in python_files:
        if skip_requirements and _is_requirements_file(file_path):
            continue

        violations = _scan_file_for_imports(file_path, forbidden_patterns)
        all_violations.extend(violations)

    return all_violations


def _format_violation_report(
    violations: list[ArchitectureViolation],
    import_pattern: str,
    package_name: str,
) -> str:
    """Format a violation report with clear, actionable messages.

    Args:
        violations: List of violations to report.
        import_pattern: The forbidden import pattern.
        package_name: Name of the package being scanned.

    Returns:
        Formatted error message string.
    """
    lines = [
        f"ARCHITECTURE VIOLATION: Found '{import_pattern}' import in {package_name}",
        "",
        "Violations found:",
    ]

    for violation in violations:
        lines.append(str(violation))

    lines.extend(
        [
            "",
            f"{package_name} must not contain infrastructure dependencies.",
            "Move these to omnibase_infra.",
        ]
    )

    return "\n".join(lines)


class TestArchitectureCompliance:
    """Verify architectural invariants are maintained.

    These tests enforce the separation between omnibase_core (pure, no I/O)
    and omnibase_infra (infrastructure, owns all I/O). The core package
    should never import infrastructure-specific libraries directly.
    """

    CORE_PACKAGE = "omnibase_core"

    def test_no_kafka_in_core(self) -> None:
        """Core should not import kafka (infra dependency).

        Kafka is an infrastructure concern for event streaming. All kafka
        usage must be in omnibase_infra, never in omnibase_core.
        """
        pattern = "kafka"
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            [pattern],
        )

        filtered = [v for v in violations if v.import_pattern == pattern]
        if filtered:
            pytest.fail(_format_violation_report(filtered, pattern, self.CORE_PACKAGE))

    def test_no_httpx_in_core(self) -> None:
        """Core should not import httpx (infra dependency).

        HTTP client libraries are infrastructure concerns. All HTTP
        operations must be in omnibase_infra, never in omnibase_core.
        """
        pattern = "httpx"
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            [pattern],
        )

        filtered = [v for v in violations if v.import_pattern == pattern]
        if filtered:
            pytest.fail(_format_violation_report(filtered, pattern, self.CORE_PACKAGE))

    def test_no_asyncpg_in_core(self) -> None:
        """Core should not import asyncpg (infra dependency).

        Database drivers are infrastructure concerns. All database
        operations must be in omnibase_infra, never in omnibase_core.
        """
        pattern = "asyncpg"
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            [pattern],
        )

        filtered = [v for v in violations if v.import_pattern == pattern]
        if filtered:
            pytest.fail(_format_violation_report(filtered, pattern, self.CORE_PACKAGE))

    @pytest.mark.xfail(
        reason="Known issue: aiohttp in omnibase_core tracked in OMN-1015",
        strict=False,
    )
    def test_no_aiohttp_in_core(self) -> None:
        """Core should not import aiohttp (infra dependency).

        Async HTTP libraries are infrastructure concerns. All HTTP
        operations must be in omnibase_infra, never in omnibase_core.

        Note: Currently xfail due to OMN-1015. Remove xfail marker when
        omnibase_core removes aiohttp dependency.
        """
        pattern = "aiohttp"
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            [pattern],
        )

        filtered = [v for v in violations if v.import_pattern == pattern]
        if filtered:
            pytest.fail(_format_violation_report(filtered, pattern, self.CORE_PACKAGE))

    @pytest.mark.xfail(
        reason="Known issue: redis in omnibase_core tracked in OMN-1295",
        strict=False,
    )
    def test_no_redis_in_core(self) -> None:
        """Core should not import redis (infra dependency).

        Redis client libraries are infrastructure concerns. All cache
        operations must be in omnibase_infra, never in omnibase_core.

        Note: Currently xfail due to OMN-1295. Remove xfail marker when
        omnibase_core removes redis dependency.
        """
        pattern = "redis"
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            [pattern],
        )

        filtered = [v for v in violations if v.import_pattern == pattern]
        if filtered:
            pytest.fail(_format_violation_report(filtered, pattern, self.CORE_PACKAGE))

    def test_no_psycopg_in_core(self) -> None:
        """Core should not import psycopg (infra dependency).

        PostgreSQL drivers are infrastructure concerns. All database
        operations must be in omnibase_infra, never in omnibase_core.
        """
        # Check both psycopg2 and psycopg3 (psycopg)
        patterns = ["psycopg", "psycopg2"]
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            patterns,
        )

        if violations:
            # Group by pattern for reporting
            report_lines = []
            for pattern in patterns:
                filtered = [v for v in violations if v.import_pattern == pattern]
                if filtered:
                    report_lines.append(
                        _format_violation_report(filtered, pattern, self.CORE_PACKAGE)
                    )
            pytest.fail("\n\n".join(report_lines))

    def test_core_package_exists(self) -> None:
        """Verify omnibase_core package can be located.

        This is a sanity check to ensure the package under test exists
        and can be found by importlib. If this fails, other tests in
        this class may give false positives.
        """
        package_path = _get_package_source_path(self.CORE_PACKAGE)
        assert package_path is not None, (
            f"Could not locate {self.CORE_PACKAGE} package. "
            "Ensure it is installed in the current environment."
        )
        assert package_path.exists(), (
            f"{self.CORE_PACKAGE} package path does not exist: {package_path}"
        )

    def test_comprehensive_infra_scan(self) -> None:
        """Comprehensive scan for all infrastructure imports in core.

        This is a catch-all test that checks for multiple infrastructure
        patterns in a single pass. Use this to quickly verify that no
        infrastructure dependencies have leaked into core.
        """
        forbidden_patterns = [
            "kafka",
            "httpx",
            "asyncpg",
            # Note: aiohttp excluded - has dedicated xfail test (OMN-1015)
            # Note: redis excluded - has dedicated xfail test (OMN-1295)
            "psycopg",
            "psycopg2",
            "consul",
            "hvac",  # Vault client
            "aiokafka",
            "confluent_kafka",
        ]

        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            forbidden_patterns,
        )

        if violations:
            # Group violations by pattern
            by_pattern: dict[str, list[ArchitectureViolation]] = {}
            for v in violations:
                by_pattern.setdefault(v.import_pattern, []).append(v)

            report_lines = [
                "ARCHITECTURE VIOLATIONS DETECTED",
                "",
                f"Found {len(violations)} violation(s) in {self.CORE_PACKAGE}:",
                "",
            ]

            for pattern, pattern_violations in sorted(by_pattern.items()):
                report_lines.append(
                    f"Pattern '{pattern}' ({len(pattern_violations)} violations):"
                )
                for v in pattern_violations:
                    report_lines.append(str(v))
                report_lines.append("")

            report_lines.extend(
                [
                    f"{self.CORE_PACKAGE} must not contain infrastructure dependencies.",
                    "Move these imports to omnibase_infra.",
                ]
            )

            pytest.fail("\n".join(report_lines))
