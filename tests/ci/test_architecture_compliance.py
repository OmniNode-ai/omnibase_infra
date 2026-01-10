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


def _find_first_unquoted_delimiter(line: str) -> tuple[str | None, int]:
    """Find the first triple-quote delimiter that starts a string on this line.

    This function finds the position of the first occurring triple-quote
    delimiter (either ''' or \"\"\") that would start a string, considering
    that one delimiter type may be inside a string of the other type.

    Args:
        line: The line of code to analyze.

    Returns:
        Tuple of (delimiter, position) where delimiter is the first triple-quote
        found, or (None, -1) if no triple-quote is found.
    """
    # Find positions of both delimiter types
    pos_double = line.find('"""')
    pos_single = line.find("'''")

    # Neither found
    if pos_double == -1 and pos_single == -1:
        return None, -1

    # Only one type found
    if pos_double == -1:
        return "'''", pos_single
    if pos_single == -1:
        return '"""', pos_double

    # Both found - return the one that appears first
    # (the first one is the "outer" delimiter, the other is inside the string)
    if pos_double < pos_single:
        return '"""', pos_double
    return "'''", pos_single


def _is_balanced_string_line(line: str, delimiter: str) -> bool:
    """Check if a line contains a balanced (single-line) string.

    A line is balanced if it contains an even number of the delimiter,
    meaning every opened string is closed on the same line.

    Args:
        line: The line of code to check.
        delimiter: The triple-quote delimiter to check for.

    Returns:
        True if the line contains balanced strings (or no strings).
    """
    count = line.count(delimiter)
    return count % 2 == 0


def _scan_file_for_imports(
    file_path: Path,
    forbidden_patterns: list[str],
) -> list[ArchitectureViolation]:
    """Scan a Python file for forbidden import patterns.

    Detects both `import X` and `from X import Y` patterns.
    Properly handles multiline docstrings (both triple-quoted variants),
    including edge cases where one delimiter type appears inside the other.

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

            # The portion of the line to check for imports
            # (modified when closing a multiline string to only check content after)
            line_to_check = line

            # Skip comment lines (only if not in multiline string)
            if not in_multiline_string and stripped.startswith("#"):
                continue

            # Handle multiline string tracking
            if not in_multiline_string:
                # Not in a multiline string - check if one starts on this line
                first_delim, _ = _find_first_unquoted_delimiter(line)
                if first_delim is not None:
                    # Check if this delimiter is balanced (single-line string)
                    if not _is_balanced_string_line(line, first_delim):
                        # Unbalanced - we're entering a multiline string
                        in_multiline_string = True
                        multiline_delimiter = first_delim
                    # else: balanced single-line string, continue to check imports
            else:
                # Already in a multiline string - check for closing delimiter
                assert multiline_delimiter is not None
                closing_pos = line.find(multiline_delimiter)
                if closing_pos != -1:
                    # First occurrence ALWAYS closes the multiline string we're in
                    # regardless of how many additional delimiters appear on this line
                    in_multiline_string = False
                    multiline_delimiter = None
                    # Content AFTER the closing delimiter may contain code or new strings
                    # 3 = len('"""') or len("'''")
                    after_close = line[closing_pos + 3 :]

                    # Check if the remainder starts a NEW multiline string
                    new_delim, _ = _find_first_unquoted_delimiter(after_close)
                    if new_delim is not None:
                        if not _is_balanced_string_line(after_close, new_delim):
                            # Unbalanced - entering a new multiline string
                            in_multiline_string = True
                            multiline_delimiter = new_delim
                            # Skip this line entirely as it's inside a new multiline
                            continue

                    line_to_check = after_close

            # Skip lines inside multiline strings (docstrings)
            if in_multiline_string:
                continue

            # Handle single-line docstrings that start the line
            # If there's content after the docstring, check that content for imports
            # Loop to handle multiple consecutive single-line strings (e.g., """a""" '''b''')
            check_stripped = line_to_check.lstrip()
            # Handle valid Python string prefixes (case insensitive):
            #   Single: r, f, b, u
            #   Combinations: rf/fr (raw f-string), rb/br (raw bytes)
            # Note: u cannot combine with other prefixes in Python 3
            prefix_pattern = r"^([rRfFbBuU]|[rR][fF]|[fF][rR]|[rR][bB]|[bB][rR])?"
            found_string = True
            while found_string:
                found_string = False
                for delimiter in ('"""', "'''"):
                    # Check if line starts with optional prefix + delimiter
                    prefix_match = re.match(
                        prefix_pattern + re.escape(delimiter), check_stripped
                    )
                    if prefix_match:
                        # Count occurrences of this delimiter
                        if _is_balanced_string_line(check_stripped, delimiter):
                            # Balanced means it's a single-line docstring
                            # Find where the second delimiter ends
                            first_pos = check_stripped.find(delimiter)
                            second_pos = check_stripped.find(delimiter, first_pos + 3)
                            if second_pos != -1:
                                # Get content after the closing delimiter
                                after_docstring = check_stripped[second_pos + 3 :]
                                if after_docstring.strip():
                                    # There's content after - continue processing
                                    check_stripped = after_docstring.lstrip()
                                    found_string = True
                                else:
                                    # Only whitespace after docstring - skip line
                                    check_stripped = ""
                                break  # Restart with updated check_stripped
            line_to_check = check_stripped

            # Skip if line_to_check is empty (was only a docstring)
            if not line_to_check.strip():
                continue

            if import_regex.match(line_to_check):
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

    @pytest.mark.parametrize(
        ("pattern", "description"),
        [
            pytest.param("kafka", "event streaming", id="no-kafka"),
            pytest.param("httpx", "HTTP client", id="no-httpx"),
            pytest.param("asyncpg", "database driver", id="no-asyncpg"),
            pytest.param(
                "aiohttp",
                "async HTTP",
                marks=pytest.mark.xfail(
                    reason="Known issue: tracked in OMN-1015", strict=False
                ),
                id="no-aiohttp",
            ),
            pytest.param(
                "redis",
                "cache",
                marks=pytest.mark.xfail(
                    reason="Known issue: tracked in OMN-1295", strict=False
                ),
                id="no-redis",
            ),
            pytest.param("psycopg", "PostgreSQL driver (v3)", id="no-psycopg"),
            pytest.param("psycopg2", "PostgreSQL driver (v2)", id="no-psycopg2"),
        ],
    )
    def test_no_infra_import_in_core(self, pattern: str, description: str) -> None:
        """Core should not import infrastructure dependencies.

        Infrastructure libraries belong in omnibase_infra, not omnibase_core.
        This test checks for forbidden import patterns.

        Args:
            pattern: The import pattern to check (e.g., 'kafka', 'httpx').
            description: Human-readable description of the dependency type.
        """
        violations = _scan_package_for_forbidden_imports(
            self.CORE_PACKAGE,
            [pattern],
        )

        filtered = [v for v in violations if v.import_pattern == pattern]
        if filtered:
            pytest.fail(_format_violation_report(filtered, pattern, self.CORE_PACKAGE))

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


class TestHelperFunctions:
    """Unit tests for helper functions used in architecture compliance checks.

    These tests verify the correct behavior of individual helper functions
    in isolation, using pytest fixtures and tmp_path for file-based tests.
    """

    # --- Tests for _is_requirements_file ---

    def test_is_requirements_file_detects_requirements_txt(
        self, tmp_path: Path
    ) -> None:
        """Verify _is_requirements_file detects requirements.txt files."""
        requirements_file = tmp_path / "requirements.txt"
        requirements_file.touch()
        assert _is_requirements_file(requirements_file) is True

    def test_is_requirements_file_detects_requirements_dev_txt(
        self, tmp_path: Path
    ) -> None:
        """Verify _is_requirements_file detects requirements-dev.txt variants."""
        requirements_file = tmp_path / "requirements-dev.txt"
        requirements_file.touch()
        assert _is_requirements_file(requirements_file) is True

    def test_is_requirements_file_detects_setup_py(self, tmp_path: Path) -> None:
        """Verify _is_requirements_file detects setup.py files."""
        setup_file = tmp_path / "setup.py"
        setup_file.touch()
        assert _is_requirements_file(setup_file) is True

    def test_is_requirements_file_detects_setup_cfg(self, tmp_path: Path) -> None:
        """Verify _is_requirements_file detects setup.cfg files."""
        setup_cfg_file = tmp_path / "setup.cfg"
        setup_cfg_file.touch()
        assert _is_requirements_file(setup_cfg_file) is True

    def test_is_requirements_file_detects_pyproject_toml(self, tmp_path: Path) -> None:
        """Verify _is_requirements_file detects pyproject.toml files."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.touch()
        assert _is_requirements_file(pyproject_file) is True

    def test_is_requirements_file_returns_false_for_regular_py(
        self, tmp_path: Path
    ) -> None:
        """Verify _is_requirements_file returns False for regular Python files."""
        regular_file = tmp_path / "my_module.py"
        regular_file.touch()
        assert _is_requirements_file(regular_file) is False

    def test_is_requirements_file_returns_false_for_test_file(
        self, tmp_path: Path
    ) -> None:
        """Verify _is_requirements_file returns False for test files."""
        test_file = tmp_path / "test_something.py"
        test_file.touch()
        assert _is_requirements_file(test_file) is False

    # --- Tests for _find_python_files ---

    def test_find_python_files_returns_empty_for_nonexistent_dir(
        self, tmp_path: Path
    ) -> None:
        """Verify _find_python_files returns empty list for nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"
        result = _find_python_files(nonexistent)
        assert result == []

    def test_find_python_files_finds_py_files(self, tmp_path: Path) -> None:
        """Verify _find_python_files finds .py files in directory."""
        py_file = tmp_path / "module.py"
        py_file.touch()
        txt_file = tmp_path / "readme.txt"
        txt_file.touch()

        result = _find_python_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "module.py"

    def test_find_python_files_finds_nested_files(self, tmp_path: Path) -> None:
        """Verify _find_python_files finds .py files in nested directories."""
        nested_dir = tmp_path / "subpackage"
        nested_dir.mkdir()
        nested_file = nested_dir / "nested_module.py"
        nested_file.touch()
        root_file = tmp_path / "root_module.py"
        root_file.touch()

        result = _find_python_files(tmp_path)
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"root_module.py", "nested_module.py"}

    # --- Tests for _scan_file_for_imports ---

    def test_scan_file_for_imports_detects_simple_import(self, tmp_path: Path) -> None:
        """Verify _scan_file_for_imports detects 'import kafka' style imports."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("import kafka\n")

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].import_pattern == "kafka"
        assert violations[0].line_number == 1

    def test_scan_file_for_imports_detects_from_import(self, tmp_path: Path) -> None:
        """Verify _scan_file_for_imports detects 'from kafka import X' imports."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("from kafka.producer import KafkaProducer\n")

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].import_pattern == "kafka"
        assert violations[0].line_number == 1

    def test_scan_file_for_imports_detects_submodule_import(
        self, tmp_path: Path
    ) -> None:
        """Verify _scan_file_for_imports detects 'import kafka.producer' imports."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("import kafka.producer\n")

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].import_pattern == "kafka"

    def test_scan_file_for_imports_skips_comments(self, tmp_path: Path) -> None:
        """Verify _scan_file_for_imports skips commented-out imports."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("# import kafka\n# from kafka import Producer\n")

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_skips_inline_comments(self, tmp_path: Path) -> None:
        """Verify _scan_file_for_imports handles lines starting with comments."""
        test_file = tmp_path / "test_module.py"
        content = """\
# This file used to use kafka
# import kafka  # old import
x = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_skips_docstrings(self, tmp_path: Path) -> None:
        """Verify _scan_file_for_imports skips imports mentioned in docstrings."""
        test_file = tmp_path / "test_module.py"
        content = '''\
"""
This module provides kafka integration.
Example:
    import kafka
    from kafka import Producer
"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_skips_single_quote_docstrings(
        self, tmp_path: Path
    ) -> None:
        """Verify _scan_file_for_imports skips single-quoted docstrings."""
        test_file = tmp_path / "test_module.py"
        content = """\
'''
import kafka
from kafka import Producer
'''

def my_func():
    pass
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_detects_real_import_after_docstring(
        self, tmp_path: Path
    ) -> None:
        """Verify _scan_file_for_imports detects imports after docstrings end."""
        test_file = tmp_path / "test_module.py"
        content = '''\
"""Module docstring."""

import kafka
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        # Line 1: docstring, Line 2: empty, Line 3: import kafka
        assert violations[0].line_number == 3

    def test_scan_file_for_imports_handles_unreadable_file(
        self, tmp_path: Path
    ) -> None:
        """Verify _scan_file_for_imports handles files that cannot be read."""
        nonexistent = tmp_path / "does_not_exist.py"
        violations = _scan_file_for_imports(nonexistent, ["kafka"])
        assert violations == []

    def test_scan_file_for_imports_no_violations_when_pattern_not_found(
        self, tmp_path: Path
    ) -> None:
        """Verify _scan_file_for_imports returns empty list when no violations."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text("import os\nimport sys\n")

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert violations == []

    # --- Tests for _format_violation_report ---

    def test_format_violation_report_includes_file_path(self, tmp_path: Path) -> None:
        """Verify _format_violation_report includes file path in output."""
        test_file = tmp_path / "my_module.py"
        violation = ArchitectureViolation(
            file_path=test_file,
            line_number=10,
            line_content="import kafka",
            import_pattern="kafka",
        )

        report = _format_violation_report([violation], "kafka", "omnibase_core")
        assert str(test_file) in report
        assert ":10:" in report

    def test_format_violation_report_includes_pattern(self) -> None:
        """Verify _format_violation_report includes the import pattern."""
        violation = ArchitectureViolation(
            file_path=Path("/fake/path.py"),
            line_number=1,
            line_content="import httpx",
            import_pattern="httpx",
        )

        report = _format_violation_report([violation], "httpx", "omnibase_core")
        assert "'httpx'" in report
        assert "ARCHITECTURE VIOLATION" in report

    def test_format_violation_report_includes_package_name(self) -> None:
        """Verify _format_violation_report includes the package name."""
        violation = ArchitectureViolation(
            file_path=Path("/fake/path.py"),
            line_number=1,
            line_content="import kafka",
            import_pattern="kafka",
        )

        report = _format_violation_report([violation], "kafka", "my_package")
        assert "my_package" in report
        assert "must not contain infrastructure dependencies" in report

    def test_format_violation_report_multiple_violations(self) -> None:
        """Verify _format_violation_report handles multiple violations."""
        violations = [
            ArchitectureViolation(
                file_path=Path("/fake/module1.py"),
                line_number=5,
                line_content="import kafka",
                import_pattern="kafka",
            ),
            ArchitectureViolation(
                file_path=Path("/fake/module2.py"),
                line_number=10,
                line_content="from kafka import Producer",
                import_pattern="kafka",
            ),
        ]

        report = _format_violation_report(violations, "kafka", "omnibase_core")
        assert "module1.py" in report
        assert "module2.py" in report
        assert ":5:" in report
        assert ":10:" in report

    # --- Tests for _get_package_source_path ---

    def test_get_package_source_path_returns_none_for_nonexistent(self) -> None:
        """Verify _get_package_source_path returns None for nonexistent packages."""
        result = _get_package_source_path("nonexistent_package_xyz_12345")
        assert result is None

    def test_get_package_source_path_finds_installed_package(self) -> None:
        """Verify _get_package_source_path finds an installed package."""
        # Use a standard library package that's guaranteed to exist
        result = _get_package_source_path("os")
        # os is a builtin, should return None or a path
        # Let's try with pathlib which is definitely a package
        result = _get_package_source_path("pathlib")
        # pathlib may also be builtin, try pytest which we know is installed
        result = _get_package_source_path("pytest")
        assert result is not None
        assert result.exists()

    # --- Tests for _scan_package_for_forbidden_imports ---

    def test_scan_package_for_forbidden_imports_raises_for_unknown_package(
        self,
    ) -> None:
        """Verify _scan_package_for_forbidden_imports raises for unknown package."""
        with pytest.raises(ValueError, match="Could not locate package"):
            _scan_package_for_forbidden_imports(
                "nonexistent_package_xyz_12345", ["kafka"]
            )

    # --- Tests for ArchitectureViolation ---

    def test_architecture_violation_str_format(self, tmp_path: Path) -> None:
        """Verify ArchitectureViolation __str__ format is correct."""
        test_file = tmp_path / "module.py"
        violation = ArchitectureViolation(
            file_path=test_file,
            line_number=42,
            line_content="  import kafka  ",
            import_pattern="kafka",
        )

        result = str(violation)
        assert str(test_file) in result
        assert ":42:" in result
        assert "import kafka" in result
        # Verify content is stripped
        assert "  import kafka  " not in result

    # --- Tests for docstring edge cases ---

    def test_scan_file_for_imports_unclosed_multiline_at_eof(
        self, tmp_path: Path
    ) -> None:
        """Verify unclosed multiline string at EOF does not produce false violations.

        If a file ends with an unclosed multiline string, the scanner should
        not report any violations from within that string.
        """
        test_file = tmp_path / "test_module.py"
        # Unclosed multiline string - no closing triple quotes
        content = """\
'''
This docstring mentions import kafka
but is never closed
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_import_after_closing_docstring_same_line(
        self, tmp_path: Path
    ) -> None:
        """Verify import after closing docstring on same line is detected.

        An import statement following a closing docstring delimiter on the
        same line should be detected as a violation.
        """
        test_file = tmp_path / "test_module.py"
        content = '"""closing docstring""" import kafka\n'
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 1
        assert violations[0].import_pattern == "kafka"

    def test_scan_file_for_imports_nested_quotes_in_docstrings(
        self, tmp_path: Path
    ) -> None:
        """Verify nested quote delimiters inside docstrings are handled.

        Triple single quotes inside triple double quotes should not break
        the docstring parsing.
        """
        test_file = tmp_path / "test_module.py"
        content = (
            '"""\n'
            "Example: use ''' for nested\n"
            "import kafka should be ignored\n"
            '"""\n'
            "\n"
            "def my_func():\n"
            "    pass\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_fstring_docstring(self, tmp_path: Path) -> None:
        """Verify f-string prefixed docstrings are handled correctly.

        Content inside f-string formatted docstrings should not trigger
        violations.
        """
        test_file = tmp_path / "test_module.py"
        content = '''\
f"""docstring with import kafka"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_raw_string_docstring(self, tmp_path: Path) -> None:
        """Verify r-string prefixed docstrings are handled correctly.

        Content inside raw string formatted docstrings should not trigger
        violations.
        """
        test_file = tmp_path / "test_module.py"
        content = '''\
r"""raw docstring with import kafka"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_mixed_delimiters(self, tmp_path: Path) -> None:
        """Verify files with both triple-quote delimiters are handled.

        A file containing both triple-double-quote and triple-single-quote
        docstrings should have both properly handled without confusion.
        """
        test_file = tmp_path / "test_module.py"
        content = (
            '"""\n'
            "First docstring with import kafka mentioned\n"
            '"""\n'
            "\n"
            "def func_one():\n"
            "    pass\n"
            "\n"
            "'''\n"
            "Second docstring with from kafka import Producer\n"
            "'''\n"
            "\n"
            "def func_two():\n"
            "    pass\n"
            "\n"
            "import os  # This should not trigger kafka violation\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_mixed_delimiters_with_real_import(
        self, tmp_path: Path
    ) -> None:
        """Verify real imports are detected after mixed delimiter docstrings.

        A file with both delimiter types should still detect real imports
        outside of docstrings.
        """
        test_file = tmp_path / "test_module.py"
        content = (
            '"""\n'
            "Module docstring mentions kafka in example\n"
            '"""\n'
            "\n"
            "'''\n"
            "Function docstring with from kafka import X\n"
            "'''\n"
            "\n"
            "import kafka  # This is a real import\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 9
        assert violations[0].import_pattern == "kafka"

    def test_scan_file_for_imports_multiple_delimiters_on_same_line(
        self, tmp_path: Path
    ) -> None:
        """Verify multiple delimiter types on same line are handled correctly.

        A line containing both triple-double and triple-single quotes
        should be handled without confusion, with imports after both detected.
        """
        test_file = tmp_path / "test_module.py"
        # Both delimiter types on the same line, followed by import
        content = "\"\"\"docstring1\"\"\" '''docstring2''' import kafka\n"
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 1
        assert violations[0].import_pattern == "kafka"

    def test_scan_file_for_imports_closing_starts_new_multiline(
        self, tmp_path: Path
    ) -> None:
        """Verify closing delimiter followed by new multiline start is handled.

        When a line closes one multiline and immediately starts another,
        content inside the new multiline should not trigger violations.
        """
        test_file = tmp_path / "test_module.py"
        content = (
            '"""\n'
            "First docstring\n"
            '""" """\n'  # Close first, start second (unbalanced - opens new multiline)
            "import kafka inside second docstring\n"
            '"""\n'
            "x = 1\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_close_multiline_opens_new(
        self, tmp_path: Path
    ) -> None:
        """Verify closing multiline then opening new multiline is handled.

        When closing a multiline string, if the remainder starts a new
        unbalanced string, the import inside should NOT be detected.
        """
        test_file = tmp_path / "test_module.py"
        # Close multiline, then `text"""` starts a NEW multiline
        # So `import kafka` is inside the new multiline - no violation
        content = (
            '"""\n'
            "Starting a multiline docstring\n"
            '"""text""" import kafka\n'  # Close first, text""" starts new multiline
            '"""\n'  # Close the second multiline
            "x = 1\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        # Import is inside new multiline string, so no violation
        assert len(violations) == 0

    def test_scan_file_for_imports_close_multiline_with_import(
        self, tmp_path: Path
    ) -> None:
        """Verify import after closing multiline on same line is detected.

        When closing a multiline string and there's an import after it
        (not inside a new string), the import should be detected.
        """
        test_file = tmp_path / "test_module.py"
        # Close multiline, then import (no new string started)
        content = (
            '"""\n'
            "Starting a multiline docstring\n"
            '""" import kafka\n'  # Close multiline, then import
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 3
        assert violations[0].import_pattern == "kafka"

    def test_scan_file_for_imports_close_multiline_balanced_then_import(
        self, tmp_path: Path
    ) -> None:
        """Verify closing multiline followed by balanced string then import.

        When closing a multiline, then having a balanced single-line string,
        then an import, the import should be detected.
        """
        test_file = tmp_path / "test_module.py"
        # Close multiline, balanced single-line string """x""", then import
        content = (
            '"""\n'
            "Starting a multiline docstring\n"
            '""" """x""" import kafka\n'  # Close first, """x""" is balanced, import
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 3
        assert violations[0].import_pattern == "kafka"

    def test_scan_file_for_imports_br_prefix_string(self, tmp_path: Path) -> None:
        """Verify br (raw bytes) string prefix is handled correctly.

        Content inside br-prefixed strings should not trigger violations.
        """
        test_file = tmp_path / "test_module.py"
        content = '''\
br"""raw bytes docstring with import kafka"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_rb_prefix_string(self, tmp_path: Path) -> None:
        """Verify rb (raw bytes) string prefix is handled correctly.

        Content inside rb-prefixed strings should not trigger violations.
        """
        test_file = tmp_path / "test_module.py"
        content = '''\
rb"""raw bytes docstring with import kafka"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_uppercase_prefix(self, tmp_path: Path) -> None:
        """Verify uppercase string prefixes are handled correctly.

        Python allows uppercase prefixes like R, F, B, RF, BR, etc.
        """
        test_file = tmp_path / "test_module.py"
        content = '''\
RF"""uppercase raw f-string with import kafka"""
BR"""uppercase raw bytes with from kafka import X"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_docstring_immediate_import_no_space(
        self, tmp_path: Path
    ) -> None:
        """Verify import immediately after docstring with no space is detected.

        Edge case where there's no space between the closing delimiter
        and the import keyword.
        """
        test_file = tmp_path / "test_module.py"
        # No space between closing """ and import
        content = '"""docstring"""import kafka\n'
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 1

    def test_scan_file_for_imports_multiline_closes_at_eof_no_newline(
        self, tmp_path: Path
    ) -> None:
        """Verify multiline string closing at EOF without trailing newline.

        Files may not have a trailing newline after the last line.
        """
        test_file = tmp_path / "test_module.py"
        # No trailing newline after closing docstring
        content = '"""\nimport kafka in docstring\n"""'
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0
