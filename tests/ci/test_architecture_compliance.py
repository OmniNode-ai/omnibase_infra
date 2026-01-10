"""Architecture compliance tests for ONEX infrastructure.

These tests verify that architectural boundaries are maintained between
omnibase_core (pure, no I/O) and omnibase_infra (infrastructure, owns all I/O).

The core principle: omnibase_core must not contain any infrastructure-specific
imports such as kafka, httpx, or asyncpg. These belong exclusively in omnibase_infra.

Ticket: OMN-255

CI Integration
==============

These tests run as part of the CI pipeline in two ways:

1. **Pre-push hook** (`.pre-commit-config.yaml`):
   - Hook ID: `onex-validate-architecture-layers`
   - Stage: `pre-push` (not pre-commit for performance)
   - Runs: `poetry run python scripts/validate.py architecture_layers`

2. **GitHub Actions** (`.github/workflows/test.yml`):
   - Job: `onex-validation` ("ONEX Validators")
   - Runs: `poetry run python scripts/validate.py all --verbose`
   - Includes architecture_layers as part of the full validation suite

Detection Limitations (IMPORTANT)
=================================

This validator uses regex-based pattern matching which has limitations
compared to full AST-based Python analysis:

**What IS detected (top-level imports):**
- `import kafka`
- `from kafka import Producer`
- `import kafka.producer`
- `from kafka.consumer import Consumer`

**What is NOT detected (inline imports inside functions/methods):**

    def my_function():
        import kafka  # NOT DETECTED by regex!
        from httpx import Client  # NOT DETECTED by regex!

Inline imports are common patterns to:
- Avoid circular import issues
- Lazy-load heavy dependencies
- Conditionally import based on runtime conditions

**Other limitations:**
- Imports constructed dynamically at runtime (`__import__()`, `importlib`)
- Imports hidden behind conditional logic (if/else blocks)
- String-based import references in configuration

For comprehensive detection including inline imports, this test file uses
proper Python regex patterns that check the actual import structure.
The bash script (`scripts/check_architecture.sh`) has additional limitations
due to grep-based parsing.

See Also
========
- scripts/check_architecture.sh - Bash-based quick check with JSON output
- scripts/validate.py - Python validation wrapper with KNOWN_ISSUES registry
- .pre-commit-config.yaml - Pre-commit/pre-push hook configuration
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


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


def _find_delimiter_positions(line: str, delimiter: str) -> list[int]:
    """Find all positions of a delimiter in a line.

    Args:
        line: The line of code to search.
        delimiter: The delimiter to find (e.g., '\"\"\"' or \"'''\").

    Returns:
        List of all positions where the delimiter starts.
    """
    positions: list[int] = []
    start = 0
    while True:
        pos = line.find(delimiter, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + len(delimiter)
    return positions


def _get_balanced_string_ranges(
    line: str, balanced_delim: str
) -> list[tuple[int, int]] | None:
    """Get the ranges of positions that are inside balanced strings.

    Args:
        line: The line of code to search.
        balanced_delim: The delimiter type (e.g., '\"\"\"' or \"'''\").

    Returns:
        List of (start, end) tuples defining ranges inside strings,
        or None if the delimiter is not balanced (odd count).
    """
    balanced_positions = _find_delimiter_positions(line, balanced_delim)

    # Odd count means unbalanced
    if len(balanced_positions) % 2 != 0:
        return None

    # Create ranges - each pair of positions defines a string
    inside_ranges: list[tuple[int, int]] = []
    for i in range(0, len(balanced_positions), 2):
        start = balanced_positions[i]
        end = balanced_positions[i + 1] + len(balanced_delim)
        inside_ranges.append((start, end))

    return inside_ranges


def _find_delimiter_outside_balanced(
    line: str, target_delim: str, balanced_delim: str
) -> int:
    """Find the first occurrence of target_delim that is outside balanced_delim strings.

    When a line contains balanced strings of one delimiter type (e.g., two \"\"\"
    forming a complete string), we need to find occurrences of the other delimiter
    type that are NOT inside those balanced strings.

    Args:
        line: The line of code to search.
        target_delim: The delimiter to find (e.g., \"'''\").
        balanced_delim: The delimiter type that is balanced (e.g., '\"\"\"').

    Returns:
        Position of the first target_delim outside balanced_delim strings,
        or -1 if not found.
    """
    inside_ranges = _get_balanced_string_ranges(line, balanced_delim)

    # If not balanced, fall back to simple find
    if inside_ranges is None:
        return line.find(target_delim)

    # Find all positions of target delimiter
    target_positions = _find_delimiter_positions(line, target_delim)

    # Return the first one that is outside all balanced string ranges
    for pos in target_positions:
        is_inside = False
        for range_start, range_end in inside_ranges:
            if range_start < pos < range_end:
                is_inside = True
                break
        if not is_inside:
            return pos

    return -1


def _count_delimiter_outside_balanced(
    line: str, target_delim: str, balanced_delim: str
) -> int:
    """Count occurrences of target_delim that are outside balanced_delim strings.

    Args:
        line: The line of code to search.
        target_delim: The delimiter to count (e.g., \"'''\").
        balanced_delim: The delimiter type that is balanced (e.g., '\"\"\"').

    Returns:
        Count of target_delim occurrences outside balanced_delim strings.
    """
    inside_ranges = _get_balanced_string_ranges(line, balanced_delim)

    # If not balanced, count all occurrences
    if inside_ranges is None:
        return len(_find_delimiter_positions(line, target_delim))

    # Count target_delim positions that are outside all balanced string ranges
    target_positions = _find_delimiter_positions(line, target_delim)
    count = 0
    for pos in target_positions:
        is_inside = False
        for range_start, range_end in inside_ranges:
            if range_start < pos < range_end:
                is_inside = True
                break
        if not is_inside:
            count += 1
    return count


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


def _find_multiline_state_after_line(
    line: str,
    current_in_multiline: bool,
    current_delimiter: str | None,
) -> tuple[bool, str | None, str]:
    """Determine multiline string state after processing a line.

    This function handles the complex logic of tracking multiline string state
    including cases where one delimiter type appears inside another, and where
    multiple strings open and close on the same line.

    Args:
        line: The line of code to analyze.
        current_in_multiline: Whether we're currently inside a multiline string.
        current_delimiter: The delimiter of the current multiline (if any).

    Returns:
        Tuple of:
        - New in_multiline state
        - New delimiter (if in multiline)
        - Content to check for imports (code outside strings)
    """
    if current_in_multiline:
        assert current_delimiter is not None
        closing_pos = line.find(current_delimiter)
        if closing_pos == -1:
            # Still inside multiline, no code to check
            return True, current_delimiter, ""

        # Found closing delimiter
        after_close = line[closing_pos + 3 :]

        # Check if remainder starts a new multiline
        new_delim, new_pos = _find_first_unquoted_delimiter(after_close)
        if new_delim is not None and not _is_balanced_string_line(
            after_close, new_delim
        ):
            # New multiline starts
            content_before_new = after_close[:new_pos] if new_pos > 0 else ""
            return True, new_delim, content_before_new.strip()

        # No new multiline - check the remainder for the other delimiter type
        # Must find other_delim that is OUTSIDE any balanced new_delim strings
        other_delim = "'''" if current_delimiter == '"""' else '"""'
        if other_delim in after_close and not _is_balanced_string_line(
            after_close, other_delim
        ):
            # Find position outside any balanced strings of new_delim type
            if new_delim is not None and _is_balanced_string_line(
                after_close, new_delim
            ):
                other_pos = _find_delimiter_outside_balanced(
                    after_close, other_delim, new_delim
                )
            else:
                other_pos = after_close.find(other_delim)
            if other_pos != -1:
                content_before_other = after_close[:other_pos] if other_pos > 0 else ""
                return True, other_delim, content_before_other.strip()

        # Not in multiline anymore
        return False, None, after_close
    else:
        # Not in multiline - check if one starts
        first_delim, first_pos = _find_first_unquoted_delimiter(line)
        if first_delim is None:
            # No triple quotes at all
            return False, None, line

        if _is_balanced_string_line(line, first_delim):
            # First delimiter type is balanced - check for the other type
            # Important: we must count other_delim OUTSIDE the balanced first_delim
            # strings, not the total count which may include delimiters inside strings
            other_delim = "'''" if first_delim == '"""' else '"""'
            other_count_outside = _count_delimiter_outside_balanced(
                line, other_delim, first_delim
            )
            if other_count_outside > 0 and other_count_outside % 2 != 0:
                # Odd count = unbalanced = entering multiline
                other_pos = _find_delimiter_outside_balanced(
                    line, other_delim, first_delim
                )
                if other_pos != -1:
                    content_before = line[:other_pos] if other_pos > 0 else ""
                    return True, other_delim, content_before

            # Both types balanced or other type not present outside first_delim
            return False, None, line
        else:
            # First delimiter is unbalanced - entering multiline
            content_before = line[:first_pos] if first_pos > 0 else ""
            return True, first_delim, content_before


def _is_in_type_checking_block(lines: Sequence[str], current_line_idx: int) -> bool:
    """Check if the current line is inside a TYPE_CHECKING conditional block.

    TYPE_CHECKING blocks are used for type-only imports that should not trigger
    architecture violations since they don't affect runtime behavior.

    A line is considered "inside" a TYPE_CHECKING block if:
    1. A previous line contains `if TYPE_CHECKING:`
    2. The current line has greater indentation than that `if` statement
    3. No intervening line at the same or less indentation has broken the block

    Args:
        lines: All lines in the file.
        current_line_idx: Index of the current line (0-based).

    Returns:
        True if the line is inside a TYPE_CHECKING block.
    """
    # Track indentation of TYPE_CHECKING if block
    type_checking_indent: int | None = None
    type_checking_line_idx: int | None = None

    for idx in range(current_line_idx + 1):
        line = lines[idx]
        stripped = line.lstrip()

        # Skip empty lines and comments for block detection
        if not stripped or stripped.startswith("#"):
            continue

        current_indent = len(line) - len(stripped)

        # Check if we've exited the TYPE_CHECKING block (less or equal indentation)
        # Only check lines AFTER the TYPE_CHECKING statement
        if (
            type_checking_indent is not None
            and type_checking_line_idx is not None
            and idx > type_checking_line_idx
            and current_indent <= type_checking_indent
        ):
            type_checking_indent = None
            type_checking_line_idx = None

        # Check for TYPE_CHECKING if statement
        if re.match(r"if\s+TYPE_CHECKING\s*:", stripped):
            type_checking_indent = current_indent
            type_checking_line_idx = idx
            continue

    # To be inside the block, we need:
    # 1. type_checking_indent to be set (we found an `if TYPE_CHECKING:`)
    # 2. The current line index must be AFTER the TYPE_CHECKING line
    # 3. The current line must have greater indentation than TYPE_CHECKING
    if type_checking_indent is None or type_checking_line_idx is None:
        return False

    if current_line_idx <= type_checking_line_idx:
        return False

    # Get the current line's indentation
    current_line = lines[current_line_idx]
    current_stripped = current_line.lstrip()
    if not current_stripped or current_stripped.startswith("#"):
        # Empty or comment lines inherit the block context
        return True
    current_indent = len(current_line) - len(current_stripped)

    return current_indent > type_checking_indent


def _scan_file_for_imports(
    file_path: Path,
    forbidden_patterns: list[str],
) -> list[ArchitectureViolation]:
    """Scan a Python file for forbidden import patterns.

    Detects both `import X` and `from X import Y` patterns.
    Properly handles multiline docstrings (both triple-quoted variants),
    including edge cases where one delimiter type appears inside the other.

    Also properly handles TYPE_CHECKING conditional blocks - imports inside
    these blocks are type-only and should not trigger violations.

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

        for line_idx, line in enumerate(lines):
            line_num = line_idx + 1  # 1-indexed for reporting
            stripped = line.lstrip()

            # Skip comment lines (only if not in multiline string)
            if not in_multiline_string and stripped.startswith("#"):
                continue

            # Use the helper to determine new multiline state and extractable content
            new_in_multiline, new_delimiter, content_to_check = (
                _find_multiline_state_after_line(
                    line, in_multiline_string, multiline_delimiter
                )
            )

            # Update state for next iteration
            prev_in_multiline = in_multiline_string
            in_multiline_string = new_in_multiline
            multiline_delimiter = new_delimiter

            # If we were in a multiline and still are (no closing found), skip
            if prev_in_multiline and not content_to_check:
                continue

            # Handle single-line docstrings that start the extracted content
            # Loop to handle multiple consecutive single-line strings
            processed_content = content_to_check.lstrip()
            # Handle valid Python string prefixes (case insensitive):
            #   Single: r, R, f, F, b, B, u, U
            #   Combinations: rf/fr (raw f-string), rb/br (raw bytes)
            # Note: u cannot combine with other prefixes in Python 3
            # The regex anchors to start and requires the delimiter to follow immediately
            # This strict pattern rejects invalid prefixes like 'xy', 'ub', 'fu', etc.
            string_prefix_pattern = (
                r"^("
                r"[rR][fF]|[fF][rR]|"  # Raw f-strings: rf, rF, Rf, RF, fr, fR, Fr, FR
                r"[rR][bB]|[bB][rR]|"  # Raw bytes: rb, rB, Rb, RB, br, bR, Br, BR
                r"[rRfFbBuU]"  # Single prefixes: r, R, f, F, b, B, u, U
                r")?"
            )
            found_docstring = True
            while found_docstring:
                found_docstring = False
                for delimiter in ('"""', "'''"):
                    # Check if content starts with optional prefix + delimiter
                    prefix_match = re.match(
                        string_prefix_pattern + re.escape(delimiter), processed_content
                    )
                    if prefix_match:
                        # Check if this delimiter is balanced (single-line string)
                        if _is_balanced_string_line(processed_content, delimiter):
                            # Balanced - find where the closing delimiter ends
                            first_pos = processed_content.find(delimiter)
                            second_pos = processed_content.find(
                                delimiter, first_pos + 3
                            )
                            if second_pos != -1:
                                # Get content after the closing delimiter
                                after_docstring = processed_content[second_pos + 3 :]
                                if after_docstring.strip():
                                    # There's content after - continue processing
                                    processed_content = after_docstring.lstrip()
                                    found_docstring = True
                                else:
                                    # Only whitespace after docstring
                                    processed_content = ""
                                break  # Restart loop with updated content

            # Skip if no content left to check
            if not processed_content.strip():
                continue

            # Skip imports inside TYPE_CHECKING blocks (type-only imports)
            if _is_in_type_checking_block(lines, line_idx):
                continue

            if import_regex.match(processed_content):
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
            pytest.param("aiohttp", "async HTTP", id="no-aiohttp"),
            pytest.param("redis", "cache", id="no-redis"),
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
            "aiohttp",
            "redis",
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

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            pytest.param("requirements.txt", True, id="requirements-txt"),
            pytest.param("requirements-dev.txt", True, id="requirements-dev-txt"),
            pytest.param("requirements_test.txt", True, id="requirements-underscore"),
            pytest.param("setup.py", True, id="setup-py"),
            pytest.param("setup.cfg", True, id="setup-cfg"),
            pytest.param("pyproject.toml", True, id="pyproject-toml"),
            pytest.param("my_module.py", False, id="regular-module"),
            pytest.param("test_something.py", False, id="test-file"),
            pytest.param("conftest.py", False, id="conftest"),
            pytest.param("__init__.py", False, id="init-file"),
        ],
    )
    def test_is_requirements_file(
        self, tmp_path: Path, filename: str, expected: bool
    ) -> None:
        """Verify _is_requirements_file correctly identifies config/requirements files.

        Args:
            tmp_path: Pytest fixture for temporary directory.
            filename: Name of the file to test.
            expected: Whether the file should be identified as a requirements file.
        """
        test_file = tmp_path / filename
        test_file.touch()
        assert _is_requirements_file(test_file) is expected

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

    @pytest.mark.parametrize(
        ("package_name", "expect_found"),
        [
            pytest.param("pytest", True, id="pytest-installed"),
            pytest.param("pathlib", True, id="pathlib-stdlib"),
            pytest.param("nonexistent_package_xyz_12345", False, id="nonexistent"),
        ],
    )
    def test_get_package_source_path_behavior(
        self, package_name: str, expect_found: bool
    ) -> None:
        """Verify _get_package_source_path behaves correctly for different packages.

        Args:
            package_name: Name of the package to locate.
            expect_found: Whether the package is expected to be found.
        """
        result = _get_package_source_path(package_name)
        if expect_found:
            assert result is not None, f"Expected to find package: {package_name}"
            assert result.exists(), f"Package path should exist: {result}"
        else:
            assert result is None, f"Expected package not found: {package_name}"

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

    @pytest.mark.parametrize(
        ("prefix", "description"),
        [
            pytest.param("f", "f-string", id="f-prefix"),
            pytest.param("r", "raw string", id="r-prefix"),
            pytest.param("b", "bytes string", id="b-prefix"),
            pytest.param("br", "raw bytes", id="br-prefix"),
            pytest.param("rb", "raw bytes (reversed)", id="rb-prefix"),
            pytest.param("fr", "raw f-string", id="fr-prefix"),
            pytest.param("rf", "raw f-string (reversed)", id="rf-prefix"),
            pytest.param("F", "uppercase f-string", id="F-prefix"),
            pytest.param("R", "uppercase raw", id="R-prefix"),
            pytest.param("BR", "uppercase raw bytes", id="BR-prefix"),
            pytest.param("RF", "uppercase raw f-string", id="RF-prefix"),
        ],
    )
    def test_scan_file_for_imports_prefixed_string(
        self, tmp_path: Path, prefix: str, description: str
    ) -> None:
        """Verify prefixed docstrings are handled correctly.

        Content inside prefixed strings should not trigger violations.

        Args:
            tmp_path: Pytest fixture for temporary directory.
            prefix: The string prefix to test (f, r, b, br, etc.).
            description: Human-readable description of the prefix type.
        """
        test_file = tmp_path / "test_module.py"
        content = f'''{prefix}"""docstring with import kafka"""

def my_func():
    pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0, f"Unexpected violation for {description} prefix"

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

    def test_scan_file_for_imports_code_between_close_and_new_multiline(
        self, tmp_path: Path
    ) -> None:
        """Verify import between closing and new multiline start is detected.

        When closing a multiline string and a new multiline starts on the
        same line, any code (including imports) between them should be
        detected. This tests the fix for the bug where such imports were
        missed because the scanner would skip directly to the new multiline.
        """
        test_file = tmp_path / "test_module.py"
        # Close first multiline, import, then start new multiline
        content = (
            '"""\n'
            "Starting a multiline docstring\n"
            '""" import kafka; """\n'  # Close first, import, start new multiline
            "content in new multiline\n"
            '"""\n'
            "x = 1\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        # Import is between two strings, should be detected
        assert len(violations) == 1
        assert violations[0].line_number == 3
        assert violations[0].import_pattern == "kafka"

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

    # --- Additional edge case tests for comprehensive coverage ---

    def test_scan_file_for_imports_empty_file(self, tmp_path: Path) -> None:
        """Verify empty files are handled without errors.

        An empty file should not produce any violations or errors.
        """
        test_file = tmp_path / "empty_module.py"
        test_file.write_text("")

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_only_comments(self, tmp_path: Path) -> None:
        """Verify files containing only comments are handled correctly.

        A file with only comment lines should not produce any violations.
        """
        test_file = tmp_path / "comments_only.py"
        content = """\
# This is a comment mentioning import kafka
# from kafka import Producer
# Another comment about kafka
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_only_docstrings(self, tmp_path: Path) -> None:
        """Verify files containing only docstrings are handled correctly.

        A file with only docstring content should not produce any violations
        even if the docstrings mention import patterns.
        """
        test_file = tmp_path / "docstrings_only.py"
        content = '''\
"""
Module docstring that mentions import kafka.
from kafka import Producer

This is documentation only.
"""

"""
Another docstring with kafka mentioned.
"""
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_deeply_nested_docstrings(
        self, tmp_path: Path
    ) -> None:
        """Verify deeply nested function docstrings are handled correctly.

        Docstrings within nested class/function definitions should all be
        properly identified and their content ignored for import scanning.
        """
        test_file = tmp_path / "nested_module.py"
        content = '''\
"""Module docstring mentioning import kafka."""

class OuterClass:
    """Class docstring with from kafka import Producer."""

    def outer_method(self):
        """Method docstring about kafka."""

        class InnerClass:
            """Inner class docstring - import kafka."""

            def inner_method(self):
                """Deeply nested - from kafka import Consumer."""
                pass

def outer_function():
    """Function docstring about kafka."""

    def inner_function():
        """Inner function - import kafka."""
        pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_consecutive_multiline_strings(
        self, tmp_path: Path
    ) -> None:
        """Verify consecutive multiline strings are handled correctly.

        Multiple multiline strings appearing one after another should all
        be properly identified and their content ignored.
        """
        test_file = tmp_path / "consecutive_strings.py"
        content = '''\
"""First multiline
with import kafka
mentioned here."""

"""Second multiline
from kafka import Producer
also mentioned."""

"""Third multiline
import kafka again."""

x = 1  # No violation here
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_mixed_quotes_consecutive(
        self, tmp_path: Path
    ) -> None:
        """Verify consecutive strings with mixed quote types are handled.

        Alternating between triple-double and triple-single quotes in
        consecutive strings should all be properly handled.
        """
        test_file = tmp_path / "mixed_quotes.py"
        content = """\
\"\"\"Double quoted with import kafka.\"\"\"

'''Single quoted with from kafka import X.'''

\"\"\"Back to double with kafka.\"\"\"

'''Again single with import kafka.'''

y = 2
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_whitespace_only_lines(self, tmp_path: Path) -> None:
        """Verify files with whitespace-only lines are handled correctly.

        Whitespace-only lines should not cause issues with parsing
        and should not be treated as imports.
        """
        test_file = tmp_path / "whitespace_module.py"
        content = """\

\t
   \t

import os


"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_indented_import(self, tmp_path: Path) -> None:
        """Verify indented imports (inside functions/classes) are detected.

        Imports can appear inside function or class bodies with indentation.
        These should still be detected as violations.
        """
        test_file = tmp_path / "indented_import.py"
        content = """\
def my_func():
    import kafka

class MyClass:
    from kafka import Producer
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 2
        assert violations[0].line_number == 2
        assert violations[1].line_number == 5

    # --- Tests for TYPE_CHECKING conditional imports ---

    def test_scan_file_for_imports_type_checking_import_allowed(
        self, tmp_path: Path
    ) -> None:
        """Verify imports inside TYPE_CHECKING blocks are not flagged.

        TYPE_CHECKING blocks are used for type-only imports that don't affect
        runtime behavior. These should be allowed even for infrastructure imports.
        """
        test_file = tmp_path / "type_checking_import.py"
        content = """\
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import kafka
    from kafka import Producer

def my_func():
    pass
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_type_checking_mixed_with_regular(
        self, tmp_path: Path
    ) -> None:
        """Verify TYPE_CHECKING imports allowed but regular imports flagged.

        When a file has both TYPE_CHECKING imports and regular imports,
        only the regular imports should be flagged as violations.
        """
        test_file = tmp_path / "mixed_type_checking.py"
        content = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kafka import Producer  # Type-only, should be allowed

import kafka  # Runtime import, should be flagged

def my_func():
    pass
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 6

    def test_scan_file_for_imports_type_checking_indented_content(
        self, tmp_path: Path
    ) -> None:
        """Verify TYPE_CHECKING block with multiple indented imports.

        Multiple imports inside a TYPE_CHECKING block should all be allowed.
        """
        test_file = tmp_path / "type_checking_multi.py"
        content = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import kafka
    from kafka import Producer
    from kafka.consumer import Consumer
    import httpx
    from asyncpg import Connection

x = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka", "httpx", "asyncpg"])
        assert len(violations) == 0

    def test_scan_file_for_imports_type_checking_nested_in_class(
        self, tmp_path: Path
    ) -> None:
        """Verify TYPE_CHECKING inside class body is handled.

        TYPE_CHECKING blocks can appear inside class bodies for
        type annotations.
        """
        test_file = tmp_path / "type_checking_in_class.py"
        content = """\
from typing import TYPE_CHECKING

class MyClass:
    if TYPE_CHECKING:
        from kafka import Producer

    def method(self) -> None:
        pass
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    # --- Tests for docstrings with unusual patterns ---

    def test_scan_file_for_imports_docstring_at_class_start(
        self, tmp_path: Path
    ) -> None:
        """Verify docstring at start of class is handled correctly.

        Class docstrings immediately after the class definition line
        should be properly identified.
        """
        test_file = tmp_path / "class_docstring.py"
        content = '''\
class MyClass:
    """This class mentions import kafka.

    Example:
        from kafka import Producer
        producer = Producer()
    """

    def method(self):
        pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_docstring_at_function_start(
        self, tmp_path: Path
    ) -> None:
        """Verify docstring at start of function is handled correctly.

        Function docstrings immediately after the def line should be
        properly identified.
        """
        test_file = tmp_path / "function_docstring.py"
        content = '''\
def my_function():
    """This function mentions import kafka.

    Args:
        from kafka import Producer - example import
    """
    return None
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_docstring_unusual_indentation(
        self, tmp_path: Path
    ) -> None:
        """Verify docstrings with unusual indentation are handled.

        Docstrings may have content at various indentation levels.
        """
        test_file = tmp_path / "unusual_indent_docstring.py"
        content = '''\
def my_function():
    """Start of docstring.
import kafka  # At column 0 inside docstring
    from kafka import Producer  # Indented inside docstring
        import kafka.consumer  # More indented inside docstring
    """
    return None
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_docstring_after_decorator(
        self, tmp_path: Path
    ) -> None:
        """Verify docstrings after decorators are handled correctly.

        Functions with decorators should still have their docstrings
        properly identified and excluded from import scanning.
        """
        test_file = tmp_path / "decorated_function.py"
        content = '''\
@decorator
def my_function():
    """Docstring mentioning import kafka."""
    return None

@decorator1
@decorator2
@decorator3
def another_function():
    """Another docstring with from kafka import Producer."""
    return None

class MyClass:
    @classmethod
    def my_method(cls):
        """Method docstring about kafka."""
        pass

    @staticmethod
    def static_method():
        """Static method docstring - import kafka."""
        pass
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_docstring_after_decorator_with_args(
        self, tmp_path: Path
    ) -> None:
        """Verify docstrings after decorators with arguments are handled.

        Decorators can have complex argument expressions including
        string literals that might contain import-like text.
        """
        test_file = tmp_path / "decorator_with_args.py"
        content = '''\
@decorator("import kafka")
def my_function():
    """Docstring mentioning import kafka."""
    return None

@route("/kafka/producer", methods=["POST"])
def kafka_route():
    """Route handler that mentions from kafka import Producer."""
    return None
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    # --- Additional edge cases for multiline strings ---

    def test_scan_file_for_imports_unclosed_double_at_eof(self, tmp_path: Path) -> None:
        """Verify unclosed double-quote multiline at EOF is handled.

        Edge case where file ends with unclosed triple double quotes.
        """
        test_file = tmp_path / "unclosed_double.py"
        content = '''\
x = 1

"""
This docstring is never closed.
import kafka
from kafka import Producer
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_unclosed_single_at_eof(self, tmp_path: Path) -> None:
        """Verify unclosed single-quote multiline at EOF is handled.

        Edge case where file ends with unclosed triple single quotes.
        """
        test_file = tmp_path / "unclosed_single.py"
        content = """\
x = 1

'''
This docstring is never closed.
import kafka
from kafka import Producer
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_starts_in_multiline_no_close(
        self, tmp_path: Path
    ) -> None:
        """Verify file that starts with multiline and never closes.

        Edge case where the entire file is inside a multiline string.
        """
        test_file = tmp_path / "all_in_string.py"
        content = '''\
"""This file starts with a multiline string.
import kafka
from kafka import Producer
Everything here is inside the string.
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_balanced_with_other_unbalanced(
        self, tmp_path: Path
    ) -> None:
        """Verify line with balanced quotes and unbalanced other type.

        Edge case: one delimiter type is balanced but the other type
        starts a multiline string.
        """
        test_file = tmp_path / "mixed_balance.py"
        # The """ are balanced (2 occurrences), but ''' starts multiline
        content = '''x = """hello""" + \'\'\'multiline
import kafka  # Inside the single-quoted multiline
\'\'\'
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_import_before_unbalanced_string(
        self, tmp_path: Path
    ) -> None:
        """Verify import before unbalanced string on same line is detected.

        The import appears before the multiline string starts, so it
        should be detected as a violation.
        """
        test_file = tmp_path / "import_before_string.py"
        content = '''import kafka; x = """multiline starts
more content
"""
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 1

    # --- Tests for helper functions ---

    def test_find_multiline_state_after_line_not_in_multiline(self) -> None:
        """Verify _find_multiline_state_after_line when not in multiline."""
        # No quotes at all
        in_ml, delim, content = _find_multiline_state_after_line("x = 1", False, None)
        assert in_ml is False
        assert delim is None
        assert content == "x = 1"

    def test_find_multiline_state_after_line_entering_multiline(self) -> None:
        """Verify _find_multiline_state_after_line when entering multiline."""
        # Single triple-quote starts multiline
        in_ml, delim, content = _find_multiline_state_after_line(
            'x = """hello', False, None
        )
        assert in_ml is True
        assert delim == '"""'
        assert content == "x = "

    def test_find_multiline_state_after_line_closing_multiline(self) -> None:
        """Verify _find_multiline_state_after_line when closing multiline."""
        # Inside multiline, then close it
        in_ml, delim, content = _find_multiline_state_after_line(
            'closing""" x = 1', True, '"""'
        )
        assert in_ml is False
        assert delim is None
        assert content == " x = 1"

    def test_find_multiline_state_after_line_still_in_multiline(self) -> None:
        """Verify _find_multiline_state_after_line when staying in multiline."""
        # Inside multiline, no closing delimiter
        in_ml, delim, content = _find_multiline_state_after_line(
            "still inside", True, '"""'
        )
        assert in_ml is True
        assert delim == '"""'
        assert content == ""

    def test_is_in_type_checking_block_outside(self) -> None:
        """Verify _is_in_type_checking_block returns False when outside."""
        lines = [
            "import os",
            "x = 1",
        ]
        assert _is_in_type_checking_block(lines, 0) is False
        assert _is_in_type_checking_block(lines, 1) is False

    def test_is_in_type_checking_block_inside(self) -> None:
        """Verify _is_in_type_checking_block returns True when inside."""
        lines = [
            "from typing import TYPE_CHECKING",
            "if TYPE_CHECKING:",
            "    import kafka",
            "x = 1",
        ]
        assert _is_in_type_checking_block(lines, 0) is False
        assert _is_in_type_checking_block(lines, 1) is False
        assert _is_in_type_checking_block(lines, 2) is True
        assert _is_in_type_checking_block(lines, 3) is False

    def test_is_in_type_checking_block_multiple_levels(self) -> None:
        """Verify _is_in_type_checking_block handles indentation changes."""
        lines = [
            "if TYPE_CHECKING:",
            "    import kafka",
            "    if True:",
            "        from kafka import Producer",
            "x = 1",
        ]
        assert _is_in_type_checking_block(lines, 1) is True
        assert _is_in_type_checking_block(lines, 3) is True
        assert _is_in_type_checking_block(lines, 4) is False

    # --- Tests for _find_delimiter_positions ---

    def test_find_delimiter_positions_empty_line(self) -> None:
        """Verify _find_delimiter_positions returns empty list for no matches."""
        positions = _find_delimiter_positions("x = 1", '"""')
        assert positions == []

    def test_find_delimiter_positions_single_occurrence(self) -> None:
        """Verify _find_delimiter_positions finds single delimiter."""
        positions = _find_delimiter_positions('x = """hello', '"""')
        assert positions == [4]

    def test_find_delimiter_positions_multiple_occurrences(self) -> None:
        """Verify _find_delimiter_positions finds all delimiters."""
        positions = _find_delimiter_positions('"""hello""" + """world"""', '"""')
        assert positions == [0, 8, 14, 22]

    def test_find_delimiter_positions_adjacent_delimiters(self) -> None:
        """Verify _find_delimiter_positions handles adjacent delimiters."""
        # Six quotes = two delimiters adjacent
        positions = _find_delimiter_positions('""""""', '"""')
        assert positions == [0, 3]

    # --- Tests for _find_delimiter_outside_balanced ---

    def test_find_delimiter_outside_balanced_no_balanced(self) -> None:
        """Verify _find_delimiter_outside_balanced with no balanced strings."""
        # No balanced_delim in line, should find target normally
        pos = _find_delimiter_outside_balanced("'''hello", "'''", '"""')
        assert pos == 0

    def test_find_delimiter_outside_balanced_target_inside(self) -> None:
        """Verify _find_delimiter_outside_balanced skips target inside balanced."""
        # ''' is inside the """ string, should not find it
        line = '"""contains \'\'\' inside"""'
        pos = _find_delimiter_outside_balanced(line, "'''", '"""')
        assert pos == -1

    def test_find_delimiter_outside_balanced_target_after(self) -> None:
        """Verify _find_delimiter_outside_balanced finds target after balanced."""
        # ''' appears after the """ string closes
        line = '"""hello""" \'\'\''
        pos = _find_delimiter_outside_balanced(line, "'''", '"""')
        assert pos == 12

    def test_find_delimiter_outside_balanced_multiple_balanced(self) -> None:
        """Verify _find_delimiter_outside_balanced handles multiple balanced strings."""
        # ''' after two balanced """ strings
        line = '"""a""" """b""" \'\'\''
        pos = _find_delimiter_outside_balanced(line, "'''", '"""')
        assert pos == 16

    def test_find_delimiter_outside_balanced_target_between_balanced(self) -> None:
        """Verify _find_delimiter_outside_balanced finds target between balanced strings."""
        # ''' between two balanced """ strings
        line = '"""a""" \'\'\' """b"""'
        pos = _find_delimiter_outside_balanced(line, "'''", '"""')
        assert pos == 8

    def test_find_delimiter_outside_balanced_complex_scenario(self) -> None:
        """Verify _find_delimiter_outside_balanced handles complex mixed scenario.

        This tests the specific bug fix where triple-single-quotes inside
        triple-double-quotes was incorrectly matched when looking for
        unbalanced triple-single-quotes outside the triple-double-quote string.
        """
        # The ''' at position 11 is inside """, the ''' at position 25 is outside
        line = "x = \"\"\"has ''' inside\"\"\" '''"
        pos = _find_delimiter_outside_balanced(line, "'''", '"""')
        assert pos == 25

    # --- Tests for edge case: balanced delimiter containing other delimiter ---

    def test_scan_file_for_imports_balanced_containing_other_delimiter(
        self, tmp_path: Path
    ) -> None:
        """Verify balanced strings containing other delimiter type work correctly.

        This tests the specific bug fix for PR #89: when one delimiter type
        is balanced but contains the other delimiter type inside, the scanner
        should correctly identify that the inner delimiter is NOT starting
        a new multiline string.
        """
        test_file = tmp_path / "balanced_containing_other.py"
        # The """ is balanced, and ''' inside should be ignored
        # The ''' after the """ should start a multiline
        content = '''x = """contains \'\'\' inside""" \'\'\'multiline starts
import kafka  # Inside the single-quoted multiline
\'\'\'
y = 1
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_file_for_imports_reverse_balanced_containing_other(
        self, tmp_path: Path
    ) -> None:
        """Verify reverse case: balanced single quotes containing double quotes.

        Same as above but with triple-single-quotes balanced and
        triple-double-quotes inside.
        """
        test_file = tmp_path / "reverse_balanced.py"
        # Content: x = '''contains """ inside''' """multiline starts...
        # Triple single quotes are balanced, triple double quotes inside are ignored
        # The trailing """ starts a multiline
        actual_content = (
            "x = '''contains \"\"\" inside''' \"\"\"multiline starts\n"
            "import kafka  # Inside the double-quoted multiline\n"
            '"""\n'
            "y = 1\n"
        )
        test_file.write_text(actual_content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    # --- Tests for TYPE_CHECKING edge cases ---

    def test_scan_file_for_imports_type_checking_after_regular_import(
        self, tmp_path: Path
    ) -> None:
        """Verify TYPE_CHECKING after regular import doesn't affect detection.

        Regular imports before TYPE_CHECKING block should still be flagged.
        """
        test_file = tmp_path / "type_check_after_regular.py"
        content = """\
import kafka  # This should be flagged

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kafka import Consumer  # This should be allowed

x = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 1

    def test_scan_file_for_imports_type_checking_with_else(
        self, tmp_path: Path
    ) -> None:
        """Verify TYPE_CHECKING with else block is handled correctly.

        Content in the else block should be checked for violations.
        """
        test_file = tmp_path / "type_check_with_else.py"
        content = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kafka import Producer  # Type-only, allowed
else:
    import kafka  # Runtime import, should be flagged

x = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 6

    def test_scan_file_for_imports_type_checking_empty_lines(
        self, tmp_path: Path
    ) -> None:
        """Verify empty lines inside TYPE_CHECKING block don't break detection."""
        test_file = tmp_path / "type_check_empty_lines.py"
        content = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import kafka

    from kafka import Producer

x = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0
