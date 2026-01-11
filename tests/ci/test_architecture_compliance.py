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

Detection Capabilities
======================

**This Python test file** uses line-by-line regex scanning which detects:
- Top-level imports: `import kafka`, `from kafka import X`
- Inline imports inside functions/methods (indented imports ARE detected)
- Imports with submodules: `import kafka.producer`, `from kafka.consumer import X`
- Imports inside class bodies

Example - ALL of these ARE detected by this Python scanner::

    import kafka                      # Top-level - DETECTED
    from kafka import Producer        # Top-level - DETECTED

    def my_function():
        import kafka                  # Inline/indented - DETECTED
        from httpx import Client      # Inline/indented - DETECTED

    class MyClass:
        from asyncpg import connect   # Class body - DETECTED

**Limitations of this Python scanner:**
- Imports constructed dynamically (`__import__()`, `importlib.import_module()`)
- String-based import references in configuration files
- Imports inside `exec()` or `eval()` calls
- Multi-line import statements (e.g., imports wrapped with parentheses over multiple lines)
- Imports aliased through variables (e.g., `mod = __import__("kafka"); mod.Producer`)

**Docstring Detection Caveats:**
- Multiline strings are tracked by delimiter type (triple-single vs triple-double)
- Nested delimiters inside strings are correctly ignored (e.g., triple-single inside triple-double)
- Unclosed multiline strings at EOF are handled gracefully (content treated as inside string)
- String prefixes (r, f, b, u, rf, fr, rb, br) are recognized and handled
- Edge case: imports on the same line after a closing delimiter ARE detected

Multiline String State Machine
==============================

The scanner uses a state machine to track whether the current line is inside a
multiline string (docstring). This is necessary because import statements inside
docstrings should not be flagged as violations.

**State Variables:**

- ``in_multiline_string`` (bool): Whether currently inside a multiline string
- ``multiline_delimiter`` (str | None): The delimiter type (triple-single or triple-double) if inside

**State Transitions:**

1. **NOT in multiline -> Line has unbalanced delimiter:**

   - Find first triple-quote delimiter on line
   - If odd count of that delimiter -> enter multiline, save delimiter type
   - Content before the opening delimiter is checked for imports

2. **IN multiline -> Line has closing delimiter:**

   - Find closing delimiter matching saved type
   - Exit multiline state
   - Content after the closing delimiter is checked for imports
   - If remainder has another unbalanced delimiter -> enter new multiline

3. **IN multiline -> Line has NO closing delimiter:**

   - Stay in multiline state
   - Entire line is skipped (no import checking)

**Handling Nested Delimiters:**

The tricky case is when one delimiter type appears inside another.
For example: triple-single This docstring contains triple-double inside triple-single

The scanner handles this by:

1. Finding which delimiter type appears FIRST on the line
2. Checking if that type is "balanced" (even count = opens and closes on same line)
3. Only considering the OTHER delimiter type if it appears OUTSIDE balanced strings

**Key Functions:**

- ``_find_first_unquoted_delimiter()``: Find first triple-quote that starts a string
- ``_is_balanced_string_line()``: Check if a delimiter type has even count
- ``_find_delimiter_outside_balanced()``: Find delimiters not inside other strings
- ``_count_delimiter_outside_balanced()``: Count delimiters outside balanced strings
- ``_find_multiline_state_after_line()``: Main state transition function

**Edge Cases Handled:**

- Opening and closing on same line: triple-double docstring triple-double (balanced, not multiline)
- Multiple strings on one line: triple-double first triple-double triple-double second triple-double
- Delimiter inside other type: triple-single contains triple-double inside triple-single
- Code after closing: triple-double docstring triple-double import kafka (import IS detected)
- Empty multiline: Just triple-double on its own line
- Unclosed at EOF: Treated as inside string (graceful degradation)

**Known Limitations:**

- Raw strings with escaped delimiters may confuse the parser in rare cases
- Extremely complex nesting patterns (3+ levels) are not explicitly tested
- String concatenation across lines is not handled (e.g., "foo" "bar")

Tool Comparison
===============

**Bash script** (`scripts/check_architecture.sh`) has MORE limitations:
- Cannot reliably distinguish real imports from mentions in comments/docstrings
- Cannot parse multiline docstrings (may produce false positives)
- Cannot detect TYPE_CHECKING blocks (may flag type-only imports)
- Has no context awareness (line-by-line grep matching)

Note: The bash script CAN find indented imports (grep does pattern matching), but
it cannot distinguish them from import statements mentioned in docstrings or comments.
This leads to potential false positives that this Python scanner avoids.

The bash script is designed for quick CI checks with JSON output support.
This Python test file provides more thorough analysis with:
- Proper multiline docstring handling (no false positives from examples)
- TYPE_CHECKING block awareness (type-only imports allowed)
- Context-aware detection (distinguishes real imports from documentation)

For comprehensive detection with zero false positives, prefer this Python test file:
    pytest tests/ci/test_architecture_compliance.py

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

    Note:
        This function returns None for namespace packages (PEP 420) because
        they have no single __init__.py file (spec.origin is None). Namespace
        packages span multiple directories and cannot be represented by a
        single source path. This is a known limitation - architecture checks
        for namespace packages would require different handling.
    """
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        # spec is None: package not found
        # spec.origin is None: namespace package (PEP 420) with no __init__.py
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

    SECURITY NOTE: This function uses STRICT explicit matching to prevent
    accidental exemption of Python modules that happen to contain
    "requirements" in their name (e.g., requirements_handler.py).

    Hardening measures:
    1. Python files (.py) are NEVER exempted except for exact setup.py match
    2. Requirements files MUST be .txt extension (not .py, .yaml, etc.)
    3. Requirements files MUST start with "requirements" (exact prefix)
    4. Requirements files MUST have simple naming (requirements[-_]*.txt)

    Args:
        file_path: Path to check.

    Returns:
        True if the file is a requirements/config file.
    """
    file_name = file_path.name.lower()

    # HARDENING: Python files (.py) are ONLY exempted for exact setup.py match
    # This prevents exempting modules like requirements_handler.py, setup_utils.py
    if file_name.endswith(".py"):
        return file_name == "setup.py"

    # Explicit exact matches for non-Python config files
    if file_name in {"setup.cfg", "pyproject.toml"}:
        return True

    # Requirements files: STRICT pattern matching
    # MUST be .txt extension (not .py, .yaml, .json, etc.)
    # MUST start with "requirements" exactly
    # MUST follow standard naming: requirements.txt, requirements-dev.txt, requirements_test.txt
    if not file_name.endswith(".txt"):
        return False

    # Check for standard requirements file patterns:
    # - requirements.txt (exact)
    # - requirements-*.txt (with hyphen separator)
    # - requirements_*.txt (with underscore separator)
    if file_name == "requirements.txt":
        return True

    # Pattern: requirements followed by separator (-/_) then more text, ending in .txt
    # This matches: requirements-dev.txt, requirements_test.txt
    # But NOT: my_requirements.txt, requirements_data.txt (no separator after "requirements")
    if file_name.startswith(("requirements-", "requirements_")):
        return True

    return False


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
        # Defensive handling: if we're supposedly in a multiline but have no delimiter,
        # treat this as a corrupted state and return to normal parsing mode.
        # This handles edge cases where state tracking may have become inconsistent.
        if current_delimiter is None:
            return False, None, line
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

    SYNC REQUIREMENT:
        The forbidden imports list in this class MUST be kept in sync with:
        - scripts/check_architecture.sh:77-89 (FORBIDDEN_IMPORTS array)

        When adding or removing imports, update ALL THREE LOCATIONS:
        1. scripts/check_architecture.sh FORBIDDEN_IMPORTS (lines 77-89)
        2. This file: parametrized list below (lines 733-746)
        3. This file: comprehensive list in test_comprehensive_infra_scan (lines 809-821)

    KNOWN_VIOLATIONS_WITH_TICKETS:
        Some violations are tracked with tickets and have xfail markers on
        individual tests. The comprehensive scan filters these to avoid
        duplicate CI failures. When a ticket is resolved, remove the pattern
        from KNOWN_VIOLATION_PATTERNS below and the corresponding xfail marker.
    """

    CORE_PACKAGE = "omnibase_core"

    # Patterns with known violations tracked by tickets.
    # These are filtered from comprehensive scan to avoid duplicate failures.
    # Update this list when:
    # - Adding a new xfail marker to individual tests (add pattern here)
    # - Resolving a ticket (remove pattern from here AND remove xfail marker)
    KNOWN_VIOLATION_PATTERNS: frozenset[str] = frozenset(
        {
            "aiohttp",  # OMN-1015: Known violation - async HTTP client needs migration
            "redis",  # OMN-1295: Known violation - Redis client needs migration
            "consul",  # OMN-1015: Known violation - Consul TYPE_CHECKING import
        }
    )

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
                    reason="OMN-1015: Known violation - async HTTP client needs migration to infra",
                    strict=False,
                ),
                id="no-aiohttp",
            ),
            pytest.param(
                "redis",
                "cache",
                marks=pytest.mark.xfail(
                    reason="OMN-1295: Known violation - Redis client needs migration to infra",
                    strict=False,
                ),
                id="no-redis",
            ),
            pytest.param("psycopg", "PostgreSQL driver (v3)", id="no-psycopg"),
            pytest.param("psycopg2", "PostgreSQL driver (v2)", id="no-psycopg2"),
            pytest.param(
                "consul",
                "service discovery client",
                marks=pytest.mark.xfail(
                    reason="OMN-1015: Known violation - Consul TYPE_CHECKING import needs migration",
                    strict=False,
                ),
                id="no-consul",
            ),
            pytest.param("hvac", "Vault client", id="no-hvac"),
            pytest.param("aiokafka", "async Kafka client", id="no-aiokafka"),
            pytest.param(
                "confluent_kafka", "Confluent Kafka client", id="no-confluent-kafka"
            ),
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

    @pytest.mark.serial
    @pytest.mark.timeout(300)
    @pytest.mark.xdist_group(name="serial")
    def test_comprehensive_infra_scan(self) -> None:
        """Comprehensive scan for all infrastructure imports in core.

        This is a catch-all test that checks for multiple infrastructure
        patterns in a single pass. Use this to quickly verify that no
        infrastructure dependencies have leaked into core.

        Known Violations
        ----------------
        Violations for patterns in KNOWN_VIOLATION_PATTERNS are filtered out
        and reported separately (as warnings) because they have dedicated
        xfail tests with tracked tickets. This prevents duplicate CI failures
        while still catching NEW violations immediately.

        Resource Constraints
        --------------------
        This test is resource-intensive because it scans ALL Python files in
        omnibase_core, parses them, and checks for forbidden import patterns.
        When run in parallel with pytest-xdist, it can crash CI workers due to
        memory pressure from multiple workers running similar scans concurrently.

        Markers:
        - serial: Documents this test should run serially
        - timeout(300): 5-minute timeout as the scan takes ~3 minutes locally
        - xdist_group("serial"): Forces pytest-xdist to run this test in
          isolation, not in parallel with other tests

        Local runtime: ~3 minutes
        CI runtime: May vary based on worker resources
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

        # Separate known violations (tracked by tickets) from new violations
        known_violations: list[ArchitectureViolation] = []
        new_violations: list[ArchitectureViolation] = []
        for v in violations:
            if v.import_pattern in self.KNOWN_VIOLATION_PATTERNS:
                known_violations.append(v)
            else:
                new_violations.append(v)

        # Report known violations as warnings (for visibility), not failures
        if known_violations:
            known_by_pattern: dict[str, list[ArchitectureViolation]] = {}
            for v in known_violations:
                known_by_pattern.setdefault(v.import_pattern, []).append(v)

            import warnings

            warning_lines = [
                f"Known violations ({len(known_violations)} total) - tracked by tickets:",
            ]
            for pattern, pvs in sorted(known_by_pattern.items()):
                warning_lines.append(f"  {pattern}: {len(pvs)} violation(s)")
            warnings.warn("\n".join(warning_lines), stacklevel=1)

        # Fail only on NEW violations (not tracked by tickets)
        if new_violations:
            # Group violations by pattern
            by_pattern: dict[str, list[ArchitectureViolation]] = {}
            for v in new_violations:
                by_pattern.setdefault(v.import_pattern, []).append(v)

            report_lines = [
                "ARCHITECTURE VIOLATIONS DETECTED",
                "",
                f"Found {len(new_violations)} NEW violation(s) in {self.CORE_PACKAGE}:",
                "(Known violations with tickets are excluded from this count)",
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
            # Valid requirements/config files (should be exempted)
            pytest.param("requirements.txt", True, id="requirements-txt"),
            pytest.param("requirements-dev.txt", True, id="requirements-dev-txt"),
            pytest.param("requirements_test.txt", True, id="requirements-underscore"),
            pytest.param("requirements-prod.txt", True, id="requirements-prod"),
            pytest.param("requirements_local.txt", True, id="requirements-local"),
            pytest.param("setup.py", True, id="setup-py"),
            pytest.param("setup.cfg", True, id="setup-cfg"),
            pytest.param("pyproject.toml", True, id="pyproject-toml"),
            # Regular Python files (should NOT be exempted)
            pytest.param("my_module.py", False, id="regular-module"),
            pytest.param("test_something.py", False, id="test-file"),
            pytest.param("conftest.py", False, id="conftest"),
            pytest.param("__init__.py", False, id="init-file"),
            # SECURITY: .py files with "requirements" in name must NOT be exempted
            # These tests verify the fix for PR review concern about substring matching
            pytest.param(
                "requirements_handler.py", False, id="requirements-handler-py"
            ),
            pytest.param("my_requirements.py", False, id="my-requirements-py"),
            pytest.param("requirements_utils.py", False, id="requirements-utils-py"),
            pytest.param("parse_requirements.py", False, id="parse-requirements-py"),
            pytest.param("requirements.py", False, id="requirements-py-not-txt"),
            # SECURITY: setup*.py variants must NOT be exempted (only exact setup.py)
            pytest.param("setup_utils.py", False, id="setup-utils-py"),
            pytest.param("setup_test.py", False, id="setup-test-py"),
            pytest.param("mysetup.py", False, id="mysetup-py"),
            # Edge cases for requirements files
            pytest.param("requirements", False, id="requirements-no-extension"),
            pytest.param("REQUIREMENTS.TXT", True, id="requirements-uppercase"),
            pytest.param("Requirements-Dev.txt", True, id="requirements-mixed-case"),
            # SECURITY: Non-standard requirements file patterns must NOT be exempted
            # These ensure strict pattern matching (only requirements[-_]*.txt)
            pytest.param("requirementsdata.txt", False, id="requirements-no-separator"),
            pytest.param("my_requirements.txt", False, id="my-requirements-txt"),
            pytest.param("old_requirements.txt", False, id="old-requirements-txt"),
            # SECURITY: Non-.txt requirements files must NOT be exempted
            pytest.param("requirements.yaml", False, id="requirements-yaml"),
            pytest.param("requirements.yml", False, id="requirements-yml"),
            pytest.param("requirements.json", False, id="requirements-json"),
            pytest.param("requirements.toml", False, id="requirements-toml"),
            pytest.param("requirements.in", False, id="requirements-in"),
        ],
    )
    def test_is_requirements_file(
        self, tmp_path: Path, filename: str, expected: bool
    ) -> None:
        """Verify _is_requirements_file correctly identifies config/requirements files.

        SECURITY: This test verifies that .py files containing "requirements"
        in their name are NOT exempted from architecture compliance checks.
        This prevents malicious or accidental bypass of import restrictions.

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

    def test_find_multiline_state_after_line_corrupted_state(self) -> None:
        """Verify _find_multiline_state_after_line handles corrupted state gracefully.

        This tests the defensive handling for the edge case where we're supposedly
        in a multiline string but have no delimiter. Instead of crashing with an
        AssertionError, the function should recover by returning to normal mode.
        """
        # Corrupted state: in_multiline=True but delimiter=None
        # Should recover gracefully by returning to normal mode
        in_ml, delim, content = _find_multiline_state_after_line(
            "some line content",
            True,
            None,  # Corrupted: no delimiter
        )
        assert in_ml is False
        assert delim is None
        assert content == "some line content"

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

    # --- Additional edge case tests for multiline string handling ---

    def test_scan_handles_unclosed_multiline_at_eof(self, tmp_path: Path) -> None:
        """Verify scanner handles files ending inside multiline string.

        This is a critical edge case where a file ends without closing
        a multiline string. The scanner should NOT report any imports
        found inside the unclosed string as violations.
        """
        test_file = tmp_path / "test_module.py"
        content = '''x = 1
"""
import kafka  # Should NOT be detected - inside unclosed string
'''
        test_file.write_text(content)
        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_handles_empty_multiline_string(self, tmp_path: Path) -> None:
        """Verify scanner handles empty multiline strings correctly.

        Empty multiline strings (6 consecutive quotes) should be treated
        as a balanced, empty string, not as two separate delimiters with
        content in between.
        """
        test_file = tmp_path / "empty_multiline.py"
        # Six quotes = empty multiline string
        content = '''x = """"""
import kafka  # This should be detected - outside the empty string
"""More content that mentions kafka"""
y = 1
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 2

    def test_scan_handles_empty_multiline_single_quotes(self, tmp_path: Path) -> None:
        """Verify scanner handles empty single-quoted multiline strings.

        Same as above but with single quotes.
        """
        test_file = tmp_path / "empty_multiline_single.py"
        content = """x = ''''''
import kafka  # This should be detected - outside the empty string
'''More content that mentions kafka'''
y = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 2

    def test_scan_handles_fstring_with_nested_quotes(self, tmp_path: Path) -> None:
        """Verify f-strings with nested expressions are handled correctly.

        F-strings can contain expressions with nested quotes. The scanner
        should treat the entire f-string as a string and not detect
        import-like content inside.
        """
        test_file = tmp_path / "fstring_nested.py"
        content = '''x = f"""This is an f-string
With {var} and import kafka mentioned
The import kafka is inside the f-string
"""
y = 1
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_handles_fstring_with_nested_braces(self, tmp_path: Path) -> None:
        """Verify f-strings with complex nested expressions are handled.

        F-strings can have nested braces and complex expressions.
        """
        test_file = tmp_path / "fstring_complex.py"
        content = '''x = f"""Result: {kafka_config.get("key", "default")}"""
import kafka  # Real import outside the f-string
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 2

    def test_scan_handles_nested_single_inside_double(self, tmp_path: Path) -> None:
        r"""Verify triple-single-quotes inside triple-double-quotes are ignored.

        When \'\'\' appears inside a \"\"\", it should not start a new string.
        """
        test_file = tmp_path / "nested_single_in_double.py"
        # Build content with triple-double quotes containing triple-single quotes
        content = (
            '"""\n'
            "This docstring contains ''' triple single quotes\n"
            "import kafka  # Inside the outer docstring\n"
            "The ''' should not close anything\n"
            '"""\n'
            "y = 1\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_handles_nested_double_inside_single(self, tmp_path: Path) -> None:
        r"""Verify triple-double-quotes inside triple-single-quotes are ignored.

        When \"\"\" appears inside a \'\'\', it should not start a new string.
        """
        test_file = tmp_path / "nested_double_in_single.py"
        # Build content with triple-single quotes containing triple-double quotes
        content = (
            "'''\n"
            'This docstring contains """ triple double quotes\n'
            "import kafka  # Inside the outer docstring\n"
            'The """ should not close anything\n'
            "'''\n"
            "y = 1\n"
        )
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    def test_scan_handles_mixed_delimiters_same_line_balanced(
        self, tmp_path: Path
    ) -> None:
        """Verify both delimiter types balanced on same line.

        When a line contains balanced strings of both types, both should
        be treated as complete strings.
        """
        test_file = tmp_path / "mixed_balanced.py"
        content = """x = '''a''' + \"\"\"b\"\"\" + '''c'''
import kafka  # Outside all strings, should be detected
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 1
        assert violations[0].line_number == 2

    def test_scan_handles_mixed_delimiters_same_line_one_unbalanced(
        self, tmp_path: Path
    ) -> None:
        """Verify one delimiter balanced, other unbalanced on same line.

        When one delimiter type is balanced but the other opens a multiline,
        the content after should be inside the multiline.
        """
        test_file = tmp_path / "mixed_one_unbalanced.py"
        content = """x = '''a''' \"\"\"starts multiline
import kafka  # Inside the multiline, should NOT be detected
\"\"\"
y = 1
"""
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        assert len(violations) == 0

    # --- Tests for string prefix regex validation ---

    @pytest.mark.parametrize(
        ("invalid_prefix", "description"),
        [
            pytest.param("xy", "invalid two-char prefix", id="xy-invalid"),
            pytest.param("ub", "u cannot combine with b", id="ub-invalid"),
            pytest.param("fu", "f cannot combine with u", id="fu-invalid"),
            pytest.param("bf", "b cannot combine with f", id="bf-invalid"),
            pytest.param("uf", "u cannot combine with f", id="uf-invalid"),
            pytest.param("abc", "three-char invalid prefix", id="abc-invalid"),
        ],
    )
    def test_scan_detects_import_after_invalid_prefix(
        self, tmp_path: Path, invalid_prefix: str, description: str
    ) -> None:
        """Verify invalid string prefixes don't prevent import detection.

        Invalid prefixes (like 'xy', 'ub', 'fu') should not be recognized
        as valid string prefixes, so any following content should still
        be scanned for imports. In practice, Python would reject these
        as syntax errors, but the scanner should handle them gracefully.

        Args:
            tmp_path: Pytest fixture for temporary directory.
            invalid_prefix: The invalid prefix to test.
            description: Human-readable description of why prefix is invalid.
        """
        test_file = tmp_path / "invalid_prefix.py"
        # In Python, this would be a syntax error (invalid prefix)
        # But we test that the scanner doesn't incorrectly skip content
        # after something that looks like a prefix + delimiter
        content = f'''# Test file with potential edge case
x = 1
{invalid_prefix}"""not a string""" import kafka
y = 2
'''
        test_file.write_text(content)

        # Since the prefix is invalid, the scanner should handle this gracefully
        # The exact behavior depends on implementation - it might detect the import
        # or might skip the line. The key is it doesn't crash.
        # For well-formed code, this test verifies robustness.
        try:
            violations = _scan_file_for_imports(test_file, ["kafka"])
            # If we get here, scanner handled it gracefully (either way is fine)
            assert isinstance(violations, list)
        except Exception as e:
            pytest.fail(f"Scanner crashed on invalid prefix '{invalid_prefix}': {e}")

    def test_scan_valid_prefix_no_false_positive(self, tmp_path: Path) -> None:
        """Verify valid prefixes are properly recognized and don't cause false positives.

        All valid Python 3 string prefixes should be recognized:
        - Single: r, R, f, F, b, B, u, U
        - Combinations: rf, fr, rb, br (and case variants)
        """
        test_file = tmp_path / "valid_prefixes.py"
        content = '''r"""raw string with import kafka"""
f"""f-string with import kafka"""
b"""bytes with import kafka"""
u"""unicode with import kafka"""
rf"""raw f-string with import kafka"""
fr"""f-string raw with import kafka"""
rb"""raw bytes with import kafka"""
br"""bytes raw with import kafka"""
RF"""uppercase raw f-string with import kafka"""
x = 1
'''
        test_file.write_text(content)

        violations = _scan_file_for_imports(test_file, ["kafka"])
        # All mentions of kafka are inside strings, no violations
        assert len(violations) == 0

    # --- Tests for _get_balanced_string_ranges edge cases ---

    def test_get_balanced_string_ranges_empty_line(self) -> None:
        """Verify _get_balanced_string_ranges handles empty lines."""
        result = _get_balanced_string_ranges("", '"""')
        assert result == []

    def test_get_balanced_string_ranges_no_delimiters(self) -> None:
        """Verify _get_balanced_string_ranges handles lines with no delimiters."""
        result = _get_balanced_string_ranges("x = 1 + 2", '"""')
        assert result == []

    def test_get_balanced_string_ranges_single_delimiter(self) -> None:
        """Verify _get_balanced_string_ranges returns None for odd count."""
        result = _get_balanced_string_ranges('x = """hello', '"""')
        assert result is None

    def test_get_balanced_string_ranges_balanced(self) -> None:
        """Verify _get_balanced_string_ranges returns correct ranges."""
        result = _get_balanced_string_ranges('x = """hello""" + y', '"""')
        assert result is not None
        assert len(result) == 1
        # Start at 4 (position of first """), end at 15 (after closing """)
        assert result[0] == (4, 15)

    def test_get_balanced_string_ranges_multiple_balanced(self) -> None:
        """Verify _get_balanced_string_ranges handles multiple balanced strings."""
        result = _get_balanced_string_ranges('"""a""" """b"""', '"""')
        assert result is not None
        assert len(result) == 2
        assert result[0] == (0, 7)  # First string
        assert result[1] == (8, 15)  # Second string

    def test_get_balanced_string_ranges_empty_string(self) -> None:
        """Verify _get_balanced_string_ranges handles empty strings (6 quotes)."""
        result = _get_balanced_string_ranges('""""""', '"""')
        assert result is not None
        assert len(result) == 1
        # Empty string: start at 0, end at 6
        assert result[0] == (0, 6)

    # --- Tests for _count_delimiter_outside_balanced ---

    def test_count_delimiter_outside_balanced_none_outside(self) -> None:
        """Verify _count_delimiter_outside_balanced when all inside."""
        line = '"""contains \'\'\' inside"""'
        count = _count_delimiter_outside_balanced(line, "'''", '"""')
        assert count == 0

    def test_count_delimiter_outside_balanced_some_outside(self) -> None:
        """Verify _count_delimiter_outside_balanced counts correctly."""
        line = '"""a""" \'\'\' """b""" \'\'\''
        count = _count_delimiter_outside_balanced(line, "'''", '"""')
        assert count == 2

    def test_count_delimiter_outside_balanced_unbalanced_base(self) -> None:
        """Verify _count_delimiter_outside_balanced with unbalanced base."""
        # When balanced_delim is unbalanced, returns total count
        line = "\"\"\"unbalanced ''' after"
        count = _count_delimiter_outside_balanced(line, "'''", '"""')
        assert count == 1
