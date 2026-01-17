"""Tests for validate_no_direct_env.py script.

This module tests the environment variable access validator that enforces
the SecretResolver pattern. It verifies detection of:
- Standard os.environ/os.getenv usage
- Module alias patterns (import os as _os; _os.environ)
- Bare environ/getenv patterns (from os import environ; environ[...])
- False positive prevention (custom functions/variables shouldn't match)

Ticket: OMN-764
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from validate_no_direct_env import (
    FORBIDDEN_PATTERNS,
    AllowlistValidationError,
    Violation,
    _has_inline_exclusion_marker,
    _matches_exclusion_pattern,
    load_allowlist,
    scan_file,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestForbiddenPatterns:
    """Tests for FORBIDDEN_PATTERNS regex detection."""

    # === Standard os.environ/os.getenv patterns ===

    @pytest.mark.parametrize(
        ("code_line", "description"),
        [
            ('value = os.getenv("API_KEY")', "os.getenv with string"),
            ("value = os.getenv('API_KEY')", "os.getenv with single quotes"),
            ('value = os.environ["API_KEY"]', "os.environ bracket access"),
            ("value = os.environ['API_KEY']", "os.environ single quote brackets"),
            ('value = os.environ.get("API_KEY")', "os.environ.get"),
            ('os.environ.setdefault("KEY", "default")', "os.environ.setdefault"),
            ('os.environ.pop("KEY")', "os.environ.pop"),
            ("os.environ.clear()", "os.environ.clear"),
            ('os.environ.update({"KEY": "value"})', "os.environ.update"),
        ],
    )
    def test_detects_standard_os_patterns(
        self, code_line: str, description: str
    ) -> None:
        """Verify standard os.environ/os.getenv patterns are detected."""
        matches = [p for p in FORBIDDEN_PATTERNS if p.search(code_line)]
        assert matches, f"Expected to detect: {description} in '{code_line}'"

    # === Module alias patterns ===

    @pytest.mark.parametrize(
        ("code_line", "description"),
        [
            ('value = _os.environ["API_KEY"]', "underscore alias _os.environ"),
            ('value = _os.getenv("API_KEY")', "underscore alias _os.getenv"),
            ('value = my_os.environ["KEY"]', "custom alias my_os.environ"),
            ('value = o.environ["KEY"]', "single letter alias o.environ"),
            ('value = os_module.environ["KEY"]', "long alias os_module.environ"),
            ('_os.environ.get("KEY")', "alias with .get()"),
            ('_os.environ.setdefault("KEY", "val")', "alias with .setdefault()"),
            ('my_os.environ.pop("KEY")', "alias with .pop()"),
            ("_os.environ.clear()", "alias with .clear()"),
            ('_os.environ.update({"KEY": "val"})', "alias with .update()"),
        ],
    )
    def test_detects_module_alias_patterns(
        self, code_line: str, description: str
    ) -> None:
        """Verify module alias patterns (import os as _os) are detected."""
        matches = [p for p in FORBIDDEN_PATTERNS if p.search(code_line)]
        assert matches, f"Expected to detect: {description} in '{code_line}'"

    # === Bare environ/getenv patterns ===

    @pytest.mark.parametrize(
        ("code_line", "description"),
        [
            ('value = environ["API_KEY"]', "bare environ bracket access"),
            ('value = environ.get("API_KEY")', "bare environ.get"),
            ('environ.setdefault("KEY", "val")', "bare environ.setdefault"),
            ('environ.pop("KEY")', "bare environ.pop"),
            ("environ.clear()", "bare environ.clear"),
            ('environ.update({"KEY": "val"})', "bare environ.update"),
            ('value = getenv("API_KEY")', "bare getenv function"),
        ],
    )
    def test_detects_bare_environ_getenv(
        self, code_line: str, description: str
    ) -> None:
        """Verify bare environ/getenv (from os import environ) are detected."""
        matches = [p for p in FORBIDDEN_PATTERNS if p.search(code_line)]
        assert matches, f"Expected to detect: {description} in '{code_line}'"

    # === False positive prevention (pattern-level) ===
    # Note: Comment and string literal handling is done at scan_file level, not pattern level.
    # These tests verify patterns don't match identifiers that contain environ/getenv as substrings.

    @pytest.mark.parametrize(
        ("code_line", "description"),
        [
            ('my_custom_getenv("KEY")', "custom function named getenv"),
            ("process_environ_data = {}", "variable ending with environ"),
            ('self.environ_dict["KEY"]', "object attribute environ_dict"),
            ("def getenv_wrapper():", "function definition with getenv"),
            ("class EnvironmentLoader:", "class with Environment prefix"),
        ],
    )
    def test_does_not_flag_false_positives(
        self, code_line: str, description: str
    ) -> None:
        """Verify we don't flag false positives."""
        # These should NOT match any pattern
        matches = [p for p in FORBIDDEN_PATTERNS if p.search(code_line)]
        assert not matches, f"Unexpected match for: {description} in '{code_line}'"

    # === Patterns that do match but are filtered at scan_file level ===
    # These patterns DO match at regex level, but scan_file() handles filtering them.

    @pytest.mark.parametrize(
        ("code_line", "description"),
        [
            ('# os.environ["KEY"]', "commented out code - filtered by scan_file"),
            ('s = "os.environ[KEY]"', "string literal - not reliably filtered"),
        ],
    )
    def test_patterns_match_but_filtered_elsewhere(
        self, code_line: str, description: str
    ) -> None:
        """Verify patterns that match but are filtered at scan_file level.

        These test cases document that the regex patterns DO match these lines,
        but scan_file() filters them out (for comments) or they are accepted
        as limitations (for string literals without AST parsing).
        """
        # These DO match at pattern level - that's expected behavior
        matches = [p for p in FORBIDDEN_PATTERNS if p.search(code_line)]
        # Pattern-level match is expected; scan_file handles the filtering
        assert matches, f"Expected pattern match for: {description}"


class TestInlineExclusionMarker:
    """Tests for inline exclusion marker detection."""

    def test_valid_inline_marker(self) -> None:
        """Valid inline marker should be detected."""
        line = 'value = os.getenv("KEY")  # ONEX_EXCLUDE: secret_resolver'
        assert _has_inline_exclusion_marker(line)

    def test_marker_with_whitespace(self) -> None:
        """Marker with various whitespace should be detected."""
        line = 'value = os.getenv("KEY")  #   ONEX_EXCLUDE:   secret_resolver'
        assert _has_inline_exclusion_marker(line)

    def test_marker_in_string_literal_not_detected(self) -> None:
        """Marker inside string literal should not be detected."""
        line = 's = "# ONEX_EXCLUDE: secret_resolver"'
        assert not _has_inline_exclusion_marker(line)

    def test_no_marker(self) -> None:
        """Line without marker should not be detected."""
        line = 'value = os.getenv("KEY")'
        assert not _has_inline_exclusion_marker(line)


class TestExclusionPatterns:
    """Tests for file exclusion pattern matching."""

    def test_excludes_test_prefix_files(self) -> None:
        """Files starting with test_ should be excluded."""
        filepath = Path("/repo/src/test_module.py")
        assert _matches_exclusion_pattern(filepath, "src/test_module.py")

    def test_excludes_test_suffix_files(self) -> None:
        """Files ending with _test.py should be excluded."""
        filepath = Path("/repo/src/module_test.py")
        assert _matches_exclusion_pattern(filepath, "src/module_test.py")

    def test_excludes_conftest(self) -> None:
        """conftest.py should be excluded."""
        filepath = Path("/repo/tests/conftest.py")
        assert _matches_exclusion_pattern(filepath, "tests/conftest.py")

    def test_excludes_pycache_directories(self) -> None:
        """Files in __pycache__ should be excluded."""
        filepath = Path("/repo/src/__pycache__/module.cpython-311.pyc")
        assert _matches_exclusion_pattern(
            filepath, "src/__pycache__/module.cpython-311.pyc"
        )

    def test_does_not_exclude_production_code(self) -> None:
        """Production code files should not be excluded."""
        filepath = Path("/repo/src/handlers/handler_db.py")
        assert not _matches_exclusion_pattern(filepath, "src/handlers/handler_db.py")

    def test_substring_in_path_not_excluded(self) -> None:
        """Path containing 'test' substring should not be excluded if not test file."""
        # "contest_handler.py" contains "test" but is not a test file
        filepath = Path("/repo/src/contest_handler.py")
        assert not _matches_exclusion_pattern(filepath, "src/contest_handler.py")


class TestAllowlistValidation:
    """Tests for allowlist loading and validation."""

    def test_valid_allowlist_format(self, tmp_path: Path) -> None:
        """Valid allowlist entries should be loaded."""
        allowlist_file = tmp_path / ".secretresolver_allowlist"
        allowlist_file.write_text(
            "src/handler.py:42 # migration pending\nsrc/adapter.py:10\n# Comment line\n"
        )

        allowlist = load_allowlist(tmp_path)
        assert "src/handler.py:42" in allowlist
        assert "src/adapter.py:10" in allowlist
        assert len(allowlist) == 2

    def test_malformed_allowlist_raises_error(self, tmp_path: Path) -> None:
        """Malformed allowlist entries should raise AllowlistValidationError."""
        allowlist_file = tmp_path / ".secretresolver_allowlist"
        allowlist_file.write_text(
            "src/handler.py:42\n"
            "invalid entry without line number\n"  # Malformed
            "also:bad:format\n"  # Malformed
        )

        with pytest.raises(AllowlistValidationError) as exc_info:
            load_allowlist(tmp_path)

        # Should have 2 malformed entries
        assert len(exc_info.value.malformed_entries) == 2

    def test_empty_allowlist(self, tmp_path: Path) -> None:
        """Empty/missing allowlist should return empty set."""
        allowlist = load_allowlist(tmp_path)
        assert allowlist == set()


class TestScanFile:
    """Tests for file scanning functionality."""

    def test_detects_violation_in_file(
        self, tmp_path: Path, create_test_file: Callable[[str, str], Path]
    ) -> None:
        """Violations in files should be detected."""
        # Create a test file with a violation
        content = '''"""Module docstring."""
import os

def get_api_key():
    return os.getenv("API_KEY")
'''
        test_file = create_test_file(content, "module.py")

        violations = scan_file(test_file, tmp_path)
        assert len(violations) == 1
        assert violations[0].line_number == 5
        # Pattern contains regex escapes, check for the key part
        assert "getenv" in violations[0].pattern

    def test_inline_exclusion_skips_violation(
        self, tmp_path: Path, create_test_file: Callable[[str, str], Path]
    ) -> None:
        """Lines with ONEX_EXCLUDE marker should not be flagged."""
        content = '''"""Module docstring."""
import os

def get_api_key():
    return os.getenv("API_KEY")  # ONEX_EXCLUDE: secret_resolver
'''
        test_file = create_test_file(content, "module.py")

        violations = scan_file(test_file, tmp_path)
        assert len(violations) == 0

    def test_comment_lines_not_flagged(
        self, tmp_path: Path, create_test_file: Callable[[str, str], Path]
    ) -> None:
        """Commented out code should not be flagged."""
        content = '''"""Module docstring."""
# value = os.getenv("API_KEY")
'''
        test_file = create_test_file(content, "module.py")

        violations = scan_file(test_file, tmp_path)
        assert len(violations) == 0


class TestViolation:
    """Tests for Violation class."""

    def test_allowlist_key_format(self) -> None:
        """Violation allowlist key should be filepath:line_number."""
        v = Violation(
            filepath="src/handler.py",
            line_number=42,
            line_content='os.getenv("KEY")',
            pattern="os.getenv",
        )
        assert v.allowlist_key() == "src/handler.py:42"

    def test_string_representation(self) -> None:
        """Violation string should include file, line, pattern, and content."""
        v = Violation(
            filepath="src/handler.py",
            line_number=42,
            line_content='os.getenv("KEY")',
            pattern=r"os\.getenv",
        )
        s = str(v)
        assert "src/handler.py:42" in s
        assert r"os\.getenv" in s
