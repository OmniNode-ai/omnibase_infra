# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for LocalHandler import validator.

Tests cover:
- Detection of LocalHandler imports in production code
- Various import patterns (standard, aliased, direct module)
- Directory skipping (tests/, __pycache__)
- Result model fields and formatting
- Violation context (file path, line number)

Policy: LocalHandler is test-only and must NEVER be imported in src/omnibase_infra/.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from omnibase_infra.validation.validator_localhandler import (
    validate_localhandler,
    validate_localhandler_ci,
    validate_localhandler_in_file,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def _create_test_file(temp_dir: Path, content: str, filename: str = "test.py") -> Path:
    """Create a test Python file with given content.

    Args:
        temp_dir: Directory to create file in.
        content: Python source code content.
        filename: Name of the file to create.

    Returns:
        Path to created file.
    """
    filepath = temp_dir / filename
    filepath.write_text(dedent(content), encoding="utf-8")
    return filepath


@pytest.fixture
def clean_code() -> str:
    """Fixture providing clean Python code with no LocalHandler imports."""
    return """
    import os
    from pathlib import Path
    """


@pytest.fixture
def localhandler_import_code() -> str:
    """Fixture providing code with standard LocalHandler import."""
    return """
    from omnibase_core.handlers import LocalHandler

    handler = LocalHandler()
    """


@pytest.fixture
def aliased_import_code() -> str:
    """Fixture providing code with aliased LocalHandler import."""
    return """
    from omnibase_core.handlers import LocalHandler as LH

    handler = LH()
    """


@pytest.fixture
def direct_import_code() -> str:
    """Fixture providing code with direct module LocalHandler import."""
    return """
    from omnibase_core.handlers.handler_local import LocalHandler

    handler = LocalHandler()
    """


# =============================================================================
# Detection Tests: No Violations
# =============================================================================


class TestNoLocalHandlerImports:
    """Test that clean files pass validation."""

    def test_pass_with_no_localhandler_imports(self, tmp_path: Path) -> None:
        """Directory with clean Python files should pass validation."""
        code = """
        import os
        from pathlib import Path
        from omnibase_core.handlers import SomeOtherHandler

        class MyHandler:
            pass
        """
        _create_test_file(tmp_path, code, "clean.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0
        assert result.files_checked >= 1

    def test_pass_with_empty_file(self, tmp_path: Path) -> None:
        """Empty Python file should pass validation."""
        _create_test_file(tmp_path, "", "empty.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_pass_with_comments_only(self, tmp_path: Path) -> None:
        """File with only comments should pass validation."""
        code = """
        # This is a comment mentioning LocalHandler
        # from omnibase_core.handlers import LocalHandler
        '''
        LocalHandler is mentioned here too
        '''
        """
        _create_test_file(tmp_path, code, "comments.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_pass_with_similar_names(self, tmp_path: Path) -> None:
        """Files with similar but different names should pass."""
        code = """
        from mymodule import LocalHandlerBase
        from mymodule import LocalHandlerImpl
        from mymodule import MyLocalHandler

        class LocalHandlerWrapper:
            pass
        """
        _create_test_file(tmp_path, code, "similar.py")

        result = validate_localhandler_ci(tmp_path)

        # LocalHandlerBase, LocalHandlerImpl, etc. should not be flagged
        # Only exact "LocalHandler" in import from omnibase_core should be flagged
        assert result.passed


# =============================================================================
# Detection Tests: Import Violations
# =============================================================================


class TestLocalHandlerImportDetection:
    """Test detection of various LocalHandler import patterns."""

    def test_fail_with_localhandler_import(self, tmp_path: Path) -> None:
        """Should detect standard import: from omnibase_core.handlers import LocalHandler."""
        code = """
        from omnibase_core.handlers import LocalHandler

        handler = LocalHandler()
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1
        assert "LocalHandler" in result.violations[0].import_line
        assert result.violations[0].line_number == 2

    def test_fail_with_aliased_import(self, tmp_path: Path) -> None:
        """Should detect aliased import: from omnibase_core.handlers import LocalHandler as LH."""
        code = """
        from omnibase_core.handlers import LocalHandler as LH

        handler = LH()
        """
        _create_test_file(tmp_path, code, "aliased.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1
        assert "LocalHandler" in result.violations[0].import_line
        assert "as LH" in result.violations[0].import_line

    def test_fail_with_direct_import(self, tmp_path: Path) -> None:
        """Should detect direct module import: from omnibase_core.handlers.handler_local import LocalHandler."""
        code = """
        from omnibase_core.handlers.handler_local import LocalHandler

        handler = LocalHandler()
        """
        _create_test_file(tmp_path, code, "direct.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1
        assert "LocalHandler" in result.violations[0].import_line

    def test_fail_with_local_handler_module_import(self, tmp_path: Path) -> None:
        """Should detect import from local_handler module."""
        code = """
        from omnibase_core.handlers.local_handler import LocalHandler

        handler = LocalHandler()
        """
        _create_test_file(tmp_path, code, "local_module.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1
        assert "LocalHandler" in result.violations[0].import_line

    def test_fail_with_multiple_imports_same_line(self, tmp_path: Path) -> None:
        """Should detect LocalHandler in multi-import line."""
        code = """
        from omnibase_core.handlers import Handler, LocalHandler, OtherHandler
        """
        _create_test_file(tmp_path, code, "multi_import.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1
        assert "LocalHandler" in result.violations[0].import_line

    def test_fail_with_import_statement(self, tmp_path: Path) -> None:
        """Should detect import statement: import omnibase_core.handlers.LocalHandler."""
        code = """
        import omnibase_core.handlers.LocalHandler
        """
        _create_test_file(tmp_path, code, "import_stmt.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1

    def test_fail_with_multiple_violations(self, tmp_path: Path) -> None:
        """Should detect multiple LocalHandler imports in same file."""
        code = """
        from omnibase_core.handlers import LocalHandler
        from omnibase_core.handlers.handler_local import LocalHandler as LH
        """
        _create_test_file(tmp_path, code, "multi_violations.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 2

    def test_fail_with_violations_in_multiple_files(self, tmp_path: Path) -> None:
        """Should detect violations across multiple files."""
        code1 = """
        from omnibase_core.handlers import LocalHandler
        """
        code2 = """
        from omnibase_core.handlers.handler_local import LocalHandler
        """
        _create_test_file(tmp_path, code1, "file1.py")
        _create_test_file(tmp_path, code2, "file2.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 2


# =============================================================================
# Directory Skipping Tests
# =============================================================================


class TestDirectorySkipping:
    """Test that certain directories are skipped during validation."""

    def test_skip_test_directories(self, tmp_path: Path) -> None:
        """Should NOT flag violations in tests/ subdirectories."""
        # Create tests subdirectory with violation
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        code = """
        from omnibase_core.handlers import LocalHandler

        def test_something():
            handler = LocalHandler()
        """
        _create_test_file(tests_dir, code, "test_handler.py")

        result = validate_localhandler_ci(tmp_path)

        # Test files should be skipped - LocalHandler is allowed in tests
        assert result.passed
        assert len(result.violations) == 0

    def test_skip_nested_test_directories(self, tmp_path: Path) -> None:
        """Should skip nested tests/ directories."""
        # Create nested tests directory
        nested_tests = tmp_path / "src" / "tests" / "unit"
        nested_tests.mkdir(parents=True)
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(nested_tests, code, "test_file.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_skip_pycache(self, tmp_path: Path) -> None:
        """Should skip __pycache__ directories."""
        # Create __pycache__ directory with violation
        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(pycache_dir, code, "cached.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_skip_venv_directory(self, tmp_path: Path) -> None:
        """Should skip venv/ directories."""
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(venv_dir, code, "module.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_skip_dot_venv_directory(self, tmp_path: Path) -> None:
        """Should skip .venv/ directories."""
        venv_dir = tmp_path / ".venv"
        venv_dir.mkdir()
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(venv_dir, code, "module.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_skip_git_directory(self, tmp_path: Path) -> None:
        """Should skip .git/ directories."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(git_dir, code, "hook.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_skip_underscore_prefixed_files(self, tmp_path: Path) -> None:
        """Should skip files starting with underscore."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "_private.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert len(result.violations) == 0

    def test_detect_in_non_skipped_directories(self, tmp_path: Path) -> None:
        """Should still detect violations in non-skipped directories."""
        # Create src subdirectory (not skipped)
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(src_dir, code, "handler.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1


# =============================================================================
# Result Model Tests
# =============================================================================


class TestResultModelFields:
    """Test result model has correct fields and values."""

    def test_result_model_fields(self, tmp_path: Path) -> None:
        """Verify result has passed, violations, files_checked fields."""
        code = """
        import os
        """
        _create_test_file(tmp_path, code, "clean.py")

        result = validate_localhandler_ci(tmp_path)

        # Verify all required fields exist
        assert hasattr(result, "passed")
        assert hasattr(result, "violations")
        assert hasattr(result, "files_checked")

        # Verify field types
        assert isinstance(result.passed, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.files_checked, int)

    def test_result_passed_true_when_clean(self, tmp_path: Path) -> None:
        """Result.passed should be True when no violations."""
        code = """
        import os
        """
        _create_test_file(tmp_path, code, "clean.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed is True

    def test_result_passed_false_when_violations(self, tmp_path: Path) -> None:
        """Result.passed should be False when violations exist."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.passed is False

    def test_result_files_checked_count(self, tmp_path: Path) -> None:
        """Result.files_checked should reflect number of files validated."""
        code = """
        import os
        """
        _create_test_file(tmp_path, code, "file1.py")
        _create_test_file(tmp_path, code, "file2.py")
        _create_test_file(tmp_path, code, "file3.py")

        result = validate_localhandler_ci(tmp_path)

        assert result.files_checked == 3

    def test_result_bool_behavior(self, tmp_path: Path) -> None:
        """Result should support boolean conversion based on passed status."""
        clean_code = """
        import os
        """
        bad_code = """
        from omnibase_core.handlers import LocalHandler
        """

        # Clean file
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()
        _create_test_file(clean_dir, clean_code, "clean.py")
        clean_result = validate_localhandler_ci(clean_dir)

        # Bad file
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        _create_test_file(bad_dir, bad_code, "bad.py")
        bad_result = validate_localhandler_ci(bad_dir)

        # bool(result) should equal result.passed
        assert bool(clean_result) is True
        assert bool(bad_result) is False


# =============================================================================
# Violation Model Tests
# =============================================================================


class TestViolationModel:
    """Test violation model has correct context information."""

    def test_violation_has_file_and_line(self, tmp_path: Path) -> None:
        """Verify violations include file path and line number."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        filepath = _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)

        assert len(result.violations) == 1
        violation = result.violations[0]

        # Verify violation has file_path
        assert hasattr(violation, "file_path")
        assert violation.file_path.name == "bad.py"
        assert violation.file_path.is_absolute() or violation.file_path.exists()

        # Verify violation has line_number
        assert hasattr(violation, "line_number")
        assert violation.line_number == 2  # Line 2 after dedent

        # Verify violation has import_line
        assert hasattr(violation, "import_line")
        assert "LocalHandler" in violation.import_line

    def test_violation_line_numbers_accurate(self, tmp_path: Path) -> None:
        """Verify line numbers are accurate for violations."""
        code = """
        import os
        import sys

        # Comment line

        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)

        assert len(result.violations) == 1
        # Line 7 after dedent (accounting for blank first line)
        assert result.violations[0].line_number == 7

    def test_violation_format_for_ci(self, tmp_path: Path) -> None:
        """Verify violation can be formatted for CI output."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)

        assert len(result.violations) == 1
        violation = result.violations[0]

        # Check format_for_ci method exists and produces GitHub Actions format
        ci_output = violation.format_for_ci()
        assert "::error" in ci_output
        assert "bad.py" in ci_output
        assert "line=2" in ci_output
        assert "LocalHandler" in ci_output

    def test_violation_format_human_readable(self, tmp_path: Path) -> None:
        """Verify violation can be formatted for human reading."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)

        assert len(result.violations) == 1
        violation = result.violations[0]

        # Check format_human_readable method
        human_output = violation.format_human_readable()
        assert "bad.py" in human_output
        assert "2" in human_output  # Line number
        assert "LocalHandler" in human_output


# =============================================================================
# Single File Validation Tests
# =============================================================================


class TestSingleFileValidation:
    """Test validate_localhandler_in_file function."""

    def test_validate_single_file_clean(self, tmp_path: Path) -> None:
        """Single clean file should return empty list."""
        code = """
        import os
        """
        filepath = _create_test_file(tmp_path, code, "clean.py")

        violations = validate_localhandler_in_file(filepath)

        assert violations == []

    def test_validate_single_file_with_violation(self, tmp_path: Path) -> None:
        """Single file with violation should return violation list."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        filepath = _create_test_file(tmp_path, code, "bad.py")

        violations = validate_localhandler_in_file(filepath)

        assert len(violations) == 1
        assert "LocalHandler" in violations[0].import_line


# =============================================================================
# Directory Validation Tests
# =============================================================================


class TestDirectoryValidation:
    """Test validate_localhandler function for directory scanning."""

    def test_validate_directory_recursive(self, tmp_path: Path) -> None:
        """Recursive validation should find violations in subdirectories."""
        # Create nested directory with violation
        nested = tmp_path / "src" / "handlers"
        nested.mkdir(parents=True)
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(nested, code, "bad_handler.py")

        violations = validate_localhandler(tmp_path, recursive=True)

        assert len(violations) == 1

    def test_validate_directory_non_recursive(self, tmp_path: Path) -> None:
        """Non-recursive validation should only check immediate directory."""
        # Create file in root
        code_root = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code_root, "root.py")

        # Create subdirectory with file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        code_sub = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(subdir, code_sub, "sub.py")

        violations = validate_localhandler(tmp_path, recursive=False)

        # Only root file should be checked
        assert len(violations) == 1
        assert "root.py" in str(violations[0].file_path)


# =============================================================================
# Result Formatting Tests
# =============================================================================


class TestResultFormatting:
    """Test result formatting methods."""

    def test_format_summary_passed(self, tmp_path: Path) -> None:
        """Format summary for passed validation."""
        code = """
        import os
        """
        _create_test_file(tmp_path, code, "clean.py")

        result = validate_localhandler_ci(tmp_path)
        summary = result.format_summary()

        assert "PASS" in summary
        assert "Files checked:" in summary
        assert "No LocalHandler imports" in summary

    def test_format_summary_failed(self, tmp_path: Path) -> None:
        """Format summary for failed validation."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)
        summary = result.format_summary()

        assert "FAIL" in summary
        assert "LocalHandler" in summary
        assert "test-only" in summary.lower()

    def test_format_for_ci_output(self, tmp_path: Path) -> None:
        """Format for CI produces GitHub Actions annotations."""
        code = """
        from omnibase_core.handlers import LocalHandler
        """
        _create_test_file(tmp_path, code, "bad.py")

        result = validate_localhandler_ci(tmp_path)
        ci_output = result.format_for_ci()

        assert len(ci_output) == 1
        assert "::error" in ci_output[0]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should pass with zero files checked."""
        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert result.files_checked == 0
        assert len(result.violations) == 0

    def test_non_python_files_ignored(self, tmp_path: Path) -> None:
        """Non-Python files should be ignored."""
        (tmp_path / "readme.md").write_text(
            "from omnibase_core.handlers import LocalHandler",
            encoding="utf-8",
        )
        (tmp_path / "config.yaml").write_text(
            "handler: omnibase_core.handlers.LocalHandler",
            encoding="utf-8",
        )
        (tmp_path / "data.json").write_text(
            '{"import": "from omnibase_core.handlers import LocalHandler"}',
            encoding="utf-8",
        )

        result = validate_localhandler_ci(tmp_path)

        assert result.passed
        assert result.files_checked == 0

    def test_indented_import(self, tmp_path: Path) -> None:
        """Should detect indented imports (inside functions/classes)."""
        code = """
        class MyClass:
            def setup(self):
                from omnibase_core.handlers import LocalHandler
                return LocalHandler()
        """
        _create_test_file(tmp_path, code, "indented.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1

    def test_whitespace_variations(self, tmp_path: Path) -> None:
        """Should detect imports with various whitespace patterns."""
        code = """
        from   omnibase_core.handlers   import   LocalHandler
        """
        _create_test_file(tmp_path, code, "whitespace.py")

        result = validate_localhandler_ci(tmp_path)

        assert not result.passed
        assert len(result.violations) == 1
