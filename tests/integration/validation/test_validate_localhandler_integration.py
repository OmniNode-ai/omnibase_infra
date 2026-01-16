# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for validate.py localhandler command.

These tests verify the actual CLI execution of the LocalHandler validator
via subprocess, testing exit codes and ensuring CI gate behavior works correctly.

Test Coverage:
- TestValidateLocalHandlerIntegration: CLI exit code verification
  - Exit code 0 when no violations found
  - Non-zero exit code when violations found
  - Verbose mode output
  - Multiple file scanning

Related:
    - OMN-743: Enforce LocalHandler dev-only usage
    - PR #162: Added integration tests for LocalHandler validation
    - tests/unit/validation/test_validator_localhandler.py: Unit tests

Policy:
    LocalHandler is a test-only handler that must NEVER be imported in
    src/omnibase_infra/. The validate.py localhandler command enforces
    this policy in CI by returning exit code 0 for clean code and
    non-zero for violations.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _get_validate_script_path() -> Path:
    """Get the absolute path to the validate.py script.

    Returns:
        Path to scripts/validate.py relative to the repository root.
    """
    # Navigate from test file location to repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    return repo_root / "scripts" / "validate.py"


class TestValidateLocalHandlerIntegration:
    """Integration tests for validate.py localhandler command.

    These tests run the actual CLI command via subprocess to verify
    that the CI gate works correctly with proper exit codes.
    """

    def test_localhandler_passes_on_clean_directory(self, tmp_path: Path) -> None:
        """Verify exit code 0 when no violations found.

        Creates a clean src/omnibase_infra directory with no LocalHandler
        imports and verifies the validator returns exit code 0.
        """
        # Create clean src structure (required by validate.py)
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        # Create clean Python file with standard imports only
        clean_file = src_dir / "clean_module.py"
        clean_file.write_text(
            """\
import os
from pathlib import Path
from typing import Protocol


class MyProtocol(Protocol):
    def execute(self) -> None:
        ...
"""
        )

        # Run validator from temp directory
        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, (
            f"Expected exit code 0 for clean directory, got {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_localhandler_fails_on_violation(self, tmp_path: Path) -> None:
        """Verify non-zero exit code when violations found.

        Creates a src/omnibase_infra directory containing a LocalHandler
        import and verifies the validator returns non-zero exit code.
        """
        # Create src structure with LocalHandler import (violation)
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        bad_file = src_dir / "bad_handler.py"
        bad_file.write_text(
            """\
from omnibase_core.handlers import LocalHandler

handler = LocalHandler()
"""
        )

        # Run validator from temp directory
        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode != 0, (
            f"Expected non-zero exit code for violation, got {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_localhandler_detects_aliased_import(self, tmp_path: Path) -> None:
        """Verify detection of aliased LocalHandler imports.

        Tests that imports like 'from module import LocalHandler as LH'
        are correctly detected as violations.
        """
        # Create src structure with aliased LocalHandler import
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        aliased_file = src_dir / "aliased_handler.py"
        aliased_file.write_text(
            """\
from omnibase_core.handlers import LocalHandler as LH

handler = LH()
"""
        )

        # Run validator from temp directory
        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode != 0, (
            f"Expected non-zero exit code for aliased import violation.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_localhandler_verbose_output_shows_details(self, tmp_path: Path) -> None:
        """Verify verbose mode includes file and line information.

        When running with --verbose, the output should include details
        about the violations found.
        """
        # Create src structure with LocalHandler import (violation)
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        bad_file = src_dir / "verbose_test.py"
        bad_file.write_text(
            """\
from omnibase_core.handlers import LocalHandler
"""
        )

        # Run validator with verbose flag
        result = subprocess.run(
            [
                sys.executable,
                str(_get_validate_script_path()),
                "localhandler",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0

        # Verbose output should contain file path reference
        assert "verbose_test.py" in result.stdout or "verbose_test.py" in result.stderr

    def test_localhandler_multiple_violations_all_detected(
        self, tmp_path: Path
    ) -> None:
        """Verify multiple violations across files are all detected.

        Creates multiple files with LocalHandler imports and verifies
        the validator detects violations in all of them.
        """
        # Create src structure with multiple violations
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        # File 1: Standard import
        (src_dir / "file1.py").write_text(
            """\
from omnibase_core.handlers import LocalHandler
"""
        )

        # File 2: Direct module import
        (src_dir / "file2.py").write_text(
            """\
from omnibase_core.handlers.handler_local import LocalHandler
"""
        )

        # Run validator with verbose flag to see all violations
        result = subprocess.run(
            [
                sys.executable,
                str(_get_validate_script_path()),
                "localhandler",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0

        # Output should indicate multiple violations (2 violations found)
        output = result.stdout + result.stderr
        assert "2" in output or "violations" in output.lower()

    def test_localhandler_passes_when_src_omnibase_infra_missing(
        self, tmp_path: Path
    ) -> None:
        """Verify validator handles missing src/omnibase_infra gracefully.

        When the src/omnibase_infra directory does not exist, the validator
        should skip validation and return exit code 0 (pass).
        """
        # Create empty directory (no src/omnibase_infra)
        # tmp_path is already empty

        # Run validator from temp directory
        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Should pass (skip) when directory doesn't exist
        assert result.returncode == 0, (
            f"Expected exit code 0 when src/omnibase_infra missing.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_localhandler_ignores_test_directories(self, tmp_path: Path) -> None:
        """Verify test directories are not scanned for violations.

        LocalHandler imports in tests/ directories should be allowed
        since LocalHandler is specifically for testing purposes.
        """
        # Create src structure (clean production code)
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)
        (src_dir / "clean.py").write_text(
            """\
import os
"""
        )

        # Create tests directory with LocalHandler import (should be allowed)
        tests_dir = tmp_path / "src" / "omnibase_infra" / "tests"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_handler.py").write_text(
            """\
from omnibase_core.handlers import LocalHandler

def test_something():
    handler = LocalHandler()
"""
        )

        # Run validator from temp directory
        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Should pass because tests/ directories are skipped
        assert result.returncode == 0, (
            f"Expected exit code 0 when only tests/ has LocalHandler.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_localhandler_nested_subdirectory_violation(self, tmp_path: Path) -> None:
        """Verify violations in nested subdirectories are detected.

        LocalHandler imports should be detected regardless of how deep
        in the directory structure they appear.
        """
        # Create deeply nested src structure with violation
        nested_dir = tmp_path / "src" / "omnibase_infra" / "handlers" / "auth"
        nested_dir.mkdir(parents=True)

        (nested_dir / "deep_handler.py").write_text(
            """\
from omnibase_core.handlers import LocalHandler

class AuthHandler:
    def __init__(self):
        self.local = LocalHandler()
"""
        )

        # Run validator from temp directory
        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Should fail - nested violations must be detected
        assert result.returncode != 0, (
            f"Expected non-zero exit code for nested violation.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


class TestValidateLocalHandlerCIIntegration:
    """Tests verifying CI-specific behavior of the LocalHandler validator.

    These tests focus on behaviors important for CI/CD integration:
    - Exit codes for pass/fail
    - Output format compatibility with CI systems
    - Consistent behavior across different scenarios
    """

    def test_exit_code_zero_is_pass(self, tmp_path: Path) -> None:
        """Verify exit code 0 indicates validation passed.

        This is the fundamental CI contract: exit code 0 means success.
        """
        # Create clean src structure
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)
        (src_dir / "ok.py").write_text("import os\n")

        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0

    def test_exit_code_one_is_fail(self, tmp_path: Path) -> None:
        """Verify exit code 1 indicates validation failed.

        Exit code 1 (or non-zero) means violations were found.
        """
        # Create src structure with violation
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)
        (src_dir / "bad.py").write_text(
            "from omnibase_core.handlers import LocalHandler\n"
        )

        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 1, (
            f"Expected exit code 1 for failed validation, got {result.returncode}"
        )

    def test_stdout_contains_pass_or_fail_indicator(self, tmp_path: Path) -> None:
        """Verify stdout contains human-readable pass/fail indicator.

        The output should clearly indicate whether validation passed or failed
        for developers reading CI logs.
        """
        # Test passing case
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)
        (src_dir / "clean.py").write_text("import os\n")

        pass_result = subprocess.run(
            [
                sys.executable,
                str(_get_validate_script_path()),
                "localhandler",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Verbose mode should show PASS for clean code
        assert "PASS" in pass_result.stdout, (
            f"Expected 'PASS' in stdout for clean code.\nstdout: {pass_result.stdout}"
        )

        # Test failing case
        (src_dir / "bad.py").write_text(
            "from omnibase_core.handlers import LocalHandler\n"
        )

        fail_result = subprocess.run(
            [
                sys.executable,
                str(_get_validate_script_path()),
                "localhandler",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        # Should show FAIL indicator for violations
        assert "FAIL" in fail_result.stdout, (
            f"Expected 'FAIL' in stdout for violation.\nstdout: {fail_result.stdout}"
        )


class TestValidateLocalHandlerEdgeCases:
    """Tests for edge cases and error handling in LocalHandler validation."""

    def test_empty_python_file_passes(self, tmp_path: Path) -> None:
        """Verify empty Python files do not cause errors."""
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)
        (src_dir / "empty.py").write_text("")

        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0

    def test_python_file_with_only_comments_passes(self, tmp_path: Path) -> None:
        """Verify files with only comments pass validation."""
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)
        (src_dir / "comments.py").write_text(
            """\
# This is a comment mentioning LocalHandler
# from omnibase_core.handlers import LocalHandler
'''
Docstring mentioning LocalHandler should not trigger.
'''
"""
        )

        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, (
            f"Comments mentioning LocalHandler should not trigger violation.\n"
            f"stdout: {result.stdout}"
        )

    def test_non_python_files_ignored(self, tmp_path: Path) -> None:
        """Verify non-Python files are not scanned."""
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        # Create non-Python files with LocalHandler references
        (src_dir / "readme.md").write_text(
            "from omnibase_core.handlers import LocalHandler"
        )
        (src_dir / "config.yaml").write_text(
            "handler: omnibase_core.handlers.LocalHandler"
        )
        (src_dir / "clean.py").write_text("import os\n")

        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, (
            f"Non-Python files should be ignored.\nstdout: {result.stdout}"
        )

    def test_underscore_prefixed_files_skipped(self, tmp_path: Path) -> None:
        """Verify files starting with underscore are skipped.

        Files like _private.py are typically internal/test fixtures
        and should be excluded from validation.
        """
        src_dir = tmp_path / "src" / "omnibase_infra"
        src_dir.mkdir(parents=True)

        # Private file with violation (should be skipped)
        (src_dir / "_private.py").write_text(
            "from omnibase_core.handlers import LocalHandler\n"
        )

        # Public file that is clean
        (src_dir / "public.py").write_text("import os\n")

        result = subprocess.run(
            [sys.executable, str(_get_validate_script_path()), "localhandler"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            check=False,
        )

        assert result.returncode == 0, (
            f"Underscore-prefixed files should be skipped.\nstdout: {result.stdout}"
        )
