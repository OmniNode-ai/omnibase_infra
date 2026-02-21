# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for root directory cleanliness validator.

Tests the validate_clean_root.py script that enforces a clean, organized
repository root structure suitable for public release.

This module tests:
- Allowed files pass validation (README.md, pyproject.toml, etc.)
- Allowed directories pass validation (src/, tests/, docs/, etc.)
- Violations are detected correctly for unauthorized files
- Pattern matching works (.env.*, *.egg-info patterns)
- _suggest_action() returns appropriate suggestions
- ValidationResult.__bool__() custom behavior
- RootViolation.__str__() formatting
- Edge cases (empty directory, non-existent path, path is file)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Import the module from scripts/validation/validate_clean_root.py
# Since scripts/ is not a package, use importlib to load it
_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts/validation/validate_clean_root.py"
)
_spec = importlib.util.spec_from_file_location("validate_clean_root", _SCRIPT_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module from {_SCRIPT_PATH}")
validate_clean_root_module = importlib.util.module_from_spec(_spec)
sys.modules["validate_clean_root"] = validate_clean_root_module
_spec.loader.exec_module(validate_clean_root_module)

# Import the components we need to test
validate_root_directory = validate_clean_root_module.validate_root_directory
generate_report = validate_clean_root_module.generate_report
_matches_pattern = validate_clean_root_module._matches_pattern
_suggest_action = validate_clean_root_module._suggest_action
ValidationResult = validate_clean_root_module.ValidationResult
RootViolation = validate_clean_root_module.RootViolation
ALLOWED_ROOT_FILES = validate_clean_root_module.ALLOWED_ROOT_FILES
ALLOWED_ROOT_DIRECTORIES = validate_clean_root_module.ALLOWED_ROOT_DIRECTORIES
ALLOWED_ROOT_PATTERNS = validate_clean_root_module.ALLOWED_ROOT_PATTERNS


class TestAllowedFilesPassValidation:
    """Test that allowed files pass validation without violations."""

    def test_readme_allowed(self, tmp_path: Path) -> None:
        """Test that README.md passes validation."""
        (tmp_path / "README.md").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0
        assert result.checked_items == 1

    def test_pyproject_toml_allowed(self, tmp_path: Path) -> None:
        """Test that pyproject.toml passes validation."""
        (tmp_path / "pyproject.toml").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_gitignore_allowed(self, tmp_path: Path) -> None:
        """Test that .gitignore passes validation."""
        (tmp_path / ".gitignore").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_license_allowed(self, tmp_path: Path) -> None:
        """Test that LICENSE passes validation."""
        (tmp_path / "LICENSE").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_multiple_allowed_files(self, tmp_path: Path) -> None:
        """Test multiple allowed files all pass validation."""
        allowed_files = [
            "README.md",
            "pyproject.toml",
            ".gitignore",
            "LICENSE",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            ".pre-commit-config.yaml",
            "Makefile",
        ]
        for file_name in allowed_files:
            (tmp_path / file_name).touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0
        assert result.checked_items == len(allowed_files)

    def test_onex_specific_files_allowed(self, tmp_path: Path) -> None:
        """Test ONEX-specific files like CLAUDE.md pass validation."""
        (tmp_path / "CLAUDE.md").touch()
        (tmp_path / ".env.example").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_docker_files_allowed(self, tmp_path: Path) -> None:
        """Test Docker-related files pass validation."""
        docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        for file_name in docker_files:
            (tmp_path / file_name).touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_ci_config_files_allowed(self, tmp_path: Path) -> None:
        """Test CI/CD configuration files pass validation."""
        ci_files = [".travis.yml", "tox.ini", "noxfile.py"]
        for file_name in ci_files:
            (tmp_path / file_name).touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0


class TestAllowedDirectoriesPassValidation:
    """Test that allowed directories pass validation without violations."""

    def test_src_directory_allowed(self, tmp_path: Path) -> None:
        """Test that src/ directory passes validation."""
        (tmp_path / "src").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_tests_directory_allowed(self, tmp_path: Path) -> None:
        """Test that tests/ directory passes validation."""
        (tmp_path / "tests").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_docs_directory_allowed(self, tmp_path: Path) -> None:
        """Test that docs/ directory passes validation."""
        (tmp_path / "docs").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_scripts_directory_allowed(self, tmp_path: Path) -> None:
        """Test that scripts/ directory passes validation."""
        (tmp_path / "scripts").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_multiple_allowed_directories(self, tmp_path: Path) -> None:
        """Test multiple allowed directories all pass validation."""
        allowed_dirs = [
            "src",
            "tests",
            "docs",
            "scripts",
            "contracts",
            "examples",
            "benchmarks",
            "tools",
        ]
        for dir_name in allowed_dirs:
            (tmp_path / dir_name).mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0
        assert result.checked_items == len(allowed_dirs)

    def test_hidden_directories_allowed(self, tmp_path: Path) -> None:
        """Test hidden directories like .git, .github pass validation."""
        hidden_dirs = [".git", ".github", ".vscode", ".claude"]
        for dir_name in hidden_dirs:
            (tmp_path / dir_name).mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_cache_directories_allowed(self, tmp_path: Path) -> None:
        """Test cache directories pass validation (should be gitignored)."""
        cache_dirs = [".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__"]
        for dir_name in cache_dirs:
            (tmp_path / dir_name).mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_venv_directories_allowed(self, tmp_path: Path) -> None:
        """Test virtual environment directories pass validation."""
        venv_dirs = [".venv", "venv"]
        for dir_name in venv_dirs:
            (tmp_path / dir_name).mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_build_directories_allowed(self, tmp_path: Path) -> None:
        """Test build/dist directories pass validation."""
        build_dirs = ["build", "dist", "htmlcov"]
        for dir_name in build_dirs:
            (tmp_path / dir_name).mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0


class TestViolationsDetected:
    """Test that violations are correctly detected for unauthorized items."""

    def test_disallowed_markdown_detected(self, tmp_path: Path) -> None:
        """Test that unauthorized markdown files are detected as violations."""
        (tmp_path / "WORKING_NOTES.md").touch()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == 1
        assert result.violations[0].path.name == "WORKING_NOTES.md"
        assert not result.violations[0].is_directory

    def test_disallowed_directory_detected(self, tmp_path: Path) -> None:
        """Test that unauthorized directories are detected as violations."""
        (tmp_path / "my_custom_dir").mkdir()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == 1
        assert result.violations[0].path.name == "my_custom_dir"
        assert result.violations[0].is_directory

    def test_multiple_violations_detected(self, tmp_path: Path) -> None:
        """Test that multiple violations are all detected."""
        (tmp_path / "EXECUTION_PLAN.md").touch()
        (tmp_path / "random_file.txt").touch()
        (tmp_path / "custom_directory").mkdir()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == 3

    def test_mixed_allowed_and_disallowed(self, tmp_path: Path) -> None:
        """Test validation with mix of allowed and disallowed items."""
        # Allowed items
        (tmp_path / "README.md").touch()
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "src").mkdir()

        # Disallowed items
        (tmp_path / "MIGRATION_NOTES.md").touch()
        (tmp_path / "custom_dir").mkdir()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == 2
        assert result.checked_items == 5

        violation_names = {v.path.name for v in result.violations}
        assert "MIGRATION_NOTES.md" in violation_names
        assert "custom_dir" in violation_names

    def test_working_document_patterns_detected(self, tmp_path: Path) -> None:
        """Test that working document patterns are detected as violations."""
        working_docs = [
            "EXECUTION_PLAN.md",
            "ENHANCEMENT_SUMMARY.md",
            "ERROR_HANDLING_FIX.md",
            "MIGRATION_LOG.md",
            "TODO_LIST.md",
            "DESIGN_NOTES.md",
        ]
        for doc in working_docs:
            (tmp_path / doc).touch()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == len(working_docs)


class TestPatternMatching:
    """Test pattern-based allowlist matching."""

    def test_env_pattern_is_violation(self, tmp_path: Path) -> None:
        """Test that .env.* files are violations (not in ALLOWED_ROOT_PATTERNS).

        .env variants like .env.local, .env.production are intentionally excluded
        from the allowlist. These files must live in ~/.omnibase/ or be managed
        by Infisical. Only .env.example and .env.template are explicitly allowed.
        """
        env_files = [".env.local", ".env.production", ".env.test"]
        for env_file in env_files:
            (tmp_path / env_file).touch()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == len(env_files)

    def test_egg_info_pattern_matches(self, tmp_path: Path) -> None:
        """Test that *.egg-info pattern matches build artifacts."""
        # Note: egg-info is typically a directory, but the pattern applies to files too
        (tmp_path / "omnibase_infra.egg-info").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_matches_pattern_function_positive(self) -> None:
        """Test _matches_pattern returns True for matching patterns."""
        assert _matches_pattern("package.egg-info", ("*.egg-info",))
        assert _matches_pattern("my_package.egg-info", ("*.egg-info",))

    def test_matches_pattern_function_negative(self) -> None:
        """Test _matches_pattern returns False for non-matching patterns."""
        assert not _matches_pattern("random.txt", ("*.egg-info",))
        assert not _matches_pattern("egg-info", ("*.egg-info",))  # no prefix

    def test_matches_pattern_multiple_patterns(self) -> None:
        """Test _matches_pattern with multiple patterns."""
        patterns = ("*.egg-info", "*.tmp")
        assert _matches_pattern("package.egg-info", patterns)
        assert _matches_pattern("temp.tmp", patterns)
        assert not _matches_pattern("random.txt", patterns)


class TestSuggestAction:
    """Test _suggest_action returns appropriate suggestions."""

    def test_plan_document_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for planning documents."""
        item = tmp_path / "EXECUTION_PLAN.md"
        item.touch()

        suggestion = _suggest_action(item)

        assert "docs/" in suggestion.lower() or "delete" in suggestion.lower()

    def test_notes_document_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for notes documents."""
        item = tmp_path / "WORKING_NOTES.md"
        item.touch()

        suggestion = _suggest_action(item)

        assert "docs/" in suggestion.lower() or "delete" in suggestion.lower()

    def test_log_file_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for log files."""
        item = tmp_path / "debug.log"
        item.touch()

        suggestion = _suggest_action(item)

        # Log files should suggest deletion or gitignore
        assert "delete" in suggestion.lower() or ".gitignore" in suggestion.lower()

    def test_coverage_file_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for coverage reports."""
        item = tmp_path / "coverage_report.html"
        item.touch()

        suggestion = _suggest_action(item)

        assert "delete" in suggestion.lower() or ".gitignore" in suggestion.lower()

    def test_unknown_file_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for unknown files."""
        item = tmp_path / "random_file.xyz"
        item.touch()

        suggestion = _suggest_action(item)

        # Should suggest moving to appropriate directory
        assert "move" in suggestion.lower() or "delete" in suggestion.lower()

    def test_unknown_directory_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for unknown directories."""
        item = tmp_path / "custom_directory"
        item.mkdir()

        suggestion = _suggest_action(item)

        # Should suggest moving contents or adding to allowlist
        assert "move" in suggestion.lower() or "ALLOWED_ROOT_DIRECTORIES" in suggestion

    def test_markdown_without_keywords_suggestion(self, tmp_path: Path) -> None:
        """Test suggestion for markdown files without recognized keywords."""
        item = tmp_path / "RANDOM_DOCUMENT.md"
        item.touch()

        suggestion = _suggest_action(item)

        assert "docs/" in suggestion.lower()


class TestValidationResultBoolBehavior:
    """Test ValidationResult.__bool__() custom behavior."""

    def test_bool_true_when_valid(self) -> None:
        """Test that bool(result) returns True when no violations."""
        result = ValidationResult(violations=[], checked_items=5)

        assert bool(result) is True
        assert result.is_valid is True
        # Can use result directly in conditionals
        if result:
            passed = True
        else:
            passed = False
        assert passed is True

    def test_bool_false_when_violations(self) -> None:
        """Test that bool(result) returns False when violations exist."""
        violation = RootViolation(
            path=Path("/test/file.md"),
            suggestion="Move to docs/",
            is_directory=False,
        )
        result = ValidationResult(violations=[violation], checked_items=1)

        assert bool(result) is False
        assert result.is_valid is False
        # Can use result directly in conditionals
        if result:
            passed = True
        else:
            passed = False
        assert passed is False

    def test_bool_false_with_multiple_violations(self) -> None:
        """Test bool behavior with multiple violations."""
        violations = [
            RootViolation(
                path=Path("/test/file1.md"), suggestion="Move", is_directory=False
            ),
            RootViolation(
                path=Path("/test/file2.md"), suggestion="Delete", is_directory=False
            ),
        ]
        result = ValidationResult(violations=violations, checked_items=10)

        assert bool(result) is False
        assert not result  # Pythonic check

    def test_is_valid_matches_bool(self) -> None:
        """Test that is_valid property always matches bool() result."""
        # Valid case
        valid_result = ValidationResult(violations=[], checked_items=5)
        assert valid_result.is_valid == bool(valid_result)

        # Invalid case
        violation = RootViolation(
            path=Path("/test/x"), suggestion="Y", is_directory=False
        )
        invalid_result = ValidationResult(violations=[violation], checked_items=1)
        assert invalid_result.is_valid == bool(invalid_result)


class TestRootViolationStrFormatting:
    """Test RootViolation.__str__() formatting."""

    def test_file_violation_str(self) -> None:
        """Test string representation of file violation."""
        violation = RootViolation(
            path=Path("/repo/WORKING_NOTES.md"),
            suggestion="Move to docs/ or delete if no longer relevant",
            is_directory=False,
        )

        str_repr = str(violation)

        assert "File:" in str_repr
        assert "WORKING_NOTES.md" in str_repr
        assert "docs/" in str_repr

    def test_directory_violation_str(self) -> None:
        """Test string representation of directory violation."""
        violation = RootViolation(
            path=Path("/repo/custom_dir"),
            suggestion="Move contents to appropriate location",
            is_directory=True,
        )

        str_repr = str(violation)

        assert "Directory:" in str_repr
        assert "custom_dir" in str_repr
        assert "Move contents" in str_repr

    def test_str_format_with_arrow(self) -> None:
        """Test that string format includes arrow for suggestion."""
        violation = RootViolation(
            path=Path("/test/file.txt"),
            suggestion="Delete this file",
            is_directory=False,
        )

        str_repr = str(violation)

        assert "->" in str_repr or "→" in str_repr


class TestGenerateReport:
    """Test generate_report function output."""

    def test_clean_report(self, tmp_path: Path) -> None:
        """Test report for clean directory."""
        (tmp_path / "README.md").touch()
        result = validate_root_directory(tmp_path)

        report = generate_report(result, tmp_path)

        assert "clean" in report.lower()
        assert "1 items checked" in report

    def test_violation_report_header(self, tmp_path: Path) -> None:
        """Test report header for violations."""
        (tmp_path / "WORKING_NOTES.md").touch()
        result = validate_root_directory(tmp_path)

        report = generate_report(result, tmp_path)

        assert "FAILED" in report
        assert "1 item(s)" in report or "1 items" in report

    def test_violation_report_contains_violations(self, tmp_path: Path) -> None:
        """Test that report contains all violation details."""
        (tmp_path / "FILE1.md").touch()
        (tmp_path / "FILE2.md").touch()
        result = validate_root_directory(tmp_path)

        report = generate_report(result, tmp_path)

        assert "FILE1.md" in report
        assert "FILE2.md" in report

    def test_report_contains_guidance(self, tmp_path: Path) -> None:
        """Test that report contains guidance sections."""
        (tmp_path / "RANDOM.md").touch()
        result = validate_root_directory(tmp_path)

        report = generate_report(result, tmp_path)

        assert "WHY THIS MATTERS" in report
        assert "HOW TO FIX" in report
        assert "docs/" in report


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test validation of empty directory."""
        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0
        assert result.checked_items == 0

    def test_nonexistent_directory_raises_error(self, tmp_path: Path) -> None:
        """Test that non-existent directory raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_root_directory(nonexistent)

        assert "does not exist" in str(exc_info.value)

    def test_path_is_file_raises_error(self, tmp_path: Path) -> None:
        """Test that passing a file path raises ValueError."""
        file_path = tmp_path / "somefile.txt"
        file_path.touch()

        with pytest.raises(ValueError) as exc_info:
            validate_root_directory(file_path)

        assert "not a directory" in str(exc_info.value)

    def test_symlink_file(self, tmp_path: Path) -> None:
        """Test handling of symbolic links to files."""
        # Create a real file and symlink to it
        real_file = tmp_path / "real_file.md"
        real_file.touch()

        # Symlink that looks like a violation
        symlink = tmp_path / "NOTES.md"
        symlink.symlink_to(real_file)

        result = validate_root_directory(tmp_path)

        # Both should be detected - real file is allowed (.md with no keywords)
        # but NOTES.md (symlink) has "notes" keyword and is a violation
        # Actually, real_file.md is also a violation since it's not in allowed list
        assert len(result.violations) == 2

    def test_verbose_mode_does_not_affect_result(self, tmp_path: Path) -> None:
        """Test that verbose mode doesn't change validation result."""
        (tmp_path / "README.md").touch()
        (tmp_path / "VIOLATION.md").touch()

        result_normal = validate_root_directory(tmp_path, verbose=False)
        result_verbose = validate_root_directory(tmp_path, verbose=True)

        assert result_normal.is_valid == result_verbose.is_valid
        assert len(result_normal.violations) == len(result_verbose.violations)


class TestAllowlistCompleteness:
    """Test that the allowlists are complete and consistent."""

    def test_allowed_files_is_frozenset(self) -> None:
        """Test that ALLOWED_ROOT_FILES is immutable."""
        assert isinstance(ALLOWED_ROOT_FILES, frozenset)

    def test_allowed_directories_is_frozenset(self) -> None:
        """Test that ALLOWED_ROOT_DIRECTORIES is immutable."""
        assert isinstance(ALLOWED_ROOT_DIRECTORIES, frozenset)

    def test_allowed_patterns_is_tuple(self) -> None:
        """Test that ALLOWED_ROOT_PATTERNS is a tuple."""
        assert isinstance(ALLOWED_ROOT_PATTERNS, tuple)

    def test_common_files_in_allowlist(self) -> None:
        """Test that common project files are in the allowlist."""
        common_files = [
            "README.md",
            "pyproject.toml",
            "LICENSE",
            ".gitignore",
            "Makefile",
        ]
        for file_name in common_files:
            assert file_name in ALLOWED_ROOT_FILES, f"{file_name} should be allowed"

    def test_common_directories_in_allowlist(self) -> None:
        """Test that common project directories are in the allowlist."""
        common_dirs = ["src", "tests", "docs", "scripts", ".git", ".github"]
        for dir_name in common_dirs:
            assert dir_name in ALLOWED_ROOT_DIRECTORIES, f"{dir_name} should be allowed"

    def test_env_pattern_not_in_patterns(self) -> None:
        """Test that .env.* pattern is NOT in ALLOWED_ROOT_PATTERNS.

        .env variants are intentionally excluded — they must live in
        ~/.omnibase/ or be managed by Infisical at runtime.
        """
        assert ".env.*" not in ALLOWED_ROOT_PATTERNS

    def test_egg_info_pattern_in_patterns(self) -> None:
        """Test that *.egg-info pattern is in ALLOWED_ROOT_PATTERNS."""
        assert "*.egg-info" in ALLOWED_ROOT_PATTERNS


class TestIntegrationScenarios:
    """Integration tests for realistic repository scenarios."""

    def test_typical_python_project_passes(self, tmp_path: Path) -> None:
        """Test that a typical Python project structure passes validation."""
        # Files
        (tmp_path / "README.md").touch()
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / "uv.lock").touch()
        (tmp_path / ".gitignore").touch()
        (tmp_path / "LICENSE").touch()
        (tmp_path / "CONTRIBUTING.md").touch()
        (tmp_path / ".pre-commit-config.yaml").touch()

        # Directories
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "docs").mkdir()
        (tmp_path / ".git").mkdir()
        (tmp_path / ".github").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_onex_project_passes(self, tmp_path: Path) -> None:
        """Test that an ONEX project structure passes validation."""
        # ONEX-specific files
        (tmp_path / "README.md").touch()
        (tmp_path / "CLAUDE.md").touch()
        (tmp_path / "pyproject.toml").touch()
        (tmp_path / ".env.example").touch()
        (tmp_path / ".pre-commit-config.yaml").touch()

        # ONEX directories
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "docs").mkdir()
        (tmp_path / "scripts").mkdir()
        (tmp_path / "contracts").mkdir()
        (tmp_path / ".claude").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_project_with_working_docs_fails(self, tmp_path: Path) -> None:
        """Test that a project with working documents in root fails."""
        # Valid files
        (tmp_path / "README.md").touch()
        (tmp_path / "pyproject.toml").touch()

        # Working documents that should be in docs/
        (tmp_path / "EXECUTION_PLAN.md").touch()
        (tmp_path / "MIGRATION_NOTES.md").touch()
        (tmp_path / "TODO.md").touch()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == 3

        violation_names = {v.path.name for v in result.violations}
        assert "EXECUTION_PLAN.md" in violation_names
        assert "MIGRATION_NOTES.md" in violation_names
        assert "TODO.md" in violation_names

    def test_project_with_build_artifacts_cached(self, tmp_path: Path) -> None:
        """Test that common build artifacts/caches pass (should be gitignored)."""
        # Valid structure
        (tmp_path / "README.md").touch()
        (tmp_path / "src").mkdir()

        # Build artifacts that should be gitignored but are allowed
        (tmp_path / "build").mkdir()
        (tmp_path / "dist").mkdir()
        (tmp_path / ".mypy_cache").mkdir()
        (tmp_path / ".pytest_cache").mkdir()
        (tmp_path / "__pycache__").mkdir()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0


class TestEnvFileEnforcement:
    """Test that .env files are correctly denied (zero-repo-env policy)."""

    def test_env_file_is_violation(self, tmp_path: Path) -> None:
        """Test that .env is flagged as a violation (zero-repo-env policy)."""
        (tmp_path / ".env").touch()

        result = validate_root_directory(tmp_path)

        assert not result.is_valid
        assert len(result.violations) == 1
        assert result.violations[0].path.name == ".env"

    def test_env_example_still_allowed(self, tmp_path: Path) -> None:
        """Test that .env.example passes (template files are allowed via .env.* pattern)."""
        (tmp_path / ".env.example").touch()

        result = validate_root_directory(tmp_path)

        assert result.is_valid
        assert len(result.violations) == 0

    def test_env_not_in_allowed_root_files(self) -> None:
        """Test that .env is not in ALLOWED_ROOT_FILES allowlist."""
        assert ".env" not in ALLOWED_ROOT_FILES

    def test_env_not_in_allowed_root_directories(self) -> None:
        """Test that .env is not in ALLOWED_ROOT_DIRECTORIES allowlist."""
        assert ".env" not in ALLOWED_ROOT_DIRECTORIES
