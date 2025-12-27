# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for path skipping logic in infra_validators.

These tests verify that the path skipping functions correctly identify
directories to skip (archive, archived, examples, __pycache__) while
avoiding false positives from similar-looking paths.

The key requirements are:
1. Exact directory component matching (not substring matching)
2. Only check parent directories (not filenames)
3. Case-sensitive matching
4. Configurable via YAML (with hardcoded fallback)
"""

from pathlib import Path

import pytest

from omnibase_infra.validation.infra_validators import (
    SKIP_DIRECTORY_NAMES,
    get_skip_directories,
    is_skip_directory,
    load_skip_directories_from_yaml,
    should_skip_path,
)


class TestSkipDirectoryNames:
    """Tests for the SKIP_DIRECTORY_NAMES constant."""

    def test_skip_directory_names_is_frozenset(self) -> None:
        """Verify SKIP_DIRECTORY_NAMES is a frozenset for O(1) lookup."""
        assert isinstance(SKIP_DIRECTORY_NAMES, frozenset)

    def test_skip_directory_names_contains_expected_values(self) -> None:
        """Verify all expected directory names are in the set."""
        expected = {
            # Historical/demo code
            "archive",
            "archived",
            "examples",
            # Bytecode and caches
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            # Virtual environments
            ".venv",
            "venv",
            # Build outputs
            "build",
            "dist",
            ".eggs",
            # Version control
            ".git",
            # Testing
            ".tox",
            # Node.js (if present)
            "node_modules",
        }
        assert expected == SKIP_DIRECTORY_NAMES

    def test_skip_directory_names_is_immutable(self) -> None:
        """Verify the set cannot be modified."""
        with pytest.raises(AttributeError):
            SKIP_DIRECTORY_NAMES.add("new_dir")  # type: ignore[attr-defined]


class TestIsSkipDirectory:
    """Tests for the is_skip_directory function."""

    @pytest.mark.parametrize(
        "component",
        [
            "archive",
            "archived",
            "examples",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".venv",
            "venv",
            "build",
            "dist",
            ".eggs",
            ".git",
            ".tox",
            "node_modules",
        ],
    )
    def test_exact_match_returns_true(self, component: str) -> None:
        """Verify exact matches return True."""
        assert is_skip_directory(component) is True

    @pytest.mark.parametrize(
        "component",
        [
            # Prefix matches should NOT match
            "archived_feature",
            "archive_old",
            "examples_utils",
            "__pycache___extra",
            # Suffix matches should NOT match
            "my_archive",
            "old_archived",
            "code_examples",
            "test__pycache__",
            # Substring matches should NOT match
            "my_archived_code",
            "contains_archive_here",
            "has_examples_inside",
            "some__pycache__dir",
            # Different case should NOT match (case-sensitive)
            "Archive",
            "ARCHIVED",
            "Examples",
            "EXAMPLES",
            "__PYCACHE__",
            # Common non-skip directories
            "src",
            "tests",
            "models",
            "utils",
            "",
        ],
    )
    def test_non_exact_match_returns_false(self, component: str) -> None:
        """Verify non-exact matches return False (no false positives)."""
        assert is_skip_directory(component) is False


class TestShouldSkipPath:
    """Tests for the should_skip_path function."""

    # Paths that SHOULD be skipped (true positives)
    @pytest.mark.parametrize(
        "path",
        [
            # archive directory
            Path("src/archive/foo.py"),
            Path("src/omnibase_infra/archive/bar.py"),
            Path("/workspace/src/archive/deep/nested/file.py"),
            # archived directory
            Path("src/archived/foo.py"),
            Path("src/omnibase_infra/archived/bar.py"),
            # examples directory
            Path("src/examples/foo.py"),
            Path("docs/examples/sample.py"),
            # __pycache__ directory
            Path("src/__pycache__/foo.pyc"),
            Path("src/models/__pycache__/bar.cpython-311.pyc"),
            # Multiple skip directories in path
            Path("src/archive/examples/foo.py"),
            Path("src/archived/__pycache__/bar.pyc"),
        ],
    )
    def test_skip_directory_in_path_returns_true(self, path: Path) -> None:
        """Verify paths containing skip directories are skipped."""
        assert should_skip_path(path) is True

    # Paths that should NOT be skipped (avoiding false positives)
    @pytest.mark.parametrize(
        ("path", "description"),
        [
            # Files with names similar to skip directories
            (Path("src/archive.py"), "file named 'archive.py'"),
            (Path("src/archived.py"), "file named 'archived.py'"),
            (Path("src/examples.py"), "file named 'examples.py'"),
            (Path("src/__pycache__.py"), "file named '__pycache__.py'"),
            # Directories with similar prefixes (not exact match)
            (Path("src/archived_feature/foo.py"), "prefix 'archived_feature'"),
            (Path("src/archive_old/bar.py"), "prefix 'archive_old'"),
            (Path("src/examples_utils/baz.py"), "prefix 'examples_utils'"),
            # Directories with similar suffixes (not exact match)
            (Path("src/my_archive/foo.py"), "suffix 'my_archive'"),
            (Path("src/old_archived/bar.py"), "suffix 'old_archived'"),
            (Path("src/code_examples/baz.py"), "suffix 'code_examples'"),
            # Directories with skip name as substring (not component)
            (Path("src/my_archived_code/foo.py"), "substring 'archived'"),
            (Path("src/contains_archive_here/bar.py"), "substring 'archive'"),
            (Path("src/has_examples_inside/baz.py"), "substring 'examples'"),
            (Path("src/some__pycache__dir/qux.py"), "substring '__pycache__'"),
            # Different case (case-sensitive check)
            (Path("src/Archive/foo.py"), "different case 'Archive'"),
            (Path("src/ARCHIVED/bar.py"), "different case 'ARCHIVED'"),
            (Path("src/Examples/baz.py"), "different case 'Examples'"),
            (Path("src/__PYCACHE__/qux.py"), "different case '__PYCACHE__'"),
            # Normal directories that happen to contain skip words
            (Path("src/omnibase_infra/models/foo.py"), "normal path"),
            (Path("tests/unit/validation/bar.py"), "test path"),
        ],
    )
    def test_similar_path_not_skipped(self, path: Path, description: str) -> None:
        """Verify similar-looking paths are NOT skipped (no false positives)."""
        assert should_skip_path(path) is False, f"False positive for: {description}"

    def test_root_path_not_skipped(self) -> None:
        """Verify files at root level are not skipped."""
        assert should_skip_path(Path("foo.py")) is False
        assert should_skip_path(Path("archive.py")) is False

    def test_empty_parent_path(self) -> None:
        """Verify files with no parent directory work correctly."""
        # Path("file.py").parent is Path(".") which has parts (".",)
        assert should_skip_path(Path("file.py")) is False

    def test_absolute_vs_relative_paths(self) -> None:
        """Verify both absolute and relative paths work correctly."""
        # Relative path with skip directory
        assert should_skip_path(Path("src/archive/foo.py")) is True
        # Absolute path with skip directory
        assert should_skip_path(Path("/workspace/src/archive/foo.py")) is True
        # Relative path without skip directory
        assert should_skip_path(Path("src/models/foo.py")) is False
        # Absolute path without skip directory
        assert should_skip_path(Path("/workspace/src/models/foo.py")) is False

    def test_deeply_nested_skip_directory(self) -> None:
        """Verify skip directories work at any depth in the path."""
        # Skip directory early in path
        assert should_skip_path(Path("archive/a/b/c/d/e/foo.py")) is True
        # Skip directory deep in path
        assert should_skip_path(Path("a/b/c/d/e/archive/foo.py")) is True
        # Skip directory in middle of path
        assert should_skip_path(Path("a/b/archive/c/d/foo.py")) is True


class TestPathSkippingIntegration:
    """Integration tests for path skipping with real filesystem-like patterns."""

    def test_typical_archive_structure(self) -> None:
        """Verify typical archive directory structure is skipped."""
        archive_files = [
            Path("src/omnibase_infra/archive/old_code.py"),
            Path("src/omnibase_infra/archive/deprecated/legacy.py"),
            Path("src/omnibase_infra/archived/v1_implementation.py"),
        ]
        for path in archive_files:
            assert should_skip_path(path) is True, f"Should skip: {path}"

    def test_typical_examples_structure(self) -> None:
        """Verify typical examples directory structure is skipped."""
        example_files = [
            Path("docs/examples/quickstart.py"),
            Path("src/examples/demo_script.py"),
            Path("examples/advanced/complex_example.py"),
        ]
        for path in example_files:
            assert should_skip_path(path) is True, f"Should skip: {path}"

    def test_typical_pycache_structure(self) -> None:
        """Verify typical __pycache__ directory structure is skipped."""
        cache_files = [
            Path("src/__pycache__/module.cpython-311.pyc"),
            Path("src/models/__pycache__/model_foo.cpython-310.pyc"),
            Path("tests/__pycache__/conftest.cpython-311.pyc"),
        ]
        for path in cache_files:
            assert should_skip_path(path) is True, f"Should skip: {path}"

    def test_normal_source_files_not_skipped(self) -> None:
        """Verify normal source files are not skipped."""
        source_files = [
            Path("src/omnibase_infra/validation/infra_validators.py"),
            Path("src/omnibase_infra/models/model_foo.py"),
            Path("src/omnibase_infra/event_bus/kafka_event_bus.py"),
            Path("tests/unit/validation/test_path_skipping.py"),
        ]
        for path in source_files:
            assert should_skip_path(path) is False, f"Should NOT skip: {path}"


class TestGetSkipDirectories:
    """Tests for the get_skip_directories function."""

    def test_get_skip_directories_returns_frozenset(self) -> None:
        """Verify get_skip_directories returns a frozenset."""
        result = get_skip_directories()
        assert isinstance(result, frozenset)

    def test_get_skip_directories_contains_expected_values(self) -> None:
        """Verify get_skip_directories includes expected directory names."""
        result = get_skip_directories()
        # These are the minimum expected directories from both YAML and fallback
        expected_minimum = {
            "archive",
            "archived",
            "examples",
            "__pycache__",
            ".venv",
            "venv",
            ".git",
        }
        assert expected_minimum.issubset(result)

    def test_get_skip_directories_matches_constant(self) -> None:
        """Verify YAML config matches hardcoded constant (no drift)."""
        # The YAML configuration should produce the same set as the constant
        # This test catches configuration drift
        result = get_skip_directories()
        assert result == SKIP_DIRECTORY_NAMES, (
            f"YAML config and SKIP_DIRECTORY_NAMES constant have drifted. "
            f"Missing from YAML: {SKIP_DIRECTORY_NAMES - result}. "
            f"Extra in YAML: {result - SKIP_DIRECTORY_NAMES}."
        )


class TestLoadSkipDirectoriesFromYaml:
    """Tests for the load_skip_directories_from_yaml function."""

    def test_yaml_loader_returns_frozenset_or_none(self) -> None:
        """Verify YAML loader returns frozenset or None."""
        # Clear the cache to ensure fresh load
        load_skip_directories_from_yaml.cache_clear()
        result = load_skip_directories_from_yaml()
        assert result is None or isinstance(result, frozenset)

    def test_yaml_loader_is_cached(self) -> None:
        """Verify YAML loader caches results for performance."""
        # Clear cache and load twice
        load_skip_directories_from_yaml.cache_clear()
        result1 = load_skip_directories_from_yaml()
        result2 = load_skip_directories_from_yaml()
        # Both calls should return the same object (cached)
        if result1 is not None:
            assert result1 is result2


class TestPathSkippingConfigurationIntegrity:
    """Tests to ensure configuration consistency."""

    def test_no_empty_directory_names(self) -> None:
        """Verify no empty strings in skip directories."""
        result = get_skip_directories()
        assert "" not in result, "Empty string found in skip directories"
        assert all(d.strip() == d for d in result), "Whitespace-padded entries found"

    def test_all_directories_are_strings(self) -> None:
        """Verify all skip directory entries are strings."""
        result = get_skip_directories()
        assert all(isinstance(d, str) for d in result)

    def test_no_path_separators_in_directory_names(self) -> None:
        """Verify skip directories are single components (no path separators)."""
        result = get_skip_directories()
        for d in result:
            assert "/" not in d, f"Path separator found in '{d}'"
            assert "\\" not in d, f"Backslash found in '{d}'"
