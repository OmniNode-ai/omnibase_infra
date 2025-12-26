# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for util_semver module, focusing on deprecation warnings.

These tests verify the FutureWarning deprecation warnings for string
version input to normalize_version() and normalize_version_cached().

The deprecation timeline is:
    - v1.1.0: FutureWarning added for string version input
    - v1.2.0: Warning upgraded to DeprecationWarning (visible by default)
    - v2.0.0: String input will raise TypeError; require ModelSemVer instance
"""

from __future__ import annotations

import warnings

import pytest

from omnibase_infra.utils.util_semver import (
    clear_normalize_version_cache,
    normalize_version,
    normalize_version_cached,
)


class TestNormalizeVersionDeprecationWarning:
    """Tests for normalize_version() deprecation warning."""

    def test_normalize_version_emits_future_warning(self) -> None:
        """Test that normalize_version emits FutureWarning for string input."""
        with pytest.warns(
            FutureWarning,
            match=r"Passing string version to normalize_version\(\) is deprecated",
        ):
            result = normalize_version("1.0.0")
        assert result == "1.0.0"

    def test_normalize_version_warning_mentions_model_semver(self) -> None:
        """Test that the warning message provides migration guidance."""
        with pytest.warns(FutureWarning) as record:
            normalize_version("1.0.0")

        assert len(record) == 1
        warning_message = str(record[0].message)
        assert "ModelSemVer" in warning_message
        assert "v2.0.0" in warning_message

    def test_normalize_version_still_works_correctly(self) -> None:
        """Test that normalize_version still functions despite deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)

            # Test various version formats
            assert normalize_version("1.0.0") == "1.0.0"
            assert normalize_version("1.0") == "1.0.0"
            assert normalize_version("1") == "1.0.0"
            assert normalize_version("v1.2.3") == "1.2.3"
            assert normalize_version("  2.0.0  ") == "2.0.0"
            assert normalize_version("1.0.0-beta") == "1.0.0-beta"

    def test_normalize_version_internal_flag_suppresses_warning(self) -> None:
        """Test that _emit_warning=False suppresses the warning.

        This internal flag is used by normalize_version_cached to
        prevent double-warning.
        """
        # Should not emit any FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # This should not raise because warning is suppressed
            result = normalize_version("1.0.0", _emit_warning=False)
            assert result == "1.0.0"


class TestNormalizeVersionCachedDeprecationWarning:
    """Tests for normalize_version_cached() deprecation warning."""

    def setup_method(self) -> None:
        """Clear the cache before each test."""
        clear_normalize_version_cache()

    def test_normalize_version_cached_emits_warning_on_cache_miss(self) -> None:
        """Test that normalize_version_cached emits FutureWarning on cache miss."""
        with pytest.warns(
            FutureWarning,
            match=r"Passing string version to normalize_version_cached\(\) is deprecated",
        ):
            result = normalize_version_cached("1.0.0")
        assert result == "1.0.0"

    def test_normalize_version_cached_no_warning_on_cache_hit(self) -> None:
        """Test that normalize_version_cached does NOT emit warning on cache hit."""
        # First call - will emit warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            normalize_version_cached("3.0.0")

        # Second call - should NOT emit warning (cache hit)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # This should not raise because it's a cache hit
            result = normalize_version_cached("3.0.0")
            assert result == "3.0.0"

    def test_normalize_version_cached_warning_per_unique_version(self) -> None:
        """Test that each unique version string triggers one warning."""
        warnings_captured: list[str] = []

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", FutureWarning)

            # First version - should warn
            normalize_version_cached("1.0.0")
            # Same version again - should NOT warn
            normalize_version_cached("1.0.0")
            # Different version - should warn
            normalize_version_cached("2.0.0")
            # First version again - should NOT warn (still in cache)
            normalize_version_cached("1.0.0")

            warnings_captured = [
                str(warning.message)
                for warning in w
                if issubclass(warning.category, FutureWarning)
            ]

        # Only 2 warnings: one for "1.0.0" and one for "2.0.0"
        assert len(warnings_captured) == 2

    def test_normalize_version_cached_warning_mentions_migration(self) -> None:
        """Test that the warning provides helpful migration guidance."""
        # Use a unique version to ensure cache miss
        with pytest.warns(FutureWarning) as record:
            normalize_version_cached("4.5.6")  # Unique version for this test

        assert len(record) == 1
        warning_message = str(record[0].message)
        assert "ModelSemVer" in warning_message
        assert "v2.0.0" in warning_message
        assert "major=X, minor=Y, patch=Z" in warning_message

    def test_normalize_version_cached_still_works_correctly(self) -> None:
        """Test that normalize_version_cached still functions despite deprecation."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)

            # Test various version formats
            assert normalize_version_cached("1.0.0") == "1.0.0"
            assert normalize_version_cached("1.0") == "1.0.0"
            assert normalize_version_cached("1") == "1.0.0"
            assert normalize_version_cached("v1.2.3") == "1.2.3"


class TestDeprecationTimelineDocumentation:
    """Tests to verify deprecation timeline is properly documented."""

    def test_module_docstring_contains_deprecation_timeline(self) -> None:
        """Test that module docstring documents the deprecation timeline."""
        from omnibase_infra.utils import util_semver

        docstring = util_semver.__doc__
        assert docstring is not None
        assert "deprecated" in docstring.lower()
        assert "v1.1.0" in docstring
        assert "v1.2.0" in docstring
        assert "v2.0.0" in docstring

    def test_normalize_version_docstring_contains_deprecation(self) -> None:
        """Test that normalize_version docstring documents deprecation."""
        docstring = normalize_version.__doc__
        assert docstring is not None
        assert "deprecated" in docstring.lower()
        assert "Migration Guide" in docstring or "migration" in docstring.lower()

    def test_normalize_version_cached_docstring_contains_deprecation(self) -> None:
        """Test that normalize_version_cached docstring documents deprecation."""
        docstring = normalize_version_cached.__doc__
        assert docstring is not None
        assert "deprecated" in docstring.lower()
