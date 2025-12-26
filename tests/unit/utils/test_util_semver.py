# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for util_semver module.

These tests verify that normalize_version() and normalize_version_cached()
raise TypeError for string version input, enforcing the use of ModelSemVer.
"""

from __future__ import annotations

import pytest

from omnibase_infra.utils.util_semver import (
    SEMVER_PATTERN,
    clear_normalize_version_cache,
    normalize_version,
    normalize_version_cached,
    validate_semver,
    validate_version_lenient,
)


class TestNormalizeVersionRaisesTypeError:
    """Tests that normalize_version() raises TypeError for string input."""

    def test_normalize_version_raises_type_error(self) -> None:
        """Test that normalize_version raises TypeError for any string input."""
        with pytest.raises(TypeError, match="String version input is not allowed"):
            normalize_version("1.0.0")

    def test_normalize_version_error_mentions_model_semver(self) -> None:
        """Test that the error message provides migration guidance."""
        with pytest.raises(TypeError) as exc_info:
            normalize_version("1.0.0")

        error_message = str(exc_info.value)
        assert "ModelSemVer" in error_message
        assert "parse()" in error_message

    def test_normalize_version_raises_for_all_formats(self) -> None:
        """Test that normalize_version raises TypeError for all version formats."""
        test_versions = [
            "1.0.0",
            "1.0",
            "1",
            "v1.2.3",
            "  2.0.0  ",
            "1.0.0-beta",
        ]
        for version in test_versions:
            with pytest.raises(TypeError, match="String version input is not allowed"):
                normalize_version(version)

    def test_normalize_version_internal_flag_still_raises(self) -> None:
        """Test that _emit_warning flag is ignored and still raises TypeError."""
        with pytest.raises(TypeError, match="String version input is not allowed"):
            normalize_version("1.0.0", _emit_warning=False)


class TestNormalizeVersionCachedRaisesTypeError:
    """Tests that normalize_version_cached() raises TypeError for string input."""

    def setup_method(self) -> None:
        """Clear the cache before each test."""
        clear_normalize_version_cache()

    def test_normalize_version_cached_raises_type_error(self) -> None:
        """Test that normalize_version_cached raises TypeError for any string input."""
        with pytest.raises(TypeError, match="String version input is not allowed"):
            normalize_version_cached("1.0.0")

    def test_normalize_version_cached_error_mentions_model_semver(self) -> None:
        """Test that the error message provides migration guidance."""
        with pytest.raises(TypeError) as exc_info:
            normalize_version_cached("1.0.0")

        error_message = str(exc_info.value)
        assert "ModelSemVer" in error_message
        assert "parse()" in error_message


class TestClearNormalizeVersionCache:
    """Tests for clear_normalize_version_cache function."""

    def test_clear_cache_is_noop(self) -> None:
        """Test that clear_normalize_version_cache is a no-op (cache no longer exists)."""
        # Should not raise any errors
        clear_normalize_version_cache()


class TestValidateSemver:
    """Tests for validate_semver function."""

    def test_valid_semver(self) -> None:
        """Test valid semver strings pass validation."""
        assert validate_semver("1.0.0") == "1.0.0"
        assert validate_semver("0.0.1") == "0.0.1"
        assert validate_semver("10.20.30") == "10.20.30"

    def test_valid_semver_with_prerelease(self) -> None:
        """Test valid semver with prerelease suffix."""
        assert validate_semver("1.0.0-alpha") == "1.0.0-alpha"
        assert validate_semver("1.0.0-beta.1") == "1.0.0-beta.1"

    def test_valid_semver_with_build_metadata(self) -> None:
        """Test valid semver with build metadata."""
        assert validate_semver("1.0.0+build123") == "1.0.0+build123"
        assert validate_semver("1.0.0-alpha+build") == "1.0.0-alpha+build"

    def test_invalid_semver_partial_version(self) -> None:
        """Test partial versions are rejected by strict semver."""
        with pytest.raises(ValueError, match="Invalid semantic version"):
            validate_semver("1.0")
        with pytest.raises(ValueError, match="Invalid semantic version"):
            validate_semver("1")

    def test_invalid_semver_format(self) -> None:
        """Test invalid formats are rejected."""
        with pytest.raises(ValueError, match="Invalid semantic version"):
            validate_semver("not-a-version")
        with pytest.raises(ValueError, match="Invalid semantic version"):
            validate_semver("v1.0.0")  # v prefix not allowed in strict semver


class TestValidateVersionLenient:
    """Tests for validate_version_lenient function."""

    def test_valid_full_version(self) -> None:
        """Test full version format."""
        assert validate_version_lenient("1.0.0") == "1.0.0"
        assert validate_version_lenient("10.20.30") == "10.20.30"

    def test_valid_partial_versions(self) -> None:
        """Test partial version formats are accepted."""
        assert validate_version_lenient("1") == "1"
        assert validate_version_lenient("1.0") == "1.0"

    def test_valid_version_with_prerelease(self) -> None:
        """Test versions with prerelease suffix."""
        assert validate_version_lenient("1.0.0-alpha") == "1.0.0-alpha"
        assert validate_version_lenient("2.1.0-beta") == "2.1.0-beta"

    def test_empty_version_rejected(self) -> None:
        """Test empty version strings are rejected."""
        with pytest.raises(ValueError, match="Version cannot be empty"):
            validate_version_lenient("")
        with pytest.raises(ValueError, match="Version cannot be empty"):
            validate_version_lenient("   ")

    def test_too_many_parts_rejected(self) -> None:
        """Test versions with too many parts are rejected."""
        with pytest.raises(ValueError, match="expected format"):
            validate_version_lenient("1.2.3.4")

    def test_empty_prerelease_rejected(self) -> None:
        """Test empty prerelease suffix is rejected."""
        with pytest.raises(ValueError, match="prerelease cannot be empty"):
            validate_version_lenient("1.0.0-")

    def test_non_numeric_component_rejected(self) -> None:
        """Test non-numeric version components are rejected."""
        with pytest.raises(ValueError, match="non-integer component"):
            validate_version_lenient("abc.0.0")

    def test_negative_component_rejected(self) -> None:
        """Test version starting with '-' is rejected as empty component."""
        # Note: "-1.0.0" splits as ["", "1", "0", "0"], so it's detected as empty component
        with pytest.raises(ValueError, match="empty component"):
            validate_version_lenient("-1.0.0")


class TestSemverPattern:
    """Tests for SEMVER_PATTERN regex."""

    def test_pattern_matches_valid_semver(self) -> None:
        """Test pattern matches valid semver strings."""
        assert SEMVER_PATTERN.match("1.0.0")
        assert SEMVER_PATTERN.match("0.0.1")
        assert SEMVER_PATTERN.match("1.0.0-alpha")
        assert SEMVER_PATTERN.match("1.0.0+build")
        assert SEMVER_PATTERN.match("1.0.0-alpha+build")

    def test_pattern_rejects_invalid_semver(self) -> None:
        """Test pattern rejects invalid semver strings."""
        assert not SEMVER_PATTERN.match("1.0")
        assert not SEMVER_PATTERN.match("1")
        assert not SEMVER_PATTERN.match("v1.0.0")
        assert not SEMVER_PATTERN.match("not-a-version")


class TestDocumentation:
    """Tests to verify module documentation reflects removal of string normalization."""

    def test_module_docstring_reflects_removal(self) -> None:
        """Test that module docstring documents that normalization was removed."""
        from omnibase_infra.utils import util_semver

        docstring = util_semver.__doc__
        assert docstring is not None
        assert "REMOVED" in docstring
        assert "ModelSemVer" in docstring

    def test_normalize_version_docstring_reflects_removal(self) -> None:
        """Test that normalize_version docstring documents removal."""
        docstring = normalize_version.__doc__
        assert docstring is not None
        assert "REMOVED" in docstring
        assert "TypeError" in docstring

    def test_normalize_version_cached_docstring_reflects_removal(self) -> None:
        """Test that normalize_version_cached docstring documents removal."""
        docstring = normalize_version_cached.__doc__
        assert docstring is not None
        assert "REMOVED" in docstring
        assert "TypeError" in docstring
