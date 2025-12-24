# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelSemVer model."""

from __future__ import annotations

import pytest

from omnibase_infra.models.model_semver import SEMVER_DEFAULT, ModelSemVer


class TestModelSemVerBasics:
    """Test basic ModelSemVer functionality."""

    def test_default_values(self) -> None:
        """Test default version values."""
        version = ModelSemVer()
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease is None
        assert version.build is None

    def test_custom_values(self) -> None:
        """Test custom version values."""
        version = ModelSemVer(major=2, minor=3, patch=4)
        assert version.major == 2
        assert version.minor == 3
        assert version.patch == 4

    def test_with_prerelease(self) -> None:
        """Test version with prerelease identifier."""
        version = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        assert version.prerelease == "alpha"

    def test_with_build(self) -> None:
        """Test version with build metadata."""
        version = ModelSemVer(major=1, minor=0, patch=0, build="20231215")
        assert version.build == "20231215"

    def test_with_prerelease_and_build(self) -> None:
        """Test version with both prerelease and build metadata."""
        version = ModelSemVer(
            major=1, minor=0, patch=0, prerelease="beta.1", build="abc123"
        )
        assert version.prerelease == "beta.1"
        assert version.build == "abc123"


class TestModelSemVerStringConversion:
    """Test ModelSemVer string conversion."""

    def test_basic_version_string(self) -> None:
        """Test basic version string output."""
        version = ModelSemVer(major=1, minor=2, patch=3)
        assert str(version) == "1.2.3"

    def test_version_with_prerelease_string(self) -> None:
        """Test version with prerelease string output."""
        version = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        assert str(version) == "1.0.0-alpha"

    def test_version_with_build_string(self) -> None:
        """Test version with build metadata string output."""
        version = ModelSemVer(major=1, minor=0, patch=0, build="build123")
        assert str(version) == "1.0.0+build123"

    def test_version_with_prerelease_and_build_string(self) -> None:
        """Test version with both prerelease and build string output."""
        version = ModelSemVer(
            major=2, minor=1, patch=0, prerelease="beta", build="sha.45678"
        )
        assert str(version) == "2.1.0-beta+sha.45678"


class TestModelSemVerFromString:
    """Test ModelSemVer.from_string() parsing."""

    def test_parse_basic_version(self) -> None:
        """Test parsing basic semver string."""
        version = ModelSemVer.from_string("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build is None

    def test_parse_version_with_prerelease(self) -> None:
        """Test parsing semver string with prerelease."""
        version = ModelSemVer.from_string("1.0.0-alpha")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha"
        assert version.build is None

    def test_parse_version_with_build(self) -> None:
        """Test parsing semver string with build metadata."""
        version = ModelSemVer.from_string("1.0.0+build123")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease is None
        assert version.build == "build123"

    def test_parse_version_with_prerelease_and_build(self) -> None:
        """Test parsing semver string with prerelease and build."""
        version = ModelSemVer.from_string("2.1.0-beta.1+sha.45678")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0
        assert version.prerelease == "beta.1"
        assert version.build == "sha.45678"

    def test_parse_invalid_version_raises_error(self) -> None:
        """Test parsing invalid version string raises ValueError."""
        with pytest.raises(ValueError):
            ModelSemVer.from_string("1.0")  # Missing patch

    def test_parse_invalid_format_raises_error(self) -> None:
        """Test parsing invalid format raises ValueError."""
        with pytest.raises(ValueError):
            ModelSemVer.from_string("not-a-version")


class TestModelSemVerComparison:
    """Test ModelSemVer comparison operations."""

    def test_equal_versions(self) -> None:
        """Test equal version comparison."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=1, minor=0, patch=0)
        assert not (v1 < v2)
        assert not (v2 < v1)

    def test_major_version_comparison(self) -> None:
        """Test major version comparison."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=2, minor=0, patch=0)
        assert v1 < v2
        assert not (v2 < v1)

    def test_minor_version_comparison(self) -> None:
        """Test minor version comparison."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=1, minor=1, patch=0)
        assert v1 < v2
        assert not (v2 < v1)

    def test_patch_version_comparison(self) -> None:
        """Test patch version comparison."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=1, minor=0, patch=1)
        assert v1 < v2
        assert not (v2 < v1)

    def test_prerelease_less_than_release(self) -> None:
        """Test prerelease version is less than release version."""
        prerelease = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        release = ModelSemVer(major=1, minor=0, patch=0)
        assert prerelease < release
        assert not (release < prerelease)

    def test_prerelease_comparison(self) -> None:
        """Test prerelease version comparison."""
        alpha = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        beta = ModelSemVer(major=1, minor=0, patch=0, prerelease="beta")
        assert alpha < beta  # Alphabetical comparison

    def test_comparison_with_non_semver_returns_not_implemented(self) -> None:
        """Test comparison with non-ModelSemVer returns NotImplemented."""
        version = ModelSemVer(major=1, minor=0, patch=0)
        result = version.__lt__("1.0.0")
        assert result is NotImplemented


class TestSemVerDefault:
    """Test SEMVER_DEFAULT constant."""

    def test_default_version_value(self) -> None:
        """Test SEMVER_DEFAULT has expected values."""
        assert SEMVER_DEFAULT.major == 1
        assert SEMVER_DEFAULT.minor == 0
        assert SEMVER_DEFAULT.patch == 0
        assert SEMVER_DEFAULT.prerelease is None
        assert SEMVER_DEFAULT.build is None

    def test_default_version_string(self) -> None:
        """Test SEMVER_DEFAULT string representation."""
        assert str(SEMVER_DEFAULT) == "1.0.0"


class TestModelSemVerRoundtrip:
    """Test ModelSemVer serialization roundtrip."""

    @pytest.mark.parametrize(
        "version_str",
        [
            "0.0.1",
            "1.0.0",
            "1.2.3",
            "10.20.30",
            "1.0.0-alpha",
            "1.0.0-beta.1",
            "1.0.0+build123",
            "1.0.0-alpha+build123",
            "2.1.0-rc.1+sha.45678",
        ],
    )
    def test_parse_and_stringify_roundtrip(self, version_str: str) -> None:
        """Test that parsing and stringifying produces same result."""
        version = ModelSemVer.from_string(version_str)
        assert str(version) == version_str


class TestModelSemVerValidation:
    """Test ModelSemVer field validation."""

    def test_negative_major_rejected(self) -> None:
        """Test negative major version is rejected."""
        with pytest.raises(ValueError):
            ModelSemVer(major=-1, minor=0, patch=0)

    def test_negative_minor_rejected(self) -> None:
        """Test negative minor version is rejected."""
        with pytest.raises(ValueError):
            ModelSemVer(major=1, minor=-1, patch=0)

    def test_negative_patch_rejected(self) -> None:
        """Test negative patch version is rejected."""
        with pytest.raises(ValueError):
            ModelSemVer(major=1, minor=0, patch=-1)

    def test_zero_version_valid(self) -> None:
        """Test zero version (0.0.0) is valid."""
        version = ModelSemVer(major=0, minor=0, patch=0)
        assert str(version) == "0.0.0"
