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

    def test_parse_non_string_raises_type_error(self) -> None:
        """Test parsing non-string input raises TypeError."""
        with pytest.raises(TypeError, match="version must be a string"):
            ModelSemVer.from_string(123)  # type: ignore[arg-type]

    def test_parse_none_raises_type_error(self) -> None:
        """Test parsing None raises TypeError."""
        with pytest.raises(TypeError, match="version must be a string"):
            ModelSemVer.from_string(None)  # type: ignore[arg-type]

    def test_parse_empty_string_raises_value_error(self) -> None:
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelSemVer.from_string("")

    def test_parse_whitespace_only_raises_value_error(self) -> None:
        """Test parsing whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelSemVer.from_string("   ")


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

    def test_comparison_with_non_semver_raises_type_error(self) -> None:
        """Test < operator with non-ModelSemVer raises TypeError."""
        version = ModelSemVer(major=1, minor=0, patch=0)
        with pytest.raises(TypeError):
            _ = version < "1.0.0"  # type: ignore[operator]

    def test_eq_with_non_semver_returns_not_implemented(self) -> None:
        """Test __eq__ with non-ModelSemVer returns NotImplemented."""
        version = ModelSemVer(major=1, minor=0, patch=0)
        result = version.__eq__("1.0.0")
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


class TestModelSemVerFromTuple:
    """Test ModelSemVer.from_tuple() parsing."""

    def test_basic_tuple(self) -> None:
        """Test creating version from basic tuple."""
        version = ModelSemVer.from_tuple((1, 2, 3))
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build is None

    def test_tuple_with_prerelease(self) -> None:
        """Test creating version from tuple with prerelease."""
        version = ModelSemVer.from_tuple((1, 0, 0), prerelease="alpha")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha"

    def test_tuple_with_build(self) -> None:
        """Test creating version from tuple with build metadata."""
        version = ModelSemVer.from_tuple((2, 1, 0), build="build123")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0
        assert version.build == "build123"

    def test_tuple_with_prerelease_and_build(self) -> None:
        """Test creating version from tuple with both prerelease and build."""
        version = ModelSemVer.from_tuple((1, 0, 0), prerelease="beta", build="abc123")
        assert version.prerelease == "beta"
        assert version.build == "abc123"

    def test_zero_tuple_valid(self) -> None:
        """Test zero version tuple (0, 0, 0) is valid."""
        version = ModelSemVer.from_tuple((0, 0, 0))
        assert str(version) == "0.0.0"

    def test_wrong_length_tuple_raises_error(self) -> None:
        """Test tuple with wrong number of elements raises ValueError."""
        with pytest.raises(ValueError, match="exactly 3 elements"):
            ModelSemVer.from_tuple((1, 2))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="exactly 3 elements"):
            ModelSemVer.from_tuple((1, 2, 3, 4))  # type: ignore[arg-type]

    def test_non_tuple_raises_type_error(self) -> None:
        """Test non-tuple input raises TypeError."""
        with pytest.raises(TypeError, match="Expected tuple"):
            ModelSemVer.from_tuple([1, 2, 3])  # type: ignore[arg-type]

    def test_non_integer_element_raises_type_error(self) -> None:
        """Test non-integer elements raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ModelSemVer.from_tuple((1, "2", 3))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be an integer"):
            ModelSemVer.from_tuple((1.5, 2, 3))  # type: ignore[arg-type]

    def test_negative_element_raises_value_error(self) -> None:
        """Test negative elements raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ModelSemVer.from_tuple((-1, 0, 0))

        with pytest.raises(ValueError, match="must be non-negative"):
            ModelSemVer.from_tuple((1, -1, 0))

        with pytest.raises(ValueError, match="must be non-negative"):
            ModelSemVer.from_tuple((1, 0, -1))


class TestModelSemVerTotalOrdering:
    """Test @total_ordering generated comparison operators."""

    def test_less_than_or_equal(self) -> None:
        """Test __le__ operator (generated by @total_ordering)."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=2, minor=0, patch=0)
        v3 = ModelSemVer(major=1, minor=0, patch=0)

        assert v1 <= v2
        assert v1 <= v3
        assert not (v2 <= v1)

    def test_greater_than(self) -> None:
        """Test __gt__ operator (generated by @total_ordering)."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=2, minor=0, patch=0)

        assert v2 > v1
        assert not (v1 > v2)

    def test_greater_than_or_equal(self) -> None:
        """Test __ge__ operator (generated by @total_ordering)."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=2, minor=0, patch=0)
        v3 = ModelSemVer(major=2, minor=0, patch=0)

        assert v2 >= v1
        assert v2 >= v3
        assert not (v1 >= v2)

    def test_sorting_with_total_ordering(self) -> None:
        """Test that versions sort correctly using total_ordering operators."""
        versions = [
            ModelSemVer(major=2, minor=0, patch=0),
            ModelSemVer(major=1, minor=1, patch=0),
            ModelSemVer(major=1, minor=0, patch=0),
            ModelSemVer(major=1, minor=0, patch=1),
        ]
        sorted_versions = sorted(versions)
        expected = [
            ModelSemVer(major=1, minor=0, patch=0),
            ModelSemVer(major=1, minor=0, patch=1),
            ModelSemVer(major=1, minor=1, patch=0),
            ModelSemVer(major=2, minor=0, patch=0),
        ]
        assert sorted_versions == expected

    def test_prerelease_sorting(self) -> None:
        """Test sorting with prerelease versions."""
        versions = [
            ModelSemVer(major=1, minor=0, patch=0),  # Release
            ModelSemVer(major=1, minor=0, patch=0, prerelease="beta"),
            ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha"),
        ]
        sorted_versions = sorted(versions)
        # alpha < beta < release
        assert sorted_versions[0].prerelease == "alpha"
        assert sorted_versions[1].prerelease == "beta"
        assert sorted_versions[2].prerelease is None


class TestModelSemVerEquality:
    """Test ModelSemVer equality and hash operations."""

    def test_equal_versions(self) -> None:
        """Test equal versions compare as equal."""
        v1 = ModelSemVer(major=1, minor=2, patch=3)
        v2 = ModelSemVer(major=1, minor=2, patch=3)
        assert v1 == v2

    def test_equal_versions_with_prerelease(self) -> None:
        """Test equal versions with prerelease compare as equal."""
        v1 = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        v2 = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        assert v1 == v2

    def test_different_prerelease_not_equal(self) -> None:
        """Test versions with different prerelease are not equal."""
        v1 = ModelSemVer(major=1, minor=0, patch=0, prerelease="alpha")
        v2 = ModelSemVer(major=1, minor=0, patch=0, prerelease="beta")
        assert v1 != v2

    def test_build_metadata_ignored_in_equality(self) -> None:
        """Test build metadata is ignored in equality per semver spec."""
        v1 = ModelSemVer(major=1, minor=0, patch=0, build="build1")
        v2 = ModelSemVer(major=1, minor=0, patch=0, build="build2")
        assert v1 == v2

    def test_hash_equal_versions(self) -> None:
        """Test equal versions have same hash."""
        v1 = ModelSemVer(major=1, minor=2, patch=3)
        v2 = ModelSemVer(major=1, minor=2, patch=3)
        assert hash(v1) == hash(v2)

    def test_hash_different_versions(self) -> None:
        """Test different versions have different hash (usually)."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=2, minor=0, patch=0)
        assert hash(v1) != hash(v2)

    def test_usable_in_set(self) -> None:
        """Test ModelSemVer can be used in sets."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=1, minor=0, patch=0)
        v3 = ModelSemVer(major=2, minor=0, patch=0)

        version_set = {v1, v2, v3}
        assert len(version_set) == 2  # v1 and v2 are equal

    def test_usable_as_dict_key(self) -> None:
        """Test ModelSemVer can be used as dict key."""
        v1 = ModelSemVer(major=1, minor=0, patch=0)
        v2 = ModelSemVer(major=1, minor=0, patch=0)

        version_dict = {v1: "first"}
        version_dict[v2] = "second"

        assert len(version_dict) == 1
        assert version_dict[v1] == "second"

    def test_equality_with_non_semver_returns_false(self) -> None:
        """Test equality with non-ModelSemVer returns False via NotImplemented."""
        version = ModelSemVer(major=1, minor=0, patch=0)
        assert version != "1.0.0"
        assert version != 100
        assert version != (1, 0, 0)
