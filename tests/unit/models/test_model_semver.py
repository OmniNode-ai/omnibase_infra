# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelSemVer model using omnibase_core implementation.

Note: Tests for prerelease, build metadata, and from_tuple() have been removed
as the core ModelSemVer implementation only supports major.minor.patch versioning.

The core ModelSemVer has different behavior than the deprecated infra version:
- All fields (major, minor, patch) are required (no defaults)
- Invalid input raises ModelOnexError instead of ValueError
- Comparison with non-ModelSemVer types raises AttributeError
"""

from __future__ import annotations

import pytest
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.primitives.model_semver import ModelSemVer

# Create SEMVER_DEFAULT inline as the core module doesn't export it
SEMVER_DEFAULT = ModelSemVer(major=1, minor=0, patch=0)


class TestModelSemVerBasics:
    """Test basic ModelSemVer functionality."""

    def test_custom_values(self) -> None:
        """Test custom version values."""
        version = ModelSemVer(major=2, minor=3, patch=4)
        assert version.major == 2
        assert version.minor == 3
        assert version.patch == 4

    def test_fields_are_required(self) -> None:
        """Test that all version fields are required (no defaults)."""
        with pytest.raises(Exception):  # ValidationError from Pydantic
            ModelSemVer()  # type: ignore[call-arg]


class TestModelSemVerStringConversion:
    """Test ModelSemVer string conversion."""

    def test_basic_version_string(self) -> None:
        """Test basic version string output."""
        version = ModelSemVer(major=1, minor=2, patch=3)
        assert str(version) == "1.2.3"
        assert version.to_string() == "1.2.3"


class TestModelSemVerParse:
    """Test ModelSemVer.parse() method."""

    def test_parse_basic_version(self) -> None:
        """Test parsing basic semver string."""
        version = ModelSemVer.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_parse_invalid_version_raises_error(self) -> None:
        """Test parsing invalid version string raises ModelOnexError."""
        with pytest.raises(ModelOnexError):
            ModelSemVer.parse("1.0")  # Missing patch

    def test_parse_invalid_format_raises_error(self) -> None:
        """Test parsing invalid format raises ModelOnexError."""
        with pytest.raises(ModelOnexError):
            ModelSemVer.parse("not-a-version")

    def test_parse_non_string_raises_type_error(self) -> None:
        """Test parsing non-string input raises TypeError."""
        with pytest.raises(TypeError):
            ModelSemVer.parse(123)  # type: ignore[arg-type]

    def test_parse_none_raises_type_error(self) -> None:
        """Test parsing None raises TypeError."""
        with pytest.raises(TypeError):
            ModelSemVer.parse(None)  # type: ignore[arg-type]

    def test_parse_empty_string_raises_error(self) -> None:
        """Test parsing empty string raises ModelOnexError."""
        with pytest.raises(ModelOnexError):
            ModelSemVer.parse("")


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
        ],
    )
    def test_parse_and_stringify_roundtrip(self, version_str: str) -> None:
        """Test that parsing and stringifying produces same result."""
        version = ModelSemVer.parse(version_str)
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


class TestModelSemVerEquality:
    """Test ModelSemVer equality and hash operations."""

    def test_equal_versions(self) -> None:
        """Test equal versions compare as equal."""
        v1 = ModelSemVer(major=1, minor=2, patch=3)
        v2 = ModelSemVer(major=1, minor=2, patch=3)
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
