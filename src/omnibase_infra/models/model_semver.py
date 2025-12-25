# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic version model for ONEX infrastructure.

.. deprecated::
    This module is DEPRECATED. Use ``omnibase_core.models.primitives.model_semver``
    instead for all new code.

    Migration guide:
        - Replace ``from omnibase_infra.models.model_semver import ModelSemVer``
          with ``from omnibase_core.models.primitives.model_semver import ModelSemVer``
        - Replace ``ModelSemVer.from_string(...)`` with ``ModelSemVer.parse(...)``
        - Replace ``str(semver)`` with ``semver.to_string()`` (both work, but to_string() is preferred)
        - Create SEMVER_DEFAULT inline: ``SEMVER_DEFAULT = ModelSemVer.parse("1.0.0")``

    This file is kept for backwards compatibility with existing code and tests.
    It will be removed in a future version.

Provides a strongly-typed Pydantic model for semantic versioning,
replacing raw string usage with validated, structured version data.
"""

from __future__ import annotations

import warnings
from functools import total_ordering

from pydantic import BaseModel, ConfigDict, Field

# Issue deprecation warning at import time
warnings.warn(
    "omnibase_infra.models.model_semver is deprecated. "
    "Use omnibase_core.models.primitives.model_semver instead. "
    "Replace from_string() with parse() and str() with to_string().",
    DeprecationWarning,
    stacklevel=2,
)

from omnibase_infra.utils.util_semver import validate_semver


def _validate_version_component(value: int, name: str) -> None:
    """Validate a semantic version component.

    Args:
        value: The version component value to validate.
        name: The name of the component (for error messages).

    Raises:
        TypeError: If value is not an integer.
        ValueError: If value is negative.
    """
    if not isinstance(value, int):
        raise TypeError(
            f"Version component '{name}' must be an integer, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(
            f"Version component '{name}' must be non-negative, got {value}"
        )


@total_ordering
class ModelSemVer(BaseModel):
    """Semantic version model following semver.org specification.

    Provides structured representation of semantic versions with validation.
    Supports serialization to string and full comparison operations via
    @total_ordering decorator (__eq__, __lt__, __le__, __gt__, __ge__).

    Attributes:
        major: Major version number (breaking changes)
        minor: Minor version number (backwards-compatible features)
        patch: Patch version number (backwards-compatible bug fixes)
        prerelease: Optional prerelease identifier (e.g., "alpha", "beta.1")
        build: Optional build metadata (e.g., "20231215", "abc123")

    Example:
        >>> version = ModelSemVer(major=1, minor=2, patch=3)
        >>> str(version)
        '1.2.3'
        >>> version = ModelSemVer.from_string("1.0.0-beta+build123")
        >>> version.prerelease
        'beta'
        >>> v1 = ModelSemVer(major=1, minor=0, patch=0)
        >>> v2 = ModelSemVer(major=2, minor=0, patch=0)
        >>> v1 < v2
        True
        >>> v1 <= v2
        True
        >>> v2 > v1
        True
    """

    model_config = ConfigDict(
        from_attributes=True,  # pytest-xdist compatibility
    )

    major: int = Field(default=1, ge=0, description="Major version number")
    minor: int = Field(default=0, ge=0, description="Minor version number")
    patch: int = Field(default=0, ge=0, description="Patch version number")
    prerelease: str | None = Field(default=None, description="Prerelease identifier")
    build: str | None = Field(default=None, description="Build metadata")

    @classmethod
    def from_string(cls, version: str) -> ModelSemVer:
        """Parse a semantic version string into ModelSemVer.

        Parses a string following the semantic versioning specification
        (semver.org) into a structured ModelSemVer instance.

        Args:
            version: Semantic version string. Must be a non-empty string
                in the format MAJOR.MINOR.PATCH with optional prerelease
                and build metadata (e.g., "1.2.3", "1.0.0-alpha+build123").

        Returns:
            A new ModelSemVer instance with parsed version components.

        Raises:
            TypeError: If version is not a string.
            ValueError: If version string is invalid. Specific cases include:
                - Empty or whitespace-only string
                - Missing required components (must have MAJOR.MINOR.PATCH)
                - Non-numeric version components
                - Invalid prerelease or build metadata format

        Example:
            >>> ModelSemVer.from_string("1.2.3")
            ModelSemVer(major=1, minor=2, patch=3, prerelease=None, build=None)
            >>> ModelSemVer.from_string("1.0.0-alpha+build123")
            ModelSemVer(major=1, minor=0, patch=0, prerelease='alpha', build='build123')
            >>> ModelSemVer.from_string("")  # Raises ValueError
            >>> ModelSemVer.from_string(123)  # Raises TypeError
        """
        # Type validation - explicit check before any processing
        if not isinstance(version, str):
            raise TypeError(f"version must be a string, got {type(version).__name__}")

        # Empty string validation - check before passing to validator
        if not version or not version.strip():
            raise ValueError("version cannot be empty or whitespace-only")

        # Format validation using semver validator (raises ValueError if invalid)
        validate_semver(version)

        # Parse components - version is now guaranteed to be valid format
        version_str = version
        build_part: str | None = None
        prerelease_part: str | None = None

        if "+" in version_str:
            version_str, build_part = version_str.split("+", 1)
        if "-" in version_str:
            version_str, prerelease_part = version_str.split("-", 1)

        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2])

        # Validate parsed components using shared helper
        _validate_version_component(major, "major")
        _validate_version_component(minor, "minor")
        _validate_version_component(patch, "patch")

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease_part,
            build=build_part,
        )

    @classmethod
    def from_tuple(
        cls,
        version_tuple: tuple[int, int, int],
        prerelease: str | None = None,
        build: str | None = None,
    ) -> ModelSemVer:
        """Create ModelSemVer from a tuple of (major, minor, patch).

        Args:
            version_tuple: A tuple of exactly 3 non-negative integers
                representing (major, minor, patch) version components.
            prerelease: Optional prerelease identifier (e.g., "alpha", "beta.1").
            build: Optional build metadata (e.g., "20231215", "abc123").

        Returns:
            ModelSemVer instance

        Raises:
            ValueError: If tuple does not have exactly 3 elements.
            TypeError: If elements are not integers.
            ValueError: If any element is negative.

        Example:
            >>> ModelSemVer.from_tuple((1, 2, 3))
            ModelSemVer(major=1, minor=2, patch=3, prerelease=None, build=None)
            >>> ModelSemVer.from_tuple((1, 0, 0), prerelease="alpha")
            ModelSemVer(major=1, minor=0, patch=0, prerelease='alpha', build=None)
        """
        # Validate tuple length
        if not isinstance(version_tuple, tuple):
            raise TypeError(f"Expected tuple, got {type(version_tuple).__name__}")
        if len(version_tuple) != 3:
            raise ValueError(
                f"Version tuple must have exactly 3 elements (major, minor, patch), "
                f"got {len(version_tuple)}"
            )

        major, minor, patch = version_tuple

        # Validate each component using shared helper
        _validate_version_component(major, "major")
        _validate_version_component(minor, "minor")
        _validate_version_component(patch, "patch")

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build,
        )

    def __str__(self) -> str:
        """Convert to semver string format."""
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result = f"{result}-{self.prerelease}"
        if self.build:
            result = f"{result}+{self.build}"
        return result

    def __eq__(self, other: object) -> bool:
        """Check equality of semantic versions.

        Two versions are equal if they have the same major, minor, patch,
        and prerelease values. Build metadata is ignored per semver spec.

        Args:
            other: Another object to compare with.

        Returns:
            True if versions are equal, False otherwise.
            NotImplemented if other is not a ModelSemVer.
        """
        if not isinstance(other, ModelSemVer):
            return NotImplemented
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        """Return hash for use in sets and dicts.

        Hash is based on major, minor, patch, and prerelease (same as __eq__).
        Build metadata is excluded per semver spec.
        """
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def __lt__(self, other: object) -> bool:
        """Compare versions for ordering (less than).

        Implements semantic version precedence rules from semver.org:
        1. Major, minor, patch are compared numerically in order
        2. Pre-release versions have lower precedence than release versions
        3. Pre-release identifiers are compared lexicographically

        Args:
            other: Another object to compare against. Must be a ModelSemVer
                instance for comparison to succeed.

        Returns:
            True if self is less than other, False otherwise.
            Returns NotImplemented if other is not a ModelSemVer instance,
            allowing Python to try the reverse comparison or raise TypeError.

        Example:
            >>> v1 = ModelSemVer(major=1, minor=0, patch=0)
            >>> v2 = ModelSemVer(major=2, minor=0, patch=0)
            >>> v1 < v2
            True
            >>> v1 < "1.0.0"  # Returns NotImplemented, raises TypeError
        """
        if not isinstance(other, ModelSemVer):
            return NotImplemented
        # Major, minor, patch comparison
        if (self.major, self.minor, self.patch) != (
            other.major,
            other.minor,
            other.patch,
        ):
            return (self.major, self.minor, self.patch) < (
                other.major,
                other.minor,
                other.patch,
            )
        # Prerelease comparison (release > prerelease)
        if self.prerelease is None and other.prerelease is not None:
            return False
        if self.prerelease is not None and other.prerelease is None:
            return True
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False


# Default version constant
SEMVER_DEFAULT = ModelSemVer(major=1, minor=0, patch=0)

__all__ = ["ModelSemVer", "SEMVER_DEFAULT"]
