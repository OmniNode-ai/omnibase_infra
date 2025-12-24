# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic version model for ONEX infrastructure.

Provides a strongly-typed Pydantic model for semantic versioning,
replacing raw string usage with validated, structured version data.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_infra.utils.util_semver import validate_semver


class ModelSemVer(BaseModel):
    """Semantic version model following semver.org specification.

    Provides structured representation of semantic versions with validation.
    Supports serialization to string and comparison operations.

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
    """

    major: int = Field(default=1, ge=0, description="Major version number")
    minor: int = Field(default=0, ge=0, description="Minor version number")
    patch: int = Field(default=0, ge=0, description="Patch version number")
    prerelease: str | None = Field(default=None, description="Prerelease identifier")
    build: str | None = Field(default=None, description="Build metadata")

    @classmethod
    def from_string(cls, version: str) -> ModelSemVer:
        """Parse a semantic version string into ModelSemVer.

        Args:
            version: Semantic version string (e.g., "1.2.3", "1.0.0-alpha+build")

        Returns:
            ModelSemVer instance

        Raises:
            ValueError: If version string is invalid
        """
        # Use existing validator first
        validate_semver(version)

        # Parse components
        build_part = None
        prerelease_part = None

        if "+" in version:
            version, build_part = version.split("+", 1)
        if "-" in version:
            version, prerelease_part = version.split("-", 1)

        parts = version.split(".")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            prerelease=prerelease_part,
            build=build_part,
        )

    def __str__(self) -> str:
        """Convert to semver string format."""
        result = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            result = f"{result}-{self.prerelease}"
        if self.build:
            result = f"{result}+{self.build}"
        return result

    def __lt__(self, other: ModelSemVer) -> bool:
        """Compare versions for sorting."""
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
