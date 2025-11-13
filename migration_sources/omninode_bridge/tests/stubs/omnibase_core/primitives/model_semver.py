#!/usr/bin/env python3
"""
Stub for ModelSemVer from omnibase_core.

Semantic versioning model for ONEX v2.0 compliance.
"""

from typing import Optional

from pydantic import BaseModel, Field, validator


class ModelSemVer(BaseModel):
    """
    Semantic versioning model.

    Stub implementation for testing.
    """

    major: int = Field(..., ge=0, description="Major version number")
    minor: int = Field(..., ge=0, description="Minor version number")
    patch: int = Field(..., ge=0, description="Patch version number")
    prerelease: Optional[str] = Field(
        None, description="Prerelease identifier (e.g., 'alpha', 'beta')"
    )
    build: Optional[str] = Field(None, description="Build metadata")

    @validator("major", "minor", "patch")
    def validate_non_negative(cls, v):
        """Ensure version numbers are non-negative."""
        if v < 0:
            raise ValueError("Version numbers must be non-negative")
        return v

    def __str__(self) -> str:
        """String representation of semantic version."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def from_string(cls, version_string: str) -> "ModelSemVer":
        """
        Parse semantic version from string.

        Args:
            version_string: Version string (e.g., "1.2.3", "1.2.3-alpha", "1.2.3+build")

        Returns:
            ModelSemVer instance
        """
        # Simple parsing (stub implementation)
        parts = version_string.split(".")
        if len(parts) < 3:
            raise ValueError(f"Invalid version string: {version_string}")

        major = int(parts[0])
        minor = int(parts[1])

        # Handle patch with optional prerelease/build
        patch_parts = parts[2].split("-", 1)
        patch = int(patch_parts[0].split("+", 1)[0])

        prerelease = None
        build = None

        if "-" in parts[2]:
            prerelease = patch_parts[1].split("+", 1)[0]

        if "+" in parts[2]:
            build = parts[2].split("+", 1)[1]

        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build,
        )

    class Config:
        """Pydantic model configuration."""

        extra = "forbid"
