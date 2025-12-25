# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic versioning validation utilities.

Provides reusable semver pattern and validators for ONEX models.

This module provides two validation approaches:
    - validate_semver: Strict validation requiring full MAJOR.MINOR.PATCH format
    - validate_version_lenient: Lenient validation accepting partial versions (1, 1.0, 1.0.0)
"""

from __future__ import annotations

import re

# Semantic versioning pattern: MAJOR.MINOR.PATCH[-prerelease][+build]
# See: https://semver.org/
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$")


def validate_semver(v: str, field_name: str = "version") -> str:
    """Validate that a string follows strict semantic versioning format.

    Requires full MAJOR.MINOR.PATCH format with optional prerelease and build metadata.

    Args:
        v: The version string to validate.
        field_name: Name of the field for error messages (default: "version").

    Returns:
        The validated version string.

    Raises:
        ValueError: If the version string is not valid semver format.

    Example:
        >>> validate_semver("1.0.0")
        '1.0.0'
        >>> validate_semver("1.2.3-alpha")
        '1.2.3-alpha'
        >>> validate_semver("1.0")  # Raises ValueError - too few components
    """
    if not SEMVER_PATTERN.match(v):
        raise ValueError(
            f"Invalid semantic version '{v}'. "
            "Expected format: MAJOR.MINOR.PATCH[-prerelease][+build]"
        )
    return v


def validate_version_lenient(v: str) -> str:
    """Validate version format with lenient parsing.

    Accepts flexible version formats including partial versions.
    Used by compute registry models that need backward compatibility
    with abbreviated version strings.

    Accepted formats:
        - "1" (major only)
        - "1.0" (major.minor)
        - "1.0.0" (major.minor.patch)
        - "1.2.3-alpha" (with prerelease suffix)
        - "1.2.3-beta.1" (with prerelease segments)

    Args:
        v: The version string to validate.

    Returns:
        The validated and stripped version string.

    Raises:
        ValueError: If version format is invalid:
            - Empty or whitespace-only string
            - Empty prerelease suffix after '-'
            - More than 3 numeric components
            - Empty component between dots
            - Non-integer component
            - Negative component value

    Example:
        >>> validate_version_lenient("1.0.0")
        '1.0.0'
        >>> validate_version_lenient("1.0")
        '1.0'
        >>> validate_version_lenient("1")
        '1'
        >>> validate_version_lenient("2.1.0-beta")
        '2.1.0-beta'
        >>> validate_version_lenient("")  # Raises ValueError
        >>> validate_version_lenient("1.2.3.4")  # Raises ValueError - too many components
    """
    if not v or not v.strip():
        raise ValueError("Version cannot be empty")

    v = v.strip()

    # Split off prerelease suffix (e.g., "1.2.3-alpha" -> "1.2.3", "alpha")
    version_part = v
    if "-" in v:
        version_part, prerelease = v.split("-", 1)
        if not prerelease:
            raise ValueError(
                f"Invalid version '{v}': prerelease cannot be empty after '-'"
            )

    # Parse major.minor.patch components
    parts = version_part.split(".")
    if len(parts) < 1 or len(parts) > 3:
        raise ValueError(f"Invalid version '{v}': expected format 'major.minor.patch'")

    for part in parts:
        if not part:
            raise ValueError(f"Invalid version '{v}': empty component")
        try:
            num = int(part)
            if num < 0:
                raise ValueError(f"Invalid version '{v}': negative component")
        except ValueError as e:
            if "negative component" in str(e):
                raise
            raise ValueError(
                f"Invalid version '{v}': non-integer component '{part}'"
            ) from None

    return v


__all__ = ["SEMVER_PATTERN", "validate_semver", "validate_version_lenient"]
