# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic versioning validation utilities.

Provides reusable semver pattern and validator for ONEX models.
"""

from __future__ import annotations

import re

# Semantic versioning pattern: MAJOR.MINOR.PATCH[-prerelease][+build]
# See: https://semver.org/
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$")


def validate_semver(v: str, field_name: str = "version") -> str:
    """Validate that a string follows semantic versioning format.

    Args:
        v: The version string to validate.
        field_name: Name of the field for error messages (default: "version").

    Returns:
        The validated version string.

    Raises:
        ValueError: If the version string is not valid semver format.
    """
    if not SEMVER_PATTERN.match(v):
        raise ValueError(
            f"Invalid semantic version '{v}'. "
            "Expected format: MAJOR.MINOR.PATCH[-prerelease][+build]"
        )
    return v


__all__ = ["SEMVER_PATTERN", "validate_semver"]
