# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Version Normalization Utilities.

This module provides shared version normalization logic for consistent
version handling across all ONEX components. The normalization ensures
that equivalent version strings (e.g., "1.0", "1.0.0", "v1.0.0") all
resolve to the same canonical format.

This is the SINGLE SOURCE OF TRUTH for version normalization in the
runtime package, eliminating code duplication across RegistryPolicy,
ModelPolicyKey, and ModelPolicyRegistration.
"""

from __future__ import annotations

from omnibase_core.models.primitives import ModelSemVer


def normalize_version(version: str) -> str:
    """Normalize version string for consistent lookups using ModelSemVer.

    Converts version strings to canonical x.y.z format. This ensures consistent
    version handling across all ONEX components, preventing lookup mismatches
    where "1.0.0" and "1.0" might be treated as different versions.

    This is the SINGLE SOURCE OF TRUTH for version normalization, used by:
        - RegistryPolicy._normalize_version()
        - ModelPolicyKey.validate_and_normalize_version()
        - ModelPolicyRegistration.validate_and_normalize_version()

    Normalization rules:
        1. Strip leading/trailing whitespace
        2. Strip leading 'v' or 'V' prefix
        3. Expand partial versions (1 -> 1.0.0, 1.0 -> 1.0.0)
        4. Parse with ModelSemVer.parse() for validation
        5. Preserve prerelease suffix if present

    Args:
        version: The version string to normalize

    Returns:
        Normalized version string in "x.y.z" or "x.y.z-prerelease" format

    Raises:
        ValueError: If the version string is empty or invalid

    Examples:
        >>> normalize_version("1.0")
        '1.0.0'
        >>> normalize_version("v2.1")
        '2.1.0'
        >>> normalize_version("1.0.0-beta")
        '1.0.0-beta'
        >>> normalize_version("  1.2.3  ")
        '1.2.3'
    """
    if not version or not version.strip():
        raise ValueError("Version cannot be empty")

    # Strip whitespace
    normalized = version.strip()

    # Strip leading 'v' or 'V' prefix
    if normalized.startswith(("v", "V")):
        normalized = normalized[1:]

    # Check for empty prerelease suffix (e.g., "1.0.0-")
    if normalized.endswith("-"):
        raise ValueError("Prerelease suffix cannot be empty after hyphen")

    # Split on first hyphen to handle prerelease suffix
    parts = normalized.split("-", 1)
    version_part = parts[0]
    prerelease = parts[1] if len(parts) > 1 else None

    # Expand to three-part version (x.y.z) for ModelSemVer parsing
    version_nums = version_part.split(".")
    while len(version_nums) < 3:
        version_nums.append("0")
    expanded_version = ".".join(version_nums)

    # Parse with ModelSemVer for validation
    try:
        semver = ModelSemVer.parse(expanded_version)
    except Exception as e:
        raise ValueError(f"Invalid version format: {e}") from e

    result: str = semver.to_string()

    # Re-add prerelease if present
    if prerelease:
        result = f"{result}-{prerelease}"

    return result


__all__: list[str] = ["normalize_version"]
