# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Semantic versioning validation utilities.

Provides reusable semver pattern and validators for ONEX models.

This module provides two validation approaches:
    - validate_semver: Strict validation requiring full MAJOR.MINOR.PATCH format
    - validate_version_lenient: Lenient validation accepting partial versions (1, 1.0, 1.0.0)

.. deprecated:: 1.1.0
    String version input to normalize_version() and normalize_version_cached()
    is deprecated. In v2.0.0, these functions will require ModelSemVer instances
    or structured version dictionaries instead of raw version strings.

    Recommended migration:
        # Instead of:
        normalize_version("1.0.0")

        # Use ModelSemVer directly:
        from omnibase_core.models.primitives.model_semver import ModelSemVer
        version = ModelSemVer(major=1, minor=0, patch=0)
        version_str = version.to_string()  # "1.0.0"

    Deprecation Timeline:
        - v1.1.0: FutureWarning added for string version input
        - v1.2.0: Warning upgraded to DeprecationWarning (visible by default)
        - v2.0.0: String input will raise TypeError; require ModelSemVer instance
"""

from __future__ import annotations

import re
import warnings

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


def normalize_version(version: str, *, _emit_warning: bool = True) -> str:
    """Normalize version string to canonical x.y.z format using ModelSemVer.

    This is the SINGLE SOURCE OF TRUTH for version normalization across the
    omnibase_infra codebase. All version normalization should use this function
    to ensure consistent behavior.

    .. deprecated:: 1.1.0
        String version input is deprecated. Use ModelSemVer instances directly
        for version handling. String input will be removed in v2.0.0.

        Deprecation Timeline:
            - v1.1.0: FutureWarning added for string version input
            - v1.2.0: Warning upgraded to DeprecationWarning (visible by default)
            - v2.0.0: String input will raise TypeError; require ModelSemVer instance

        Migration Guide:
            # Instead of:
            version_str = normalize_version("1.0.0")

            # Use ModelSemVer directly:
            from omnibase_core.models.primitives.model_semver import ModelSemVer
            version = ModelSemVer(major=1, minor=0, patch=0)
            version_str = version.to_string()  # "1.0.0"

            # Or for parsing external input:
            version = ModelSemVer.parse("1.0.0")
            version_str = version.to_string()

    Normalization rules:
        1. Strip leading/trailing whitespace
        2. Strip leading 'v' or 'V' prefix (e.g., "v1.0.0" -> "1.0.0")
        3. Validate format with validate_version_lenient (accepts 1, 1.0, 1.0.0)
        4. Expand to three-part version (e.g., "1" -> "1.0.0", "1.2" -> "1.2.0")
        5. Parse with ModelSemVer.parse() for final validation
        6. Return normalized string via ModelSemVer.to_string()
        7. Re-add prerelease suffix if present

    Args:
        version: The version string to normalize
        _emit_warning: Internal flag to control warning emission.
            Used by normalize_version_cached to emit warning only once per
            unique input string rather than on every cache hit.
            Callers should NOT set this parameter.

    Returns:
        Normalized version string in "x.y.z" or "x.y.z-prerelease" format

    Raises:
        ValueError: If the version string is invalid and cannot be parsed

    Examples:
        >>> normalize_version("1.0")
        '1.0.0'
        >>> normalize_version("v2.1")
        '2.1.0'
        >>> normalize_version("  1.2.3-beta  ")
        '1.2.3-beta'
        >>> normalize_version("1")
        '1.0.0'
    """
    # Import here to avoid circular import at module level
    from omnibase_core.models.primitives.model_semver import ModelSemVer

    # Don't normalize empty/whitespace-only - let downstream validation handle it
    if not version or not version.strip():
        return version

    # Emit deprecation warning for string input
    # Using FutureWarning for better visibility (shown by default in Python)
    # stacklevel=2 points to the caller of normalize_version
    if _emit_warning:
        warnings.warn(
            "Passing string version to normalize_version() is deprecated. "
            "Use ModelSemVer(major=X, minor=Y, patch=Z) for structured version handling, "
            "or ModelSemVer.parse() for external input. "
            "String input will be removed in v2.0.0.",
            FutureWarning,
            stacklevel=2,
        )

    # Strip whitespace
    normalized = version.strip()

    # Strip leading 'v' or 'V' prefix
    if normalized.startswith(("v", "V")):
        normalized = normalized[1:]

    # Validate format with lenient parsing (accepts 1, 1.0, 1.0.0)
    # This will raise ValueError for invalid formats
    validate_version_lenient(normalized)

    # Split on first hyphen to handle prerelease suffix
    parts = normalized.split("-", 1)
    version_part = parts[0]
    prerelease = parts[1] if len(parts) > 1 else None

    # Expand to three-part version (x.y.z) for ModelSemVer parsing
    version_nums = version_part.split(".")
    while len(version_nums) < 3:
        version_nums.append("0")
    expanded_version = ".".join(version_nums)

    # Parse base version with ModelSemVer for validation
    # Uses parse() from omnibase_core ModelSemVer
    # Note: core ModelSemVer strips prerelease from to_string(), so we
    # must preserve and re-add the prerelease suffix manually
    try:
        semver = ModelSemVer.parse(expanded_version)
    except Exception as e:
        # Convert ModelOnexError to ValueError for consistent error handling
        raise ValueError(str(e)) from e
    normalized = semver.to_string()

    # Re-add prerelease if present (core ModelSemVer strips it from to_string)
    if prerelease:
        normalized = f"{normalized}-{prerelease}"

    return normalized


# LRU cache for normalize_version to avoid redundant processing
# Cache size of 256 is sufficient for typical deployments with < 200 unique versions
_normalize_version_cache: dict[str, str] = {}
_NORMALIZE_CACHE_MAX_SIZE: int = 256


def normalize_version_cached(version: str) -> str:
    """Cached version of normalize_version for performance.

    Uses a simple dict cache with size limit to avoid memory growth.
    This is preferred over functools.lru_cache because it allows
    programmatic cache clearing for testing.

    .. deprecated:: 1.1.0
        String version input is deprecated. Use ModelSemVer instances directly
        for version handling. String input will be removed in v2.0.0.

        See normalize_version() docstring for full deprecation details and
        migration guide.

        Note: The FutureWarning is emitted only once per unique version string
        (on cache miss), not on every call. This prevents warning spam while
        still alerting developers to update their code.

    Performance Characteristics:
        - Cache hit: O(1) dict lookup
        - Cache miss: normalize_version() + O(1) dict insert
        - Memory: ~256 * 100 bytes = ~25KB worst case

    Args:
        version: The version string to normalize

    Returns:
        Normalized version string (cached)

    Raises:
        ValueError: If the version string is invalid

    Example:
        >>> normalize_version_cached("1.0")
        '1.0.0'
        >>> normalize_version_cached("1.0")  # Cache hit
        '1.0.0'
    """
    # Check cache first - no warning on cache hit
    if version in _normalize_version_cache:
        return _normalize_version_cache[version]

    # Cache miss - emit warning once (on first occurrence of this version string)
    # Using FutureWarning for better visibility (shown by default in Python)
    # stacklevel=2 points to the caller of normalize_version_cached
    warnings.warn(
        "Passing string version to normalize_version_cached() is deprecated. "
        "Use ModelSemVer(major=X, minor=Y, patch=Z) for structured version handling, "
        "or ModelSemVer.parse() for external input. "
        "String input will be removed in v2.0.0.",
        FutureWarning,
        stacklevel=2,
    )

    # Normalize without emitting warning again (already emitted above)
    normalized = normalize_version(version, _emit_warning=False)

    # Simple cache eviction: clear half when full
    if len(_normalize_version_cache) >= _NORMALIZE_CACHE_MAX_SIZE:
        # Remove oldest half of entries (approximate LRU)
        keys_to_remove = list(_normalize_version_cache.keys())[
            : _NORMALIZE_CACHE_MAX_SIZE // 2
        ]
        for key in keys_to_remove:
            del _normalize_version_cache[key]

    _normalize_version_cache[version] = normalized
    return normalized


def clear_normalize_version_cache() -> None:
    """Clear the normalize_version cache. For testing only.

    This allows tests to verify cache behavior and ensure test isolation.

    Example:
        >>> clear_normalize_version_cache()
        >>> # Cache is now empty
    """
    _normalize_version_cache.clear()


__all__ = [
    "SEMVER_PATTERN",
    "validate_semver",
    "validate_version_lenient",
    "normalize_version",
    "normalize_version_cached",
    "clear_normalize_version_cache",
]
