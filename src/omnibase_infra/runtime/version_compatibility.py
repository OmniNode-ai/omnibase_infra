# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Version compatibility matrix for ONEX package dependencies.

This module verifies that omnibase_core and omnibase_spi versions meet the
minimum requirements for omnibase_infra at startup. It logs resolved versions
and fails fast with a clear error message if incompatible packages are detected.

The version matrix is defined here as the single source of truth and mirrors
the constraints in pyproject.toml. This runtime check catches version
mismatches that could occur from manual installs, editable mode conflicts,
or CI caching issues.

Architecture:
    Called during RuntimeHostProcess.start() before any other initialization.
    If versions are incompatible, raises InfraVersionIncompatibleError
    which prevents the runtime from starting in a broken state.

Related:
    - OMN-758: INFRA-017: Version compatibility matrix check
    - pyproject.toml: Declarative dependency constraints
    - service_runtime_host_process.py: Runtime startup integration

.. versionadded:: 0.11.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionConstraint:
    """A version constraint for a dependency package.

    Attributes:
        package: The package name (e.g., "omnibase_core").
        min_version: Minimum required version (inclusive).
        max_version: Maximum allowed version (exclusive).
    """

    package: str
    min_version: str
    max_version: str


# ============================================================================
# VERSION COMPATIBILITY MATRIX
# ============================================================================
# This matrix defines the version constraints for omnibase_infra's dependencies.
# It MUST be kept in sync with pyproject.toml.
#
# When bumping dependency versions:
#   1. Update pyproject.toml
#   2. Update this matrix
#   3. Run tests to verify
#
# Format: VersionConstraint(package, min_version_inclusive, max_version_exclusive)
VERSION_MATRIX: list[VersionConstraint] = [
    VersionConstraint(
        package="omnibase_core",
        min_version="0.20.0",
        max_version="0.21.0",
    ),
    VersionConstraint(
        package="omnibase_spi",
        min_version="0.13.0",
        max_version="0.14.0",
    ),
]


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Args:
        version_str: A version string like "0.20.0" or "1.2.3".

    Returns:
        Tuple of integers for comparison.

    Raises:
        ValueError: If the version string contains non-numeric parts.
    """
    parts: list[int] = []
    for part in version_str.split("."):
        # Handle pre-release suffixes (e.g., "0.20.0a1" -> strip "a1")
        numeric = ""
        for char in part:
            if char.isdigit():
                numeric += char
            else:
                break
        if numeric:
            parts.append(int(numeric))
        else:
            parts.append(0)
    return tuple(parts)


def _version_in_range(
    version: str,
    min_version: str,
    max_version: str,
) -> bool:
    """Check if a version falls within [min_version, max_version).

    Args:
        version: The version to check.
        min_version: Minimum version (inclusive).
        max_version: Maximum version (exclusive).

    Returns:
        True if min_version <= version < max_version.
    """
    v = _parse_version(version)
    v_min = _parse_version(min_version)
    v_max = _parse_version(max_version)
    return v_min <= v < v_max


def _get_installed_version(package_name: str) -> str | None:
    """Get the installed version of a package.

    Args:
        package_name: Package name (using underscores, e.g., "omnibase_core").

    Returns:
        Version string, or None if the package is not installed.
    """
    try:
        import importlib

        mod = importlib.import_module(package_name)
        version: str | None = getattr(mod, "__version__", None)
        return version
    except ImportError:
        return None


def check_version_compatibility(
    matrix: list[VersionConstraint] | None = None,
) -> list[str]:
    """Check all dependencies against the version compatibility matrix.

    Args:
        matrix: Version constraints to check. Defaults to VERSION_MATRIX.

    Returns:
        List of error messages for incompatible versions. Empty if all OK.
    """
    if matrix is None:
        matrix = VERSION_MATRIX

    errors: list[str] = []

    for constraint in matrix:
        installed = _get_installed_version(constraint.package)

        if installed is None:
            errors.append(
                f"{constraint.package}: NOT INSTALLED "
                f"(required >={constraint.min_version},<{constraint.max_version})"
            )
            continue

        if not _version_in_range(
            installed, constraint.min_version, constraint.max_version
        ):
            errors.append(
                f"{constraint.package}: {installed} is incompatible "
                f"(required >={constraint.min_version},<{constraint.max_version})"
            )

    return errors


def log_and_verify_versions() -> None:
    """Log resolved dependency versions and fail fast if incompatible.

    This function is called during RuntimeHostProcess.start() to:
    1. Log all resolved package versions for debugging
    2. Verify versions meet minimum requirements
    3. Raise an error if incompatible versions are detected

    Raises:
        RuntimeError: If any dependency version is incompatible.
    """
    import omnibase_infra

    # Log the infra version first
    logger.info(
        "ONEX version compatibility check",
        extra={"omnibase_infra_version": omnibase_infra.__version__},
    )

    # Log each dependency version
    for constraint in VERSION_MATRIX:
        installed = _get_installed_version(constraint.package)
        logger.info(
            "Dependency version resolved",
            extra={
                "package": constraint.package,
                "installed_version": installed or "NOT INSTALLED",
                "required_min": constraint.min_version,
                "required_max": constraint.max_version,
                "compatible": (
                    _version_in_range(
                        installed, constraint.min_version, constraint.max_version
                    )
                    if installed
                    else False
                ),
            },
        )

    # Check compatibility
    errors = check_version_compatibility()
    if errors:
        error_msg = (
            "ONEX version compatibility check FAILED.\n"
            "The following packages do not meet version requirements:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nVersion matrix for omnibase_infra "
            f"{omnibase_infra.__version__}:\n"
        )
        for constraint in VERSION_MATRIX:
            error_msg += (
                f"  {constraint.package}: "
                f">={constraint.min_version},<{constraint.max_version}\n"
            )
        error_msg += (
            "\nTo fix: update dependencies with `uv sync` or "
            "check pyproject.toml version constraints."
        )
        raise RuntimeError(error_msg)

    logger.info("ONEX version compatibility check PASSED")
