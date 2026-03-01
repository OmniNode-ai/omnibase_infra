# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Version compatibility matrix for ONEX package dependencies.

This module verifies that omnibase_core and omnibase_spi versions meet the
minimum requirements for omnibase_infra at startup. It logs resolved versions
and fails fast with a clear error message if incompatible packages are detected.

The version matrix is derived at import time from pyproject.toml, so it stays
automatically in sync with the declared dependency bounds after every version
bump. No manual update of this file is required when bumping dependencies.

Architecture:
    Called during RuntimeHostProcess.start() before any other initialization.
    If versions are incompatible, raises InfraVersionIncompatibleError
    which prevents the runtime from starting in a broken state.

Related:
    - OMN-758: INFRA-017: Version compatibility matrix check
    - OMN-3203: Automate version_compatibility.py matrix updates
    - pyproject.toml: Declarative dependency constraints (single source of truth)
    - service_runtime_host_process.py: Runtime startup integration
    - scripts/update_version_matrix.py: Standalone BUMP-phase update script

.. versionadded:: 0.11.0
.. versionchanged:: next (OMN-3203) Matrix auto-derived from pyproject.toml
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Packages whose version bounds we track in the matrix.
# Add entries here if new ONEX internal dependencies are introduced.
_TRACKED_PACKAGES: tuple[str, ...] = ("omnibase_core", "omnibase_spi")


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


def _locate_pyproject() -> Path | None:
    """Walk up from this file's location to find pyproject.toml.

    Returns:
        Path to pyproject.toml, or None if not found (e.g. installed package
        without source tree present).
    """
    candidate = Path(__file__).resolve()
    for _ in range(10):
        candidate = candidate.parent
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file():
            return pyproject
    return None


def _parse_dep_bounds(spec: str) -> tuple[str, str] | None:
    """Extract (min_version, max_version) from a PEP 508 specifier like
    ``omnibase-core>=0.22.0,<0.23.0``.

    Args:
        spec: A single dependency string from pyproject.toml.

    Returns:
        (min_version, max_version) tuple, or None if bounds are incomplete.
    """
    min_match = re.search(r">=\s*([0-9][^\s,;\"']*)", spec)
    max_match = re.search(r"<\s*([0-9][^\s,;\"']*)", spec)
    if min_match and max_match:
        return min_match.group(1), max_match.group(1)
    return None


def _build_matrix_from_pyproject(
    tracked: tuple[str, ...] = _TRACKED_PACKAGES,
) -> list[VersionConstraint] | None:
    """Build the version matrix by reading pyproject.toml.

    Args:
        tracked: Package names (underscore form) to include in the matrix.

    Returns:
        List of VersionConstraint objects, or None if pyproject.toml could
        not be located or parsed.
    """
    pyproject_path = _locate_pyproject()
    if pyproject_path is None:
        return None

    try:
        with open(pyproject_path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None

    dependencies: list[str] = data.get("project", {}).get("dependencies", [])

    # Build a normalised-name → spec map
    spec_map: dict[str, str] = {}
    for dep in dependencies:
        # Normalise hyphens to underscores for lookup
        dep_clean = dep.strip().strip("\"'")
        # Split on the first version operator to get the package name
        name_part = re.split(r"[><=!~]", dep_clean, maxsplit=1)[0].strip()
        normalised = name_part.replace("-", "_")
        spec_map[normalised] = dep_clean

    constraints: list[VersionConstraint] = []
    for pkg in tracked:
        spec = spec_map.get(pkg)
        if spec is None:
            # Package not in pyproject.toml — skip rather than hard-fail
            continue
        bounds = _parse_dep_bounds(spec)
        if bounds is None:
            continue
        min_ver, max_ver = bounds
        constraints.append(
            VersionConstraint(package=pkg, min_version=min_ver, max_version=max_ver)
        )

    return constraints if constraints else None


# ============================================================================
# VERSION COMPATIBILITY MATRIX
# ============================================================================
# Derived automatically from pyproject.toml at import time (OMN-3203).
# No manual update needed — just bump pyproject.toml and the matrix follows.
#
# If pyproject.toml cannot be found (e.g. installed package without source),
# we fall back to the last-known-good hardcoded values so the runtime check
# still operates rather than silently skipping.
_FALLBACK_MATRIX: list[VersionConstraint] = [
    VersionConstraint(
        package="omnibase_core",
        min_version="0.22.0",
        max_version="0.23.0",
    ),
    VersionConstraint(
        package="omnibase_spi",
        min_version="0.15.0",
        max_version="0.16.0",
    ),
]

VERSION_MATRIX: list[VersionConstraint] = (
    _build_matrix_from_pyproject() or _FALLBACK_MATRIX
)


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
