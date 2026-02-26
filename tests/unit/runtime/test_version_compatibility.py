# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for version compatibility matrix check (INFRA-017).

These tests verify that the version compatibility checking logic correctly
detects compatible and incompatible package versions, and that the version
matrix stays in sync with pyproject.toml.

Related:
    - OMN-758: INFRA-017: Version compatibility matrix check
    - omnibase_infra.runtime.version_compatibility: Implementation

.. versionadded:: 0.11.0
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.runtime.version_compatibility import (
    VERSION_MATRIX,
    VersionConstraint,
    _parse_version,
    _version_in_range,
    check_version_compatibility,
    log_and_verify_versions,
)


@pytest.mark.unit
class TestParseVersion:
    """Tests for version string parsing."""

    def test_simple_version(self) -> None:
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_two_part_version(self) -> None:
        assert _parse_version("0.20") == (0, 20)

    def test_single_part_version(self) -> None:
        assert _parse_version("5") == (5,)

    def test_zero_version(self) -> None:
        assert _parse_version("0.0.0") == (0, 0, 0)

    def test_prerelease_suffix_stripped(self) -> None:
        """Pre-release suffixes are stripped to numeric parts."""
        result = _parse_version("0.20.0a1")
        assert result == (0, 20, 0)

    def test_large_version_numbers(self) -> None:
        assert _parse_version("100.200.300") == (100, 200, 300)


@pytest.mark.unit
class TestVersionInRange:
    """Tests for version range checking."""

    def test_version_at_minimum(self) -> None:
        """Minimum version is inclusive."""
        assert _version_in_range("0.20.0", "0.20.0", "0.21.0") is True

    def test_version_at_maximum(self) -> None:
        """Maximum version is exclusive."""
        assert _version_in_range("0.21.0", "0.20.0", "0.21.0") is False

    def test_version_in_middle(self) -> None:
        assert _version_in_range("0.20.5", "0.20.0", "0.21.0") is True

    def test_version_below_minimum(self) -> None:
        assert _version_in_range("0.19.0", "0.20.0", "0.21.0") is False

    def test_version_above_maximum(self) -> None:
        assert _version_in_range("0.22.0", "0.20.0", "0.21.0") is False

    def test_patch_version_within_range(self) -> None:
        assert _version_in_range("0.20.99", "0.20.0", "0.21.0") is True

    def test_major_version_mismatch(self) -> None:
        assert _version_in_range("1.0.0", "0.20.0", "0.21.0") is False


@pytest.mark.unit
class TestCheckVersionCompatibility:
    """Tests for the full compatibility check function."""

    def test_compatible_versions(self) -> None:
        """No errors when all versions are in range."""
        matrix = [
            VersionConstraint("omnibase_core", "0.1.0", "99.0.0"),
            VersionConstraint("omnibase_spi", "0.1.0", "99.0.0"),
        ]
        errors = check_version_compatibility(matrix)
        assert errors == []

    def test_missing_package(self) -> None:
        """Missing package produces clear error."""
        matrix = [
            VersionConstraint("nonexistent_package_xyz", "1.0.0", "2.0.0"),
        ]
        errors = check_version_compatibility(matrix)
        assert len(errors) == 1
        assert "NOT INSTALLED" in errors[0]
        assert "nonexistent_package_xyz" in errors[0]

    def test_incompatible_version(self) -> None:
        """Incompatible version produces clear error with version details."""
        matrix = [
            VersionConstraint("omnibase_core", "99.0.0", "100.0.0"),
        ]
        errors = check_version_compatibility(matrix)
        assert len(errors) == 1
        assert "incompatible" in errors[0]
        assert "omnibase_core" in errors[0]

    def test_default_matrix_passes(self) -> None:
        """The default VERSION_MATRIX must pass with installed packages.

        This is the critical test: it verifies the currently installed
        packages are compatible with the declared matrix. If this fails,
        either the matrix or the installed packages need updating.
        """
        errors = check_version_compatibility()
        assert errors == [], (
            "Default VERSION_MATRIX check failed with currently installed packages:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def test_empty_matrix(self) -> None:
        """Empty matrix produces no errors."""
        errors = check_version_compatibility([])
        assert errors == []


@pytest.mark.unit
class TestLogAndVerifyVersions:
    """Tests for the log_and_verify_versions entry point."""

    def test_passes_with_current_versions(self) -> None:
        """Must not raise with currently installed packages."""
        # This should not raise
        log_and_verify_versions()

    def test_raises_on_incompatible(self) -> None:
        """Must raise RuntimeError when versions are incompatible."""
        bad_matrix = [
            VersionConstraint("omnibase_core", "99.0.0", "100.0.0"),
        ]
        with (
            patch(
                "omnibase_infra.runtime.version_compatibility.VERSION_MATRIX",
                bad_matrix,
            ),
            pytest.raises(RuntimeError, match="compatibility check FAILED"),
        ):
            log_and_verify_versions()

    def test_error_message_contains_fix_instructions(self) -> None:
        """Error message must contain actionable fix instructions."""
        bad_matrix = [
            VersionConstraint("omnibase_core", "99.0.0", "100.0.0"),
        ]
        with (
            patch(
                "omnibase_infra.runtime.version_compatibility.VERSION_MATRIX",
                bad_matrix,
            ),
            pytest.raises(RuntimeError, match="uv sync") as exc_info,
        ):
            log_and_verify_versions()

        error_msg = str(exc_info.value)
        assert "uv sync" in error_msg
        assert "pyproject.toml" in error_msg


@pytest.mark.unit
class TestVersionMatrixConsistency:
    """Tests that the VERSION_MATRIX is consistent with pyproject.toml."""

    def test_matrix_has_core_and_spi(self) -> None:
        """Matrix must include both omnibase_core and omnibase_spi."""
        packages = {c.package for c in VERSION_MATRIX}
        assert "omnibase_core" in packages
        assert "omnibase_spi" in packages

    def test_matrix_min_less_than_max(self) -> None:
        """Min version must be less than max version for every constraint."""
        for constraint in VERSION_MATRIX:
            v_min = _parse_version(constraint.min_version)
            v_max = _parse_version(constraint.max_version)
            assert v_min < v_max, (
                f"{constraint.package}: min_version {constraint.min_version} "
                f"is not less than max_version {constraint.max_version}"
            )

    def test_matrix_matches_pyproject(self) -> None:
        """VERSION_MATRIX constraints must match pyproject.toml.

        This test reads pyproject.toml and verifies the matrix stays in sync.
        """
        import tomllib
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        pyproject_path = repo_root / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        dependencies: list[str] = pyproject["project"]["dependencies"]

        # Build a map of package -> version spec from pyproject.toml
        pyproject_specs: dict[str, str] = {}
        for dep in dependencies:
            # Parse "omnibase-core>=0.20.0,<0.21.0" format
            # Normalize package name (hyphens -> underscores)
            dep_clean = dep.strip().strip('"').strip("'")
            for sep in (">=", "<=", "==", "~=", ">", "<", "!="):
                if sep in dep_clean:
                    pkg_name = dep_clean.split(sep)[0].strip()
                    pkg_name = pkg_name.replace("-", "_")
                    pyproject_specs[pkg_name] = dep_clean
                    break

        for constraint in VERSION_MATRIX:
            assert constraint.package in pyproject_specs, (
                f"{constraint.package} is in VERSION_MATRIX but not in "
                "pyproject.toml dependencies"
            )

            spec = pyproject_specs[constraint.package]
            # Verify min version appears in the spec
            assert constraint.min_version in spec, (
                f"{constraint.package}: VERSION_MATRIX min_version "
                f"{constraint.min_version} not found in pyproject.toml spec: {spec}"
            )
            # Verify max version appears in the spec
            assert constraint.max_version in spec, (
                f"{constraint.package}: VERSION_MATRIX max_version "
                f"{constraint.max_version} not found in pyproject.toml spec: {spec}"
            )

    def test_constraint_is_frozen_dataclass(self) -> None:
        """VersionConstraint must be immutable (frozen dataclass)."""
        constraint = VERSION_MATRIX[0]
        with pytest.raises(AttributeError):
            constraint.package = "hacked"  # type: ignore[misc]
