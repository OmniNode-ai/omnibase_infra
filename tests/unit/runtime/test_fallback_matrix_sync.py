# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that _FALLBACK_MATRIX stays in sync with pyproject.toml.

Bug (OMN-6442): The fallback matrix drifted from pyproject.toml during a
version bump, causing version check failures inside Docker where
pyproject.toml is absent.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.runtime.version_compatibility import (
    _FALLBACK_MATRIX,
    _parse_version,
)


@pytest.mark.unit
class TestFallbackMatrixSync:
    """Verify _FALLBACK_MATRIX matches pyproject.toml bounds."""

    def test_update_version_matrix_check_passes(self) -> None:
        """scripts/update_version_matrix.py --check must exit 0.

        This is the canonical CI gate that detects fallback drift.
        If this fails, run: uv run python scripts/update_version_matrix.py
        """
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        script = repo_root / "scripts" / "update_version_matrix.py"

        if not script.exists():
            pytest.skip("update_version_matrix.py not found (not in repo clone)")

        result = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
        )
        assert result.returncode == 0, (
            f"Fallback matrix drift detected. "
            f"Run: uv run python scripts/update_version_matrix.py\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_fallback_matrix_min_versions_resolve_in_lock(self) -> None:
        """Every _FALLBACK_MATRIX min_version must be <= the locked version.

        Catches the specific bug where min_version was bumped in fallback
        but uv.lock still resolved an older version.
        """
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        lock_path = repo_root / "uv.lock"

        if not lock_path.exists():
            pytest.skip("uv.lock not found (not in repo clone)")

        lock_text = lock_path.read_text()

        for constraint in _FALLBACK_MATRIX:
            # Find the package version in uv.lock
            # uv.lock format: [[package]]\nname = "omnibase-core"\nversion = "0.28.0"
            dist_name = constraint.package.replace("_", "-")
            pattern = (
                rf'name\s*=\s*"{re.escape(dist_name)}"\s*\n' rf'version\s*=\s*"([^"]+)"'
            )
            match = re.search(pattern, lock_text)
            if match is None:
                pytest.fail(f"{constraint.package} ({dist_name}) not found in uv.lock")

            locked_version = match.group(1)
            assert _parse_version(locked_version) >= _parse_version(
                constraint.min_version
            ), (
                f"{constraint.package}: uv.lock has {locked_version} but "
                f"_FALLBACK_MATRIX requires >= {constraint.min_version}. "
                f"Either relax the fallback or bump the lock."
            )
