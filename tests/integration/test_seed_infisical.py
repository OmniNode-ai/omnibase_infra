# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for seed-infisical.py script (OMN-2287).

Tests the seed script against real contract files without requiring
an Infisical server.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.path_utils import find_project_root

pytestmark = pytest.mark.integration

_REPO_ROOT = find_project_root(Path(__file__).resolve().parent)
SCRIPTS_DIR = _REPO_ROOT / "scripts"
NODES_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"


@pytest.mark.skipif(
    not NODES_DIR.is_dir(),
    reason="Repository nodes directory not available",
)
class TestSeedInfisicalIntegration:
    """Integration tests for the seed script."""

    def test_dry_run_with_real_contracts(self) -> None:
        """Seed script dry-run should succeed with real contracts."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "seed-infisical.py"),
                "--contracts-dir",
                str(NODES_DIR),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            f"Seed script dry-run failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert (
            "Seed Diff Summary" in result.stdout
            or "config requirements" in result.stderr
        )

    def test_dry_run_with_import_env(self, tmp_path: Path) -> None:
        """Seed script should accept --import-env flag."""
        env_file = tmp_path / ".env"
        env_file.write_text("POSTGRES_DSN=postgresql://test\n")

        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "seed-infisical.py"),
                "--contracts-dir",
                str(NODES_DIR),
                "--import-env",
                str(env_file),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            f"Seed script with --import-env failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
