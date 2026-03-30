# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for pull-all.sh and bare-clone-sync infrastructure."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
PULL_ALL = SCRIPTS_DIR / "pull-all.sh"
PLIST = SCRIPTS_DIR / "ai.omninode.bare-clone-sync.plist"
INSTALL_SCRIPT = SCRIPTS_DIR / "install-bare-clone-sync.sh"


@pytest.mark.unit
class TestPullAllScript:
    """Tests for pull-all.sh."""

    def test_script_exists_and_executable(self) -> None:
        assert PULL_ALL.exists(), f"pull-all.sh not found at {PULL_ALL}"
        assert os.access(PULL_ALL, os.X_OK), "pull-all.sh is not executable"

    def test_script_has_spdx_header(self) -> None:
        content = PULL_ALL.read_text()
        assert "SPDX-License-Identifier" in content

    def test_script_uses_parallel_execution(self) -> None:
        """Verify pull-all.sh uses parallel fetching (OMN-6869)."""
        content = PULL_ALL.read_text()
        # Should background repo fetches with &
        assert "_pull_one" in content, (
            "Expected _pull_one helper for parallel execution"
        )
        assert "wait" in content, "Expected 'wait' for parallel job synchronization"

    def test_script_uses_temp_dir_for_results(self) -> None:
        """Verify parallel results are aggregated via temp files."""
        content = PULL_ALL.read_text()
        assert "mktemp -d" in content, "Expected mktemp -d for result aggregation"
        assert "trap" in content, "Expected trap for temp dir cleanup"

    def test_script_handles_missing_repo(self) -> None:
        """Run pull-all.sh with a nonexistent repo name against a temp dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["bash", str(PULL_ALL), "nonexistent_repo_xyz"],
                capture_output=True,
                text=True,
                env={**os.environ, "OMNI_HOME": tmpdir},
                check=False,
            )
            assert result.returncode != 0
            assert "MISSING" in result.stdout

    def test_script_handles_bare_repo_no_crash(self) -> None:
        """Create a bare git repo and verify pull-all.sh does not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_name = "test_repo"
            repo_path = Path(tmpdir) / repo_name
            subprocess.run(
                ["git", "init", "--bare", str(repo_path)],
                capture_output=True,
                check=True,
            )

            result = subprocess.run(
                ["bash", str(PULL_ALL), repo_name],
                capture_output=True,
                text=True,
                env={**os.environ, "OMNI_HOME": tmpdir},
                check=False,
            )
            # Script should complete (exit 0 or 1) without crashing.
            # The bare repo has no main ref and no remote, so it will
            # report FAILED — that's expected. The key is no crash.
            assert result.returncode in (0, 1)


@pytest.mark.unit
class TestLaunchdPlist:
    """Tests for the bare-clone-sync launchd plist."""

    def test_plist_exists(self) -> None:
        assert PLIST.exists(), f"plist not found at {PLIST}"

    def test_plist_is_well_formed(self) -> None:
        content = PLIST.read_text()
        assert content.startswith("<?xml version=")
        assert '<plist version="1.0">' in content
        assert "</plist>" in content

    def test_plist_has_correct_label(self) -> None:
        content = PLIST.read_text()
        assert "ai.omninode.bare-clone-sync" in content

    def test_plist_interval_is_1800(self) -> None:
        content = PLIST.read_text()
        assert "<integer>1800</integer>" in content

    def test_plist_references_pull_all(self) -> None:
        content = PLIST.read_text()
        assert "pull-all.sh" in content


@pytest.mark.unit
class TestInstallScript:
    """Tests for install-bare-clone-sync.sh."""

    def test_install_script_exists_and_executable(self) -> None:
        assert INSTALL_SCRIPT.exists()
        assert os.access(INSTALL_SCRIPT, os.X_OK)

    def test_install_script_has_spdx_header(self) -> None:
        content = INSTALL_SCRIPT.read_text()
        assert "SPDX-License-Identifier" in content

    def test_install_script_supports_uninstall(self) -> None:
        content = INSTALL_SCRIPT.read_text()
        assert "uninstall" in content

    def test_install_script_uses_launchctl(self) -> None:
        content = INSTALL_SCRIPT.read_text()
        assert "launchctl" in content
