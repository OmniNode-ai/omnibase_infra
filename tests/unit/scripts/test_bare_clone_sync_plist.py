# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the bare-clone-sync launchd plist."""

from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

PLIST_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "ai.omninode.bare-clone-sync.plist"
)


@pytest.mark.unit
class TestBareCloneSyncPlist:
    """Validate the bare-clone-sync launchd plist."""

    def test_plist_exists(self) -> None:
        assert PLIST_PATH.exists(), f"Plist not found at {PLIST_PATH}"

    def test_plist_is_well_formed_xml(self) -> None:
        """Parse the plist with plistlib to verify well-formed XML."""
        data = plistlib.loads(PLIST_PATH.read_bytes())
        assert isinstance(data, dict), "Plist root must be a dict"

    def test_label_matches_filename(self) -> None:
        data = plistlib.loads(PLIST_PATH.read_bytes())
        assert data["Label"] == "ai.omninode.bare-clone-sync"

    def test_program_arguments_uses_absolute_paths(self) -> None:
        data = plistlib.loads(PLIST_PATH.read_bytes())
        args = data["ProgramArguments"]
        assert len(args) == 2
        assert args[0] == "/bin/bash"
        assert args[1].startswith("/"), "pull-all.sh path must be absolute"
        assert "pull-all.sh" in args[1]

    def test_start_interval_is_30_minutes(self) -> None:
        data = plistlib.loads(PLIST_PATH.read_bytes())
        assert data["StartInterval"] == 1800, "StartInterval should be 1800s (30 min)"

    def test_run_at_load_enabled(self) -> None:
        data = plistlib.loads(PLIST_PATH.read_bytes())
        assert data["RunAtLoad"] is True

    def test_log_paths_configured(self) -> None:
        data = plistlib.loads(PLIST_PATH.read_bytes())
        assert data["StandardOutPath"] == "/tmp/bare-clone-sync.log"  # noqa: S108
        assert data["StandardErrorPath"] == "/tmp/bare-clone-sync-error.log"  # noqa: S108

    def test_environment_variables_set(self) -> None:
        data = plistlib.loads(PLIST_PATH.read_bytes())
        env = data["EnvironmentVariables"]
        assert "PATH" in env
        assert "OMNI_HOME" in env
        assert env["OMNI_HOME"] == "/Volumes/PRO-G40/Code/omni_home"
