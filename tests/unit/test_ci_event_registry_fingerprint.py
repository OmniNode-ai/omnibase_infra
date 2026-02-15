# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the event registry fingerprint CI twin script (OMN-2149).

Tests cover the CI twin wrapper that delegates to the existing
``validate_event_registry_fingerprint()`` and ``_cli_stamp()`` functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_event_registry_fingerprint import (
    cmd_stamp,
    cmd_verify,
    main,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# TestCmdVerify
# ---------------------------------------------------------------------------


class TestCmdVerify:
    """Tests for cmd_verify()."""

    def test_verify_passes_with_valid_artifact(self, tmp_path: Path) -> None:
        """Verify returns 0 when artifact matches live registrations."""
        artifact = tmp_path / "fingerprint.json"
        # Stamp a valid artifact first
        cmd_stamp(str(artifact))
        assert cmd_verify(str(artifact)) == 0

    def test_verify_fails_with_missing_artifact(self, tmp_path: Path) -> None:
        """Verify returns 2 when artifact file does not exist."""
        artifact = tmp_path / "nonexistent.json"
        assert cmd_verify(str(artifact)) == 2

    def test_verify_fails_with_stale_artifact(self, tmp_path: Path) -> None:
        """Verify returns 2 when artifact does not match live registry."""
        artifact = tmp_path / "fingerprint.json"
        # Write a fake artifact that won't match
        artifact.write_text(
            json.dumps(
                {
                    "version": 1,
                    "fingerprint_sha256": "a" * 64,
                    "elements": [],
                }
            ),
            encoding="utf-8",
        )
        assert cmd_verify(str(artifact)) == 2

    def test_verify_fails_with_invalid_json(self, tmp_path: Path) -> None:
        """Verify returns 2 when artifact contains invalid JSON."""
        artifact = tmp_path / "fingerprint.json"
        artifact.write_text("not valid json {{{", encoding="utf-8")
        assert cmd_verify(str(artifact)) == 2


# ---------------------------------------------------------------------------
# TestCmdStamp
# ---------------------------------------------------------------------------


class TestCmdStamp:
    """Tests for cmd_stamp()."""

    def test_stamp_creates_artifact(self, tmp_path: Path) -> None:
        """Stamp creates a valid artifact file."""
        artifact = tmp_path / "fingerprint.json"
        assert cmd_stamp(str(artifact)) == 0
        assert artifact.exists()

        # Validate it's proper JSON with expected fields
        data = json.loads(artifact.read_text(encoding="utf-8"))
        assert "fingerprint_sha256" in data
        assert "elements" in data
        assert data["version"] == 1

    def test_stamp_then_verify_passes(self, tmp_path: Path) -> None:
        """Stamp followed by verify produces a clean pass."""
        artifact = tmp_path / "fingerprint.json"
        assert cmd_stamp(str(artifact)) == 0
        assert cmd_verify(str(artifact)) == 0

    def test_stamp_dry_run_does_not_write(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Stamp with dry_run=True does not create the file."""
        artifact = tmp_path / "fingerprint.json"
        assert cmd_stamp(str(artifact), dry_run=True) == 0
        assert not artifact.exists()

        captured = capsys.readouterr()
        assert "--dry-run" in captured.out


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() CLI entry point."""

    def test_no_args_returns_1(self) -> None:
        """No subcommand returns exit code 1."""
        assert main([]) == 1

    def test_verify_with_valid_artifact(self, tmp_path: Path) -> None:
        """verify subcommand returns 0 when artifact matches."""
        artifact = tmp_path / "fingerprint.json"
        cmd_stamp(str(artifact))

        result = main(["verify", "--artifact", str(artifact)])
        assert result == 0

    def test_verify_with_missing_artifact(self, tmp_path: Path) -> None:
        """verify subcommand returns 2 when artifact is missing."""
        artifact = tmp_path / "nonexistent.json"
        result = main(["verify", "--artifact", str(artifact)])
        assert result == 2

    def test_stamp_creates_artifact(self, tmp_path: Path) -> None:
        """stamp subcommand creates artifact and returns 0."""
        artifact = tmp_path / "fingerprint.json"
        result = main(["stamp", "--artifact", str(artifact)])
        assert result == 0
        assert artifact.exists()
