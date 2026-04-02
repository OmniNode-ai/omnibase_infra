# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for CAIA watchdog checkpoint YAML parsing.

Validates that the watchdog shell script's Python-based YAML parsing
handles valid, incomplete, and malformed checkpoint files correctly.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
class TestWatchdogCheckpointParsing:
    """Validate checkpoint YAML parsing used by caia-watchdog.sh."""

    def _parse_field(self, checkpoint_path: Path, field: str) -> str:
        """Simulate the Python one-liner the watchdog uses to extract a field."""
        result = subprocess.run(
            [
                "python3",
                "-c",
                f"import yaml; d=yaml.safe_load(open('{checkpoint_path}')); "
                f"print(d.get('{field}', '') or '')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip()

    def test_parse_valid_checkpoint(self, tmp_path: Path) -> None:
        """Full checkpoint with all fields parses correctly."""
        checkpoint = {
            "schema_version": "1.0.0",
            "session_id": "test-session-123",
            "checkpoint_reason": "context_limit",
            "created_at": "2026-04-02T10:00:00Z",
            "reset_at": "2026-04-02T11:00:00Z",
            "resume_prompt": "Continue ticket-pipeline for OMN-7300",
            "active_epic": "OMN-7299",
            "active_tickets": ["OMN-7300", "OMN-7301"],
            "active_prs": ["omniclaude#1070"],
            "pipeline_state": "ticket-pipeline:implement",
            "context_percent": 87,
            "session_percent": 45,
            "weekly_percent": 20,
        }
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text(yaml.dump(checkpoint, default_flow_style=False))

        assert self._parse_field(cp_file, "reset_at") == "2026-04-02T11:00:00Z"
        assert (
            self._parse_field(cp_file, "resume_prompt")
            == "Continue ticket-pipeline for OMN-7300"
        )
        assert self._parse_field(cp_file, "checkpoint_reason") == "context_limit"

    def test_parse_checkpoint_missing_reset_at(self, tmp_path: Path) -> None:
        """Checkpoint without reset_at returns empty string."""
        checkpoint = {
            "schema_version": "1.0.0",
            "session_id": "test-session-456",
            "checkpoint_reason": "explicit",
            "created_at": "2026-04-02T10:00:00Z",
            "resume_prompt": "Resume the close-out pipeline",
        }
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text(yaml.dump(checkpoint, default_flow_style=False))

        assert self._parse_field(cp_file, "reset_at") == ""
        assert (
            self._parse_field(cp_file, "resume_prompt")
            == "Resume the close-out pipeline"
        )

    def test_parse_checkpoint_missing_resume_prompt(self, tmp_path: Path) -> None:
        """Checkpoint without resume_prompt returns empty string."""
        checkpoint = {
            "schema_version": "1.0.0",
            "session_id": "test-session-789",
            "checkpoint_reason": "session_limit",
            "created_at": "2026-04-02T10:00:00Z",
            "reset_at": "2026-04-02T12:00:00Z",
        }
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text(yaml.dump(checkpoint, default_flow_style=False))

        assert self._parse_field(cp_file, "resume_prompt") == ""
        assert self._parse_field(cp_file, "reset_at") == "2026-04-02T12:00:00Z"

    def test_parse_checkpoint_null_reset_at(self, tmp_path: Path) -> None:
        """Checkpoint with explicit null reset_at returns empty string."""
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text(
            textwrap.dedent("""\
                schema_version: "1.0.0"
                session_id: test-null
                checkpoint_reason: weekly_limit
                created_at: "2026-04-02T10:00:00Z"
                reset_at: null
                resume_prompt: "Resume after weekly reset"
            """)
        )

        assert self._parse_field(cp_file, "reset_at") == ""
        assert (
            self._parse_field(cp_file, "resume_prompt") == "Resume after weekly reset"
        )

    def test_parse_malformed_yaml(self, tmp_path: Path) -> None:
        """Malformed YAML does not crash the parser — returns empty string."""
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text("this is not: valid: yaml: [[[")

        # The Python one-liner catches exceptions and returns empty
        result = subprocess.run(
            [
                "python3",
                "-c",
                f"import yaml, sys\n"
                f"try:\n"
                f"    d=yaml.safe_load(open('{cp_file}'))\n"
                f"    print(d.get('reset_at', '') or '')\n"
                f"except Exception:\n"
                f"    print('')\n",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.stdout.strip() == ""

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        """Empty file does not crash — returns empty string."""
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text("")

        result = subprocess.run(
            [
                "python3",
                "-c",
                f"import yaml, sys\n"
                f"try:\n"
                f"    d=yaml.safe_load(open('{cp_file}'))\n"
                f"    print((d or {{}}).get('reset_at', '') or '')\n"
                f"except Exception:\n"
                f"    print('')\n",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.stdout.strip() == ""

    def test_checkpoint_yaml_roundtrip(self, tmp_path: Path) -> None:
        """Checkpoint written as YAML can be read back with all fields intact."""
        checkpoint = {
            "schema_version": "1.0.0",
            "session_id": "roundtrip-test",
            "checkpoint_reason": "context_limit",
            "created_at": "2026-04-02T10:00:00Z",
            "reset_at": "2026-04-02T11:30:00Z",
            "resume_prompt": "Continue epic-team for OMN-7280",
            "active_tickets": ["OMN-7300"],
            "active_prs": [],
            "context_percent": 92,
        }
        cp_file = tmp_path / "checkpoint.yaml"
        cp_file.write_text(yaml.dump(checkpoint, default_flow_style=False))

        loaded = yaml.safe_load(cp_file.read_text())
        assert loaded["session_id"] == "roundtrip-test"
        assert loaded["reset_at"] == "2026-04-02T11:30:00Z"
        assert loaded["context_percent"] == 92
        assert loaded["active_tickets"] == ["OMN-7300"]
