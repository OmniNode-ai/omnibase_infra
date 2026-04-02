# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for session projector logic."""

from datetime import UTC, datetime, timezone

import pytest

from omnibase_infra.services.session_projector.projector import (
    project_session_ended,
    project_session_started,
    project_tool_executed,
)


@pytest.mark.unit
class TestProjectSessionStarted:
    def test_creates_snapshot(self) -> None:
        snapshot = project_session_started(
            agent_id="CAIA",
            session_id="sess-abc",
            terminal_id="terminal-mac-3",
            machine="jonahs-macbook",
            working_directory="/Volumes/PRO-G40/Code/omni_worktrees/OMN-7241/omnibase_infra",
            git_branch="jonah/omn-7241-learning-models",
            started_at=datetime.now(tz=UTC),
        )
        assert snapshot["agent_id"] == "CAIA"
        assert snapshot["working_directory"].endswith("omnibase_infra")
        assert snapshot["git_branch"] == "jonah/omn-7241-learning-models"
        assert snapshot["files_touched"] == []
        assert snapshot["errors_hit"] == []

    def test_extracts_ticket_from_branch(self) -> None:
        snapshot = project_session_started(
            agent_id="CAIA",
            session_id="sess-abc",
            terminal_id=None,
            machine=None,
            working_directory="/Volumes/PRO-G40/Code/test",
            git_branch="jonah/omn-7292-agent-entity-models",
            started_at=datetime.now(tz=UTC),
        )
        assert snapshot["current_ticket"] == "OMN-7292"

    def test_no_ticket_from_main(self) -> None:
        snapshot = project_session_started(
            agent_id="CAIA",
            session_id="sess-abc",
            terminal_id=None,
            machine=None,
            working_directory="/Volumes/PRO-G40/Code/test",
            git_branch="main",
            started_at=datetime.now(tz=UTC),
        )
        assert snapshot["current_ticket"] is None


@pytest.mark.unit
class TestProjectToolExecuted:
    def test_edit_adds_file(self) -> None:
        snapshot: dict[str, object] = {
            "files_touched": [],
            "errors_hit": [],
            "last_tool_name": None,
        }
        updated = project_tool_executed(
            snapshot=snapshot,
            tool_name="Edit",
            success=True,
            summary="Edited src/foo.py:42",
        )
        assert "src/foo.py" in updated["files_touched"]
        assert updated["last_tool_name"] == "Edit"
        assert updated["last_tool_success"] is True

    def test_edit_adds_absolute_path(self) -> None:
        snapshot: dict[str, object] = {
            "files_touched": [],
            "errors_hit": [],
            "last_tool_name": None,
        }
        updated = project_tool_executed(
            snapshot=snapshot,
            tool_name="Edit",
            success=True,
            summary="Edited /Volumes/PRO-G40/Code/omni_worktrees/OMN-7241/omnibase_infra/src/models/agent.py:10",
        )
        files = updated["files_touched"]
        assert any("/Volumes/" in str(f) for f in files)

    def test_failed_bash_adds_error(self) -> None:
        snapshot: dict[str, object] = {
            "files_touched": [],
            "errors_hit": [],
            "last_tool_name": None,
        }
        updated = project_tool_executed(
            snapshot=snapshot,
            tool_name="Bash",
            success=False,
            summary="ModuleNotFoundError: No module named 'foo'",
        )
        assert len(updated["errors_hit"]) == 1
        assert "ModuleNotFoundError" in updated["errors_hit"][0]


@pytest.mark.unit
class TestProjectSessionEnded:
    def test_sets_outcome(self) -> None:
        snapshot: dict[str, object] = {
            "session_ended_at": None,
            "session_outcome": None,
        }
        updated = project_session_ended(
            snapshot=snapshot,
            outcome="success",
            ended_at=datetime.now(tz=UTC),
        )
        assert updated["session_outcome"] == "success"
        assert updated["session_ended_at"] is not None
