# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for agent learning extraction consumer."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.models.agent_learning import EnumLearningTaskType
from omnibase_infra.services.agent_learning_extraction.consumer import (
    build_learning_record,
    classify_task_type,
    extract_error_signatures,
    extract_repo_from_working_dir,
    extract_ticket_from_branch,
)


@pytest.mark.unit
class TestExtractRepoFromWorkingDir:
    def test_omni_home_repo(self) -> None:
        assert (
            extract_repo_from_working_dir(
                "/Volumes/PRO-G40/Code/omni_home/omnibase_infra"
            )
            == "omnibase_infra"
        )

    def test_worktree_repo(self) -> None:
        assert (
            extract_repo_from_working_dir(
                "/Volumes/PRO-G40/Code/omni_worktrees/OMN-7100/omniclaude"
            )
            == "omniclaude"
        )

    def test_unknown_path(self) -> None:
        assert extract_repo_from_working_dir("/tmp/random") == "unknown"  # noqa: S108


@pytest.mark.unit
class TestExtractTicketFromBranch:
    def test_standard_branch(self) -> None:
        assert extract_ticket_from_branch("jonah/omn-7100-fix-ci") == "OMN-7100"

    def test_no_ticket(self) -> None:
        assert extract_ticket_from_branch("main") is None

    def test_worktree_branch(self) -> None:
        assert extract_ticket_from_branch("worktree-agent-abc123") is None


@pytest.mark.unit
class TestClassifyTaskType:
    def test_ci_fix(self) -> None:
        assert (
            classify_task_type(
                branch="jonah/omn-7100-fix-ci",
                file_paths=["pyproject.toml", ".github/workflows/ci.yml"],
            )
            == EnumLearningTaskType.CI_FIX
        )

    def test_migration(self) -> None:
        assert (
            classify_task_type(
                branch="jonah/omn-7100-add-migration",
                file_paths=["docker/migrations/forward/057_foo.sql"],
            )
            == EnumLearningTaskType.MIGRATION
        )

    def test_feature_default(self) -> None:
        assert (
            classify_task_type(
                branch="jonah/omn-7100-add-dashboard",
                file_paths=["src/app/page.tsx"],
            )
            == EnumLearningTaskType.FEATURE
        )


@pytest.mark.unit
class TestExtractErrorSignatures:
    def test_extracts_from_failed_tools(self) -> None:
        tool_events = [
            {
                "tool_name": "Bash",
                "success": False,
                "summary": "ModuleNotFoundError: No module named 'foo'",
            },
            {"tool_name": "Bash", "success": True, "summary": "Tests passed"},
            {
                "tool_name": "Bash",
                "success": False,
                "summary": "ruff check failed: E501 line too long",
            },
        ]
        errors = extract_error_signatures(tool_events)
        assert len(errors) == 2
        assert "ModuleNotFoundError: No module named 'foo'" in errors

    def test_empty_on_all_success(self) -> None:
        tool_events = [
            {"tool_name": "Bash", "success": True, "summary": "ok"},
        ]
        assert extract_error_signatures(tool_events) == []


@pytest.mark.unit
class TestBuildLearningRecord:
    def test_builds_record(self) -> None:
        record = build_learning_record(
            session_id=uuid4(),
            working_dir="/Volumes/PRO-G40/Code/omni_home/omnibase_infra",
            branch="jonah/omn-7100-fix-ci",
            resolution_summary="Fixed CI by adding --extend-exclude to pyproject.toml.",
            file_paths=["pyproject.toml"],
            error_signatures=["ruff check failed"],
            created_at=datetime.now(tz=UTC),
        )
        assert record.repo == "omnibase_infra"
        assert record.ticket_id == "OMN-7100"
        assert record.task_type == EnumLearningTaskType.CI_FIX
