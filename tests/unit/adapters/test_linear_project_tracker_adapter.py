# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for LinearProjectTrackerAdapter.

Uses callable injection — tests pass fake callables as constructor args
so no live MCP server is required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
    LinearProjectTrackerAdapter,
)
from omnibase_infra.adapters.project_tracker.model_stub_comment import ModelStubComment
from omnibase_infra.adapters.project_tracker.model_stub_issue import ModelStubIssue
from omnibase_infra.adapters.project_tracker.model_stub_project import ModelStubProject

_FAKE_ISSUE = {
    "id": "abc-123",
    "identifier": "OMN-9999",
    "title": "Test Issue",
    "description": "body",
    "state": {"name": "In Progress", "type": "started"},
    "priority": {"name": "High"},
    "assignee": {"id": "u-1", "name": "Jonah"},
    "labels": ["bug", "urgent"],
    "team": {"id": "t-1", "name": "Omninode"},
    "url": "https://linear.app/test",
    "createdAt": "2026-01-01T00:00:00Z",
    "updatedAt": "2026-01-02T00:00:00Z",
}


class TestLinearProjectTrackerAdapterLifecycle:
    def test_connect_returns_true(self) -> None:
        adapter = LinearProjectTrackerAdapter()
        assert asyncio.run(adapter.connect()) is True

    def test_health_check_healthy_after_connect(self) -> None:
        adapter = LinearProjectTrackerAdapter()

        async def _run() -> None:
            await adapter.connect()
            status = await adapter.health_check()
            assert status.status == "healthy"
            assert status.service_id == "linear-project-tracker"

        asyncio.run(_run())

    def test_capabilities_read_write(self) -> None:
        adapter = LinearProjectTrackerAdapter()
        caps = asyncio.run(adapter.get_capabilities())
        assert "read" in caps and "write" in caps

    def test_close_resets_connected(self) -> None:
        adapter = LinearProjectTrackerAdapter()

        async def _run() -> None:
            await adapter.connect()
            await adapter.close()
            status = await adapter.health_check()
            assert status.status == "not_connected"

        asyncio.run(_run())


class TestLinearProjectTrackerAdapterIssues:
    def test_get_issue_delegates_to_mcp(self) -> None:
        fake = MagicMock(return_value=_FAKE_ISSUE)
        adapter = LinearProjectTrackerAdapter(mcp_get_issue=fake)

        async def _run() -> None:
            await adapter.connect()
            issue = await adapter.get_issue("OMN-9999")
            assert isinstance(issue, ModelStubIssue)
            assert issue.identifier == "OMN-9999"
            assert issue.id == "abc-123"
            assert issue.title == "Test Issue"
            assert issue.state == "In Progress"
            assert issue.priority == "High"
            assert issue.assignee == "Jonah"
            assert issue.team == "Omninode"
            assert "bug" in issue.labels
            fake.assert_called_once_with(id="OMN-9999")

        asyncio.run(_run())

    def test_get_issue_empty_result_raises_key_error(self) -> None:
        fake = MagicMock(return_value={})
        adapter = LinearProjectTrackerAdapter(mcp_get_issue=fake)

        async def _run() -> None:
            with pytest.raises(KeyError):
                await adapter.get_issue("NOPE-1")

        asyncio.run(_run())

    def test_get_issue_without_callable_raises(self) -> None:
        adapter = LinearProjectTrackerAdapter()

        async def _run() -> None:
            with pytest.raises(NotImplementedError):
                await adapter.get_issue("OMN-1")

        asyncio.run(_run())

    def test_list_issues_with_filters_and_limit(self) -> None:
        fake = MagicMock(return_value=[_FAKE_ISSUE, _FAKE_ISSUE])
        adapter = LinearProjectTrackerAdapter(mcp_list_issues=fake)

        async def _run() -> None:
            results = await adapter.list_issues(filters={"state": "todo"}, limit=10)
            assert len(results) == 2
            assert all(isinstance(r, ModelStubIssue) for r in results)
            fake.assert_called_once_with(limit=10, state="todo")

        asyncio.run(_run())

    def test_list_issues_non_list_returns_empty(self) -> None:
        fake = MagicMock(return_value=None)
        adapter = LinearProjectTrackerAdapter(mcp_list_issues=fake)
        assert asyncio.run(adapter.list_issues()) == []

    def test_create_issue_forwards_kwargs(self) -> None:
        fake = MagicMock(return_value=_FAKE_ISSUE)
        adapter = LinearProjectTrackerAdapter(mcp_create_issue=fake)

        async def _run() -> None:
            issue = await adapter.create_issue(
                title="New",
                description="body",
                labels=["a"],
                assignee="u-1",
                priority="High",
                team="t-1",
            )
            assert isinstance(issue, ModelStubIssue)
            fake.assert_called_once_with(
                title="New",
                description="body",
                labels=["a"],
                assignee="u-1",
                priority="High",
                team="t-1",
            )

        asyncio.run(_run())

    def test_create_issue_omits_none_kwargs(self) -> None:
        fake = MagicMock(return_value=_FAKE_ISSUE)
        adapter = LinearProjectTrackerAdapter(mcp_create_issue=fake)

        async def _run() -> None:
            await adapter.create_issue(title="T", description="D")
            fake.assert_called_once_with(title="T", description="D")

        asyncio.run(_run())

    def test_update_issue_delegates(self) -> None:
        fake = MagicMock(return_value=_FAKE_ISSUE)
        adapter = LinearProjectTrackerAdapter(mcp_update_issue=fake)

        async def _run() -> None:
            result = await adapter.update_issue("OMN-9999", {"state": "done"})
            assert isinstance(result, ModelStubIssue)
            fake.assert_called_once_with(id="OMN-9999", state="done")

        asyncio.run(_run())

    def test_update_issue_missing_raises_key_error(self) -> None:
        fake = MagicMock(return_value={})
        adapter = LinearProjectTrackerAdapter(mcp_update_issue=fake)

        async def _run() -> None:
            with pytest.raises(KeyError):
                await adapter.update_issue("NOPE", {"state": "done"})

        asyncio.run(_run())

    def test_search_issues_delegates(self) -> None:
        fake = MagicMock(return_value=[_FAKE_ISSUE])
        adapter = LinearProjectTrackerAdapter(mcp_search_issues=fake)

        async def _run() -> None:
            results = await adapter.search_issues("auth middleware", limit=5)
            assert len(results) == 1
            assert results[0].identifier == "OMN-9999"
            fake.assert_called_once_with(query="auth middleware", limit=5)

        asyncio.run(_run())


class TestLinearProjectTrackerAdapterComments:
    def test_add_comment_delegates(self) -> None:
        fake = MagicMock(
            return_value={
                "id": "c-1",
                "body": "hello",
                "user": {"id": "u-1", "name": "Jonah"},
                "createdAt": "2026-01-01T00:00:00Z",
            }
        )
        adapter = LinearProjectTrackerAdapter(mcp_add_comment=fake)

        async def _run() -> None:
            comment = await adapter.add_comment("abc-123", "hello")
            assert isinstance(comment, ModelStubComment)
            assert comment.body == "hello"
            assert comment.author == "Jonah"
            fake.assert_called_once_with(issueId="abc-123", body="hello")

        asyncio.run(_run())

    def test_add_comment_missing_issue_raises_key_error(self) -> None:
        fake = MagicMock(return_value={})
        adapter = LinearProjectTrackerAdapter(mcp_add_comment=fake)

        async def _run() -> None:
            with pytest.raises(KeyError):
                await adapter.add_comment("NOPE", "text")

        asyncio.run(_run())


class TestLinearProjectTrackerAdapterProjects:
    def test_get_project_delegates(self) -> None:
        fake = MagicMock(
            return_value={
                "id": "p-1",
                "name": "Onboarding",
                "description": "desc",
                "state": {"name": "active"},
                "progress": 0.42,
                "url": "https://linear.app/p/1",
            }
        )
        adapter = LinearProjectTrackerAdapter(mcp_get_project=fake)

        async def _run() -> None:
            project = await adapter.get_project("p-1")
            assert isinstance(project, ModelStubProject)
            assert project.name == "Onboarding"
            assert project.state == "active"
            assert project.progress == pytest.approx(0.42)
            fake.assert_called_once_with(id="p-1")

        asyncio.run(_run())

    def test_get_project_missing_raises_key_error(self) -> None:
        fake = MagicMock(return_value={})
        adapter = LinearProjectTrackerAdapter(mcp_get_project=fake)

        async def _run() -> None:
            with pytest.raises(KeyError):
                await adapter.get_project("NOPE")

        asyncio.run(_run())

    def test_list_projects_delegates(self) -> None:
        fake = MagicMock(
            return_value=[
                {"id": "p-1", "name": "A"},
                {"id": "p-2", "name": "B", "progress": 1.0},
            ]
        )
        adapter = LinearProjectTrackerAdapter(mcp_list_projects=fake)

        async def _run() -> None:
            results = await adapter.list_projects(limit=25)
            assert len(results) == 2
            assert results[0].name == "A"
            assert results[1].progress == pytest.approx(1.0)
            fake.assert_called_once_with(limit=25)

        asyncio.run(_run())
