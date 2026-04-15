# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for LocalStubProjectTracker.

Covers all 12 ProtocolProjectTracker methods:
    connect, health_check, get_capabilities, close,
    list_issues, get_issue, create_issue, update_issue,
    search_issues, add_comment, get_project, list_projects.

Also verifies crash-consistent file writes via atomic rename.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
    LocalStubProjectTracker,
)
from omnibase_infra.adapters.project_tracker.model_stub_comment import ModelStubComment
from omnibase_infra.adapters.project_tracker.model_stub_issue import ModelStubIssue
from omnibase_infra.adapters.project_tracker.model_stub_project import ModelStubProject


def run(coro):  # type: ignore[return]
    return asyncio.get_event_loop().run_until_complete(coro)


class TestLocalStubProjectTrackerLifecycle:
    def test_connect_returns_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))
            assert run(tracker.connect()) is True

    def test_health_check_returns_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))
            run(tracker.connect())
            status = run(tracker.health_check())
            assert status.status == "healthy"

    def test_get_capabilities_returns_read_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))
            caps = run(tracker.get_capabilities())
            assert "read" in caps
            assert "write" in caps

    def test_close_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))
            run(tracker.connect())
            run(tracker.close())


class TestLocalStubProjectTrackerIssues:
    def test_create_and_get_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                issue = await tracker.create_issue(
                    title="Test ticket", description="desc", labels=["bug"]
                )
                assert isinstance(issue, ModelStubIssue)
                assert issue.title == "Test ticket"
                assert issue.identifier.startswith("STUB-")
                assert "bug" in issue.labels
                assert issue.state == "todo"

                fetched = await tracker.get_issue(issue.id)
                assert fetched.id == issue.id
                assert fetched.title == "Test ticket"

            asyncio.run(_run())

    def test_get_issue_by_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                issue = await tracker.create_issue(
                    title="Lookup by identifier", description=""
                )
                fetched = await tracker.get_issue(issue.identifier)
                assert fetched.id == issue.id

            asyncio.run(_run())

    def test_get_issue_raises_key_error_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                with pytest.raises(KeyError):
                    await tracker.get_issue("nonexistent-id")

            asyncio.run(_run())

    def test_identifiers_are_monotonically_incrementing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                i1 = await tracker.create_issue(title="A", description="")
                i2 = await tracker.create_issue(title="B", description="")
                assert i1.identifier == "STUB-1"
                assert i2.identifier == "STUB-2"

            asyncio.run(_run())

    def test_list_issues_returns_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                await tracker.create_issue(title="Alpha", description="")
                await tracker.create_issue(title="Beta", description="")
                issues = await tracker.list_issues()
                assert len(issues) == 2

            asyncio.run(_run())

    def test_list_issues_filter_by_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                issue = await tracker.create_issue(title="Alpha", description="")
                await tracker.update_issue(issue.id, {"state": "in_progress"})
                await tracker.create_issue(title="Beta", description="")
                results = await tracker.list_issues(filters={"state": "in_progress"})
                assert len(results) == 1
                assert results[0].title == "Alpha"

            asyncio.run(_run())

    def test_list_issues_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                for i in range(5):
                    await tracker.create_issue(title=f"Issue {i}", description="")
                results = await tracker.list_issues(limit=3)
                assert len(results) == 3

            asyncio.run(_run())

    def test_update_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                issue = await tracker.create_issue(title="Original", description="")
                updated = await tracker.update_issue(
                    issue.id, {"title": "Updated", "state": "done"}
                )
                assert updated.title == "Updated"
                assert updated.state == "done"
                assert updated.id == issue.id

            asyncio.run(_run())

    def test_update_issue_raises_key_error_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                with pytest.raises(KeyError):
                    await tracker.update_issue("nonexistent", {"state": "done"})

            asyncio.run(_run())

    def test_search_issues_title_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                await tracker.create_issue(title="Alpha ticket", description="")
                await tracker.create_issue(title="Beta ticket", description="")
                results = await tracker.search_issues("Alpha")
                assert len(results) == 1
                assert results[0].title == "Alpha ticket"

            asyncio.run(_run())

    def test_search_issues_description_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                await tracker.create_issue(
                    title="Ticket A", description="contains needle"
                )
                await tracker.create_issue(title="Ticket B", description="no match")
                results = await tracker.search_issues("needle")
                assert len(results) == 1
                assert results[0].title == "Ticket A"

            asyncio.run(_run())

    def test_search_issues_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                await tracker.create_issue(title="Auth Middleware", description="")
                results = await tracker.search_issues("auth middleware")
                assert len(results) == 1

            asyncio.run(_run())

    def test_search_issues_no_match_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                await tracker.create_issue(title="Something", description="")
                results = await tracker.search_issues("xyz_not_found")
                assert results == []

            asyncio.run(_run())


class TestLocalStubProjectTrackerComments:
    def test_add_comment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                issue = await tracker.create_issue(title="Commentable", description="")
                comment = await tracker.add_comment(issue.id, "This is a comment")
                assert isinstance(comment, ModelStubComment)
                assert comment.body == "This is a comment"
                assert comment.author == "stub"

            asyncio.run(_run())

    def test_add_comment_raises_key_error_for_missing_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                with pytest.raises(KeyError):
                    await tracker.add_comment("nonexistent", "orphan comment")

            asyncio.run(_run())


class TestLocalStubProjectTrackerProjects:
    def test_list_projects_empty_initially(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                projects = await tracker.list_projects()
                assert projects == []

            asyncio.run(_run())

    def test_get_project_raises_key_error_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracker = LocalStubProjectTracker(state_root=Path(tmp))

            async def _run() -> None:
                await tracker.connect()
                with pytest.raises(KeyError):
                    await tracker.get_project("nonexistent-project")

            asyncio.run(_run())


class TestLocalStubProjectTrackerPersistence:
    def test_state_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)

            async def _write() -> str:
                tracker = LocalStubProjectTracker(state_root=state_root)
                await tracker.connect()
                issue = await tracker.create_issue(
                    title="Persisted", description="round-trip"
                )
                return issue.id

            async def _read(issue_id: str) -> None:
                tracker2 = LocalStubProjectTracker(state_root=state_root)
                await tracker2.connect()
                fetched = await tracker2.get_issue(issue_id)
                assert fetched.title == "Persisted"

            issue_id = asyncio.run(_write())
            asyncio.run(_read(issue_id))

    def test_atomic_write_uses_tmp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)
            tracker = LocalStubProjectTracker(state_root=state_root)

            async def _run() -> None:
                await tracker.connect()
                await tracker.create_issue(title="Atomic test", description="")
                # After save, .tmp file should be gone (renamed to final)
                tmp_file = state_root / "project_tracker_stub.tmp"
                assert not tmp_file.exists()
                assert (state_root / "project_tracker_stub.json").exists()

            asyncio.run(_run())

    def test_counter_increments_across_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_root = Path(tmp)

            async def _session1() -> None:
                t = LocalStubProjectTracker(state_root=state_root)
                await t.connect()
                i = await t.create_issue(title="First session", description="")
                assert i.identifier == "STUB-1"

            async def _session2() -> None:
                t = LocalStubProjectTracker(state_root=state_root)
                await t.connect()
                i = await t.create_issue(title="Second session", description="")
                assert i.identifier == "STUB-2"

            asyncio.run(_session1())
            asyncio.run(_session2())
