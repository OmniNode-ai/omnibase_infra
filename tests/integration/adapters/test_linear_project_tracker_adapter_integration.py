# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for AdapterLinearProjectTracker.

Exercises the adapter end-to-end with fake MCP callables that simulate a
live Linear MCP server. Verifies issue creation → fetch → update →
comment → search flows against a shared in-memory store, asserting dict
→ wire-format translation at every step.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
    AdapterLinearProjectTracker,
)
from omnibase_infra.adapters.project_tracker.model_stub_issue import ModelStubIssue


class _FakeLinearMcp:
    """In-memory simulator for the subset of Linear MCP tools we wrap."""

    def __init__(self) -> None:
        self._issues: dict[str, dict[str, object]] = {}
        self._counter = 0

    def _now(self) -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def create_issue(self, **kwargs: object) -> dict[str, object]:
        self._counter += 1
        issue_id = f"fake-uuid-{self._counter}"
        identifier = f"OMN-{9000 + self._counter}"
        now = self._now()
        row: dict[str, object] = {
            "id": issue_id,
            "identifier": identifier,
            "title": kwargs.get("title", ""),
            "description": kwargs.get("description", ""),
            "state": {"name": "Todo", "type": "unstarted"},
            "priority": None,
            "assignee": None,
            "labels": kwargs.get("labels", []),
            "team": {"id": "t-1", "name": "Omninode"},
            "url": f"https://linear.app/omninode/issue/{identifier}",
            "createdAt": now,
            "updatedAt": now,
        }
        self._issues[issue_id] = row
        return row

    def get_issue(self, **kwargs: object) -> dict[str, object]:
        issue_id = str(kwargs["id"])
        for row in self._issues.values():
            if row["id"] == issue_id or row["identifier"] == issue_id:
                return row
        return {}

    def update_issue(self, **kwargs: object) -> dict[str, object]:
        issue_id = str(kwargs.pop("id"))
        for row in self._issues.values():
            if row["id"] == issue_id or row["identifier"] == issue_id:
                row.update(kwargs)
                row["updatedAt"] = self._now()
                return row
        return {}

    def search_issues(self, **kwargs: object) -> list[dict[str, object]]:
        q = str(kwargs.get("query", "")).lower()
        return [row for row in self._issues.values() if q in str(row["title"]).lower()]

    def add_comment(self, **kwargs: object) -> dict[str, object]:
        issue_id = str(kwargs["issueId"])
        if not any(
            row["id"] == issue_id or row["identifier"] == issue_id
            for row in self._issues.values()
        ):
            return {}
        return {
            "id": f"c-{len(self._issues)}-{self._counter}",
            "body": str(kwargs["body"]),
            "user": {"id": "u-1", "name": "Jonah"},
            "createdAt": self._now(),
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_issue_lifecycle() -> None:
    mcp = _FakeLinearMcp()
    adapter = AdapterLinearProjectTracker(
        mcp_create_issue=mcp.create_issue,
        mcp_get_issue=mcp.get_issue,
        mcp_update_issue=mcp.update_issue,
        mcp_search_issues=mcp.search_issues,
        mcp_add_comment=mcp.add_comment,
    )
    await adapter.connect()

    created = await adapter.create_issue(
        title="Integration test ticket",
        description="body",
        labels=["integration"],
    )
    assert isinstance(created, ModelStubIssue)
    assert created.identifier.startswith("OMN-")
    assert created.state == "Todo"
    assert "integration" in created.labels

    fetched = await adapter.get_issue(created.id)
    assert fetched.id == created.id
    assert fetched.title == "Integration test ticket"

    updated = await adapter.update_issue(created.id, {"title": "Renamed"})
    assert updated.title == "Renamed"
    assert updated.id == created.id

    hits = await adapter.search_issues("renamed")
    assert len(hits) == 1
    assert hits[0].id == created.id

    comment = await adapter.add_comment(created.id, "hello from integration")
    assert comment.body == "hello from integration"
    assert comment.author == "Jonah"

    await adapter.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_issue_missing_raises_key_error() -> None:
    mcp = _FakeLinearMcp()
    adapter = AdapterLinearProjectTracker(mcp_get_issue=mcp.get_issue)
    await adapter.connect()

    with pytest.raises(KeyError):
        await adapter.get_issue("OMN-DOES-NOT-EXIST")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_add_comment_on_missing_issue_raises_key_error() -> None:
    mcp = _FakeLinearMcp()
    adapter = AdapterLinearProjectTracker(mcp_add_comment=mcp.add_comment)
    await adapter.connect()

    with pytest.raises(KeyError):
        await adapter.add_comment("NOPE", "orphan")
