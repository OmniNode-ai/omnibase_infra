# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for AdapterProjectTrackerLinear — OMN-8773 methods.

Tests list_teams, list_issue_labels, list_issue_statuses using callable
injection (no live MCP server required).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
    AdapterProjectTrackerLinear,
)

_FAKE_TEAM = {"id": "t-1", "name": "Omninode", "key": "OMN"}
_FAKE_LABEL = {"id": "l-1", "name": "bug", "color": "#ff0000"}
_FAKE_STATUS = {
    "id": "s-1",
    "name": "In Progress",
    "type": "started",
    "color": "#0000ff",
}


@pytest.mark.unit
class TestAdapterProjectTrackerLinearListTeams:
    def test_list_teams_delegates_to_mcp(self) -> None:
        fake = MagicMock(return_value=[_FAKE_TEAM, {"id": "t-2", "name": "Platform"}])
        adapter = AdapterProjectTrackerLinear(mcp_list_teams=fake)

        async def _run() -> None:
            results = await adapter.list_teams()
            assert len(results) == 2
            assert results[0]["id"] == "t-1"
            assert results[0]["name"] == "Omninode"
            fake.assert_called_once_with()

        asyncio.run(_run())

    def test_list_teams_non_list_returns_empty(self) -> None:
        fake = MagicMock(return_value=None)
        adapter = AdapterProjectTrackerLinear(mcp_list_teams=fake)
        assert asyncio.run(adapter.list_teams()) == []

    def test_list_teams_without_callable_raises(self) -> None:
        adapter = AdapterProjectTrackerLinear()

        async def _run() -> None:
            with pytest.raises(NotImplementedError):
                await adapter.list_teams()

        asyncio.run(_run())


@pytest.mark.unit
class TestAdapterProjectTrackerLinearListIssueLabels:
    def test_list_issue_labels_delegates_to_mcp(self) -> None:
        fake = MagicMock(return_value=[_FAKE_LABEL])
        adapter = AdapterProjectTrackerLinear(mcp_list_issue_labels=fake)

        async def _run() -> None:
            results = await adapter.list_issue_labels(team="t-1")
            assert len(results) == 1
            assert results[0]["name"] == "bug"
            fake.assert_called_once_with(teamId="t-1")

        asyncio.run(_run())

    def test_list_issue_labels_non_list_returns_empty(self) -> None:
        fake = MagicMock(return_value=None)
        adapter = AdapterProjectTrackerLinear(mcp_list_issue_labels=fake)
        assert asyncio.run(adapter.list_issue_labels(team="t-1")) == []

    def test_list_issue_labels_without_callable_raises(self) -> None:
        adapter = AdapterProjectTrackerLinear()

        async def _run() -> None:
            with pytest.raises(NotImplementedError):
                await adapter.list_issue_labels(team="t-1")

        asyncio.run(_run())


@pytest.mark.unit
class TestAdapterProjectTrackerLinearListIssueStatuses:
    def test_list_issue_statuses_delegates_to_mcp(self) -> None:
        fake = MagicMock(return_value=[_FAKE_STATUS])
        adapter = AdapterProjectTrackerLinear(mcp_list_issue_statuses=fake)

        async def _run() -> None:
            results = await adapter.list_issue_statuses(team="t-1")
            assert len(results) == 1
            assert results[0]["name"] == "In Progress"
            assert results[0]["type"] == "started"
            fake.assert_called_once_with(teamId="t-1")

        asyncio.run(_run())

    def test_list_issue_statuses_non_list_returns_empty(self) -> None:
        fake = MagicMock(return_value=None)
        adapter = AdapterProjectTrackerLinear(mcp_list_issue_statuses=fake)
        assert asyncio.run(adapter.list_issue_statuses(team="t-1")) == []

    def test_list_issue_statuses_without_callable_raises(self) -> None:
        adapter = AdapterProjectTrackerLinear()

        async def _run() -> None:
            with pytest.raises(NotImplementedError):
                await adapter.list_issue_statuses(team="t-1")

        asyncio.run(_run())
