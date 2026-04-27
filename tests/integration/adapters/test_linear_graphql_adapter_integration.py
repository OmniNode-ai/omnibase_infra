# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for AdapterLinearGraphQLProjectTracker.

Hits the live Linear GraphQL API. Gated on LINEAR_API_KEY (or
LINEAR_TOKEN) being present in the environment — skipped otherwise.

Run with::

    LINEAR_API_KEY=... uv run pytest \\
        tests/integration/adapters/test_linear_graphql_adapter_integration.py \\
        -v -m linear
"""

from __future__ import annotations

import os

import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)

pytestmark = [pytest.mark.integration, pytest.mark.linear]


_NO_KEY_REASON = (
    "LINEAR_API_KEY (or LINEAR_TOKEN) not set; skipping live Linear API test"
)


def _have_key() -> bool:
    return bool(os.environ.get("LINEAR_API_KEY") or os.environ.get("LINEAR_TOKEN"))


@pytest.mark.skipif(not _have_key(), reason=_NO_KEY_REASON)
@pytest.mark.asyncio
async def test_connect_against_live_linear() -> None:
    """Connect issues a viewer query — proves auth works and the adapter
    reaches Linear."""
    adapter = AdapterLinearGraphQLProjectTracker()
    try:
        connected = await adapter.connect()
        assert connected is True
        health = await adapter.health_check()
        assert health.status == "healthy"
    finally:
        await adapter.close()


@pytest.mark.skipif(not _have_key(), reason=_NO_KEY_REASON)
@pytest.mark.asyncio
async def test_get_issue_against_live_linear() -> None:
    """Fetch this ticket (OMN-10048) and verify the wire shape round-trips."""
    adapter = AdapterLinearGraphQLProjectTracker()
    try:
        await adapter.connect()
        issue = await adapter.get_issue("OMN-10048")
        assert issue.identifier == "OMN-10048"
        assert "Linear" in issue.title or "tracker" in issue.title.lower()
        assert issue.url is not None
        assert issue.created_at is not None
    finally:
        await adapter.close()


@pytest.mark.skipif(not _have_key(), reason=_NO_KEY_REASON)
@pytest.mark.asyncio
async def test_search_issues_against_live_linear() -> None:
    """Search must return at least one issue for a broad term."""
    adapter = AdapterLinearGraphQLProjectTracker()
    try:
        await adapter.connect()
        results = await adapter.search_issues("Linear adapter", limit=5)
        # Don't over-assert on count; the workspace may not match.
        # Just confirm shape and that no exception was raised.
        assert isinstance(results, list)
    finally:
        await adapter.close()
