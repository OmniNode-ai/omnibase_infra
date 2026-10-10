# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17678: the shared Linear GraphQL transport, driven over a real HTTP client."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
    _issue_from_graphql,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "ci"
    / "fixtures"
    / "linear_graphql_issue_omn17678.json"
)


@pytest.mark.asyncio
async def test_recorded_issue_response_round_trips_through_shared_transport() -> None:
    recorded = json.loads(_FIXTURE.read_text())
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=recorded["response"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with AdapterLinearGraphQLProjectTracker.graphql_transport(
            api_key="test-key", client=client
        ) as transport:
            response = await transport.post_graphql(
                recorded["_provenance"]["query"], recorded["_provenance"]["variables"]
            )

    assert len(seen) == 1
    assert response.status_code == 200
    issue = _issue_from_graphql(response.json()["data"]["issue"])
    assert issue.identifier == "OMN-17678"
