# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for AdapterProjectTrackerLinear (OMN-12193 migration target).

Covers list_teams, list_issue_labels, list_issue_statuses, and error-mapping
paths. Uses httpx.MockTransport so no real network calls are made.
"""

from __future__ import annotations

import json

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.adapter_project_tracker_linear import (
    AdapterProjectTrackerLinear,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_issue_status import (
    ModelProjectTrackerIssueStatus,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_label import (
    ModelProjectTrackerLabel,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_team import (
    ModelProjectTrackerTeam,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
)

pytestmark = pytest.mark.unit

# ---------- fixtures ----------

_TEAMS_RESPONSE: dict[str, object] = {
    "data": {
        "teams": {
            "nodes": [
                {"id": "t1", "name": "Engineering", "key": "ENG"},
                {"id": "t2", "name": "Product", "key": "PROD"},
            ]
        }
    }
}

_LABELS_RESPONSE: dict[str, object] = {
    "data": {
        "issueLabels": {
            "nodes": [
                {"id": "l1", "name": "bug", "color": "#ff0000", "team": {"id": "t1"}},
                {
                    "id": "l2",
                    "name": "feature",
                    "color": "#00ff00",
                    "team": {"id": "t1"},
                },
            ]
        }
    }
}

_STATUSES_RESPONSE: dict[str, object] = {
    "data": {
        "workflowStates": {
            "nodes": [
                {
                    "id": "s1",
                    "name": "Backlog",
                    "type": "unstarted",
                    "team": {"id": "t1"},
                },
                {
                    "id": "s2",
                    "name": "In Progress",
                    "type": "started",
                    "team": {"id": "t1"},
                },
                {"id": "s3", "name": "Done", "type": "completed", "team": {"id": "t1"}},
            ]
        }
    }
}


def _json_transport(body: object, status: int = 200) -> httpx.MockTransport:
    raw = json.dumps(body).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=raw, headers={"content-type": "application/json"}
        )

    return httpx.MockTransport(handler)


def _build_adapter(
    transport: httpx.MockTransport,
    api_key: str = "test-key",
) -> AdapterProjectTrackerLinear:
    client = httpx.AsyncClient(transport=transport)
    adapter = AdapterProjectTrackerLinear(api_key=api_key, client=client)
    adapter._owns_client = True
    return adapter


# ---------- construction ----------


@pytest.mark.asyncio
async def test_raises_on_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINEAR_API_KEY", raising=False)
    monkeypatch.delenv("LINEAR_TOKEN", raising=False)
    with pytest.raises(InfraAuthenticationError):
        AdapterProjectTrackerLinear(api_key=None)


@pytest.mark.asyncio
async def test_custom_endpoint_stored() -> None:
    adapter = AdapterProjectTrackerLinear(
        api_key="key", endpoint="http://localhost:8080"
    )
    assert adapter._endpoint == "http://localhost:8080"
    await adapter.close()


# ---------- list_teams ----------


@pytest.mark.asyncio
async def test_list_teams_returns_models() -> None:
    adapter = _build_adapter(_json_transport(_TEAMS_RESPONSE))
    teams = await adapter.list_teams()
    assert teams == [
        ModelProjectTrackerTeam(id="t1", name="Engineering", key="ENG"),
        ModelProjectTrackerTeam(id="t2", name="Product", key="PROD"),
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_list_teams_empty() -> None:
    adapter = _build_adapter(_json_transport({"data": {"teams": {"nodes": []}}}))
    assert await adapter.list_teams() == []
    await adapter.close()


@pytest.mark.asyncio
async def test_list_teams_sends_auth_header() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=json.dumps(_TEAMS_RESPONSE).encode(),
            headers={"content-type": "application/json"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AdapterProjectTrackerLinear(api_key="lin_api_testkey", client=client)
    adapter._owns_client = True
    await adapter.list_teams()
    await adapter.close()
    assert captured[0].headers.get("authorization") == "lin_api_testkey"


# ---------- list_issue_labels ----------


@pytest.mark.asyncio
async def test_list_issue_labels_returns_models() -> None:
    adapter = _build_adapter(_json_transport(_LABELS_RESPONSE))
    labels = await adapter.list_issue_labels("ENG")
    assert labels == [
        ModelProjectTrackerLabel(id="l1", name="bug", color="#ff0000", team_id="t1"),
        ModelProjectTrackerLabel(
            id="l2", name="feature", color="#00ff00", team_id="t1"
        ),
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_list_issue_labels_sends_team_filter() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=json.dumps(_LABELS_RESPONSE).encode(),
            headers={"content-type": "application/json"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AdapterProjectTrackerLinear(api_key="key", client=client)
    adapter._owns_client = True
    await adapter.list_issue_labels("ENG")
    await adapter.close()
    body = json.loads(captured[0].content)
    assert body["variables"]["filter"]["team"]["key"]["eq"] == "ENG"


# ---------- list_issue_statuses ----------


@pytest.mark.asyncio
async def test_list_issue_statuses_returns_models() -> None:
    adapter = _build_adapter(_json_transport(_STATUSES_RESPONSE))
    statuses = await adapter.list_issue_statuses("ENG")
    assert statuses == [
        ModelProjectTrackerIssueStatus(
            id="s1", name="Backlog", type="unstarted", team_id="t1"
        ),
        ModelProjectTrackerIssueStatus(
            id="s2", name="In Progress", type="started", team_id="t1"
        ),
        ModelProjectTrackerIssueStatus(
            id="s3", name="Done", type="completed", team_id="t1"
        ),
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_list_issue_statuses_sends_team_filter() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=json.dumps(_STATUSES_RESPONSE).encode(),
            headers={"content-type": "application/json"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AdapterProjectTrackerLinear(api_key="key", client=client)
    adapter._owns_client = True
    await adapter.list_issue_statuses("ENG")
    await adapter.close()
    body = json.loads(captured[0].content)
    assert body["variables"]["filter"]["team"]["key"]["eq"] == "ENG"


# ---------- error mapping ----------


@pytest.mark.asyncio
async def test_401_raises_auth_error() -> None:
    adapter = _build_adapter(_json_transport({}, status=401))
    with pytest.raises(InfraAuthenticationError):
        await adapter.list_teams()
    await adapter.close()


@pytest.mark.asyncio
async def test_403_raises_auth_error() -> None:
    adapter = _build_adapter(_json_transport({}, status=403))
    with pytest.raises(InfraAuthenticationError):
        await adapter.list_teams()
    await adapter.close()


@pytest.mark.asyncio
async def test_500_raises_connection_error() -> None:
    adapter = _build_adapter(_json_transport({}, status=500))
    with pytest.raises(InfraConnectionError):
        await adapter.list_teams()
    await adapter.close()


@pytest.mark.asyncio
async def test_graphql_errors_raises_connection_error() -> None:
    body = {"errors": [{"message": "bad token"}], "data": None}
    adapter = _build_adapter(_json_transport(body))
    with pytest.raises(InfraConnectionError, match="Linear GraphQL error"):
        await adapter.list_teams()
    await adapter.close()


@pytest.mark.asyncio
async def test_timeout_raises_infra_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AdapterProjectTrackerLinear(api_key="key", client=client)
    adapter._owns_client = True
    with pytest.raises(InfraTimeoutError):
        await adapter.list_teams()
    await adapter.close()


# ---------- model invariants ----------


def test_models_are_frozen() -> None:
    team = ModelProjectTrackerTeam(id="t1", name="Eng", key="ENG")
    with pytest.raises(Exception):
        team.name = "changed"  # type: ignore[misc]

    label = ModelProjectTrackerLabel(id="l1", name="bug", color="#f00", team_id="t1")
    with pytest.raises(Exception):
        label.name = "changed"  # type: ignore[misc]

    status = ModelProjectTrackerIssueStatus(
        id="s1", name="Backlog", type="unstarted", team_id="t1"
    )
    with pytest.raises(Exception):
        status.name = "changed"  # type: ignore[misc]
