# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for AdapterLinearGraphQLProjectTracker.

Covers every ProtocolProjectTracker method and the error-mapping paths
(timeout / 401 / 429 / 5xx / GraphQL errors). Uses ``httpx.MockTransport``
so no real network calls are made.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraRateLimitedError,
    InfraTimeoutError,
)

pytestmark = pytest.mark.unit


# ---------- helpers ----------


def _ok(data: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json={"data": data})


def _err(messages: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"errors": [{"message": m} for m in messages], "data": None},
    )


def _build_adapter(
    handler: httpx.MockTransport,
    api_key: str = "test-key",
) -> AdapterLinearGraphQLProjectTracker:
    client = httpx.AsyncClient(
        transport=handler,
        headers={"Authorization": api_key, "Content-Type": "application/json"},
    )
    return AdapterLinearGraphQLProjectTracker(api_key=api_key, client=client)


_ISSUE_FIXTURE: dict[str, object] = {
    "id": "uuid-1",
    "identifier": "OMN-1",
    "title": "Test issue",
    "description": "body",
    "priority": 2,
    "url": "https://linear.app/omninode/issue/OMN-1",
    "createdAt": "2026-04-27T00:00:00.000Z",
    "updatedAt": "2026-04-27T01:00:00.000Z",
    "state": {"name": "Backlog", "type": "backlog"},
    "assignee": {"id": "u1", "name": "Tester"},
    "labels": {"nodes": [{"id": "l1", "name": "infra"}]},
    "team": {"id": "t1", "name": "Omninode"},
    "project": {"id": "p1"},
}

_PROJECT_FIXTURE: dict[str, object] = {
    "id": "p1",
    "name": "Active Sprint",
    "description": "current cycle",
    "state": "started",
    "progress": 0.5,
    "url": "https://linear.app/omninode/project/p1",
}


# ---------- construction ----------


class TestConstruction:
    def test_raises_when_no_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(InfraAuthenticationError):
                AdapterLinearGraphQLProjectTracker()

    def test_falls_back_to_linear_api_key_env(self) -> None:
        with patch.dict("os.environ", {"LINEAR_API_KEY": "from-env"}, clear=True):
            adapter = AdapterLinearGraphQLProjectTracker()
        assert adapter._api_key == "from-env"

    def test_falls_back_to_linear_token_env(self) -> None:
        with patch.dict("os.environ", {"LINEAR_TOKEN": "tok"}, clear=True):
            adapter = AdapterLinearGraphQLProjectTracker()
        assert adapter._api_key == "tok"

    def test_explicit_api_key_overrides_env(self) -> None:
        with patch.dict("os.environ", {"LINEAR_API_KEY": "env"}, clear=True):
            adapter = AdapterLinearGraphQLProjectTracker(api_key="explicit")
        assert adapter._api_key == "explicit"

    def test_rejects_whitespace_only_api_key(self) -> None:
        """Whitespace-only credentials must fail at construction, not on first call."""
        with patch.dict("os.environ", {"LINEAR_API_KEY": "   "}, clear=True):
            with pytest.raises(InfraAuthenticationError):
                AdapterLinearGraphQLProjectTracker()

    def test_strips_whitespace_around_api_key(self) -> None:
        """A key with surrounding whitespace is accepted but stripped."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = AdapterLinearGraphQLProjectTracker(api_key="  real-key  ")
        assert adapter._api_key == "real-key"


# ---------- lifecycle ----------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"viewer": {"id": "u1", "name": "Tester"}})
        )
        adapter = _build_adapter(handler)
        try:
            assert await adapter.connect() is True
            assert adapter._connected is True
        finally:
            await adapter.close()

    @pytest.mark.asyncio
    async def test_health_check_reflects_connection_state(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"viewer": {"id": "u1", "name": "Tester"}})
        )
        adapter = _build_adapter(handler)
        before = await adapter.health_check()
        assert before.status == "not_connected"
        await adapter.connect()
        after = await adapter.health_check()
        assert after.status == "healthy"
        assert after.service_id == "linear-graphql"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_capabilities(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({}))
        adapter = _build_adapter(handler)
        caps = await adapter.get_capabilities()
        assert "read" in caps
        assert "write" in caps
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({}))
        adapter = _build_adapter(handler)
        await adapter.close()
        # Second close on the same adapter (caller-owned client) must not raise.
        await adapter.close()

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"viewer": {"id": "u1", "name": "Tester"}})
        )
        adapter = _build_adapter(handler)
        async with adapter as a:
            assert a is adapter
            assert adapter._connected is True
        assert adapter._connected is False


# ---------- domain operations ----------


class TestGetIssue:
    @pytest.mark.asyncio
    async def test_returns_issue(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({"issue": _ISSUE_FIXTURE}))
        adapter = _build_adapter(handler)
        issue = await adapter.get_issue("uuid-1")
        assert issue.identifier == "OMN-1"
        assert issue.title == "Test issue"
        assert issue.state == "Backlog"
        assert issue.assignee == "Tester"
        assert issue.labels == ["infra"]
        assert issue.team == "Omninode"
        assert issue.project_id == "p1"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_raises_keyerror_on_missing_issue(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({"issue": None}))
        adapter = _build_adapter(handler)
        with pytest.raises(KeyError):
            await adapter.get_issue("missing")
        await adapter.close()


class TestListIssues:
    @pytest.mark.asyncio
    async def test_returns_list(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"issues": {"nodes": [_ISSUE_FIXTURE]}})
        )
        adapter = _build_adapter(handler)
        issues = await adapter.list_issues(filters={"state": "Backlog"}, limit=5)
        assert len(issues) == 1
        assert issues[0].identifier == "OMN-1"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_nodes(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({"issues": {"nodes": []}}))
        adapter = _build_adapter(handler)
        issues = await adapter.list_issues(limit=10)
        assert issues == []
        await adapter.close()


class TestSearchIssues:
    @pytest.mark.asyncio
    async def test_returns_results(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"searchIssues": {"nodes": [_ISSUE_FIXTURE]}})
        )
        adapter = _build_adapter(handler)
        results = await adapter.search_issues("test", limit=5)
        assert len(results) == 1
        assert results[0].identifier == "OMN-1"
        await adapter.close()


class TestCreateIssue:
    @pytest.mark.asyncio
    async def test_creates_issue(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok(
                {"issueCreate": {"success": True, "issue": _ISSUE_FIXTURE}}
            )
        )
        adapter = _build_adapter(handler)
        issue = await adapter.create_issue(
            title="Test",
            description="desc",
            team="t1",
            labels=["l1"],
            assignee="u1",
            priority="2",
        )
        assert issue.identifier == "OMN-1"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_rejects_missing_team(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({}))
        adapter = _build_adapter(handler)
        with pytest.raises(InfraConnectionError):
            await adapter.create_issue(title="t", description="d")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_raises_when_success_false(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"issueCreate": {"success": False, "issue": None}})
        )
        adapter = _build_adapter(handler)
        with pytest.raises(InfraConnectionError):
            await adapter.create_issue(title="t", description="d", team="t1")
        await adapter.close()


class TestUpdateIssue:
    @pytest.mark.asyncio
    async def test_updates_issue(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok(
                {"issueUpdate": {"success": True, "issue": _ISSUE_FIXTURE}}
            )
        )
        adapter = _build_adapter(handler)
        issue = await adapter.update_issue("uuid-1", {"title": "renamed"})
        assert issue.identifier == "OMN-1"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_raises_when_failure(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"issueUpdate": {"success": False, "issue": None}})
        )
        adapter = _build_adapter(handler)
        with pytest.raises(KeyError):
            await adapter.update_issue("uuid-x", {"title": "x"})
        await adapter.close()


class TestAddComment:
    @pytest.mark.asyncio
    async def test_adds_comment(self) -> None:
        comment_fixture = {
            "id": "c1",
            "body": "hi",
            "createdAt": "2026-04-27T00:00:00.000Z",
            "user": {"id": "u1", "name": "Tester"},
        }
        handler = httpx.MockTransport(
            lambda _request: _ok(
                {"commentCreate": {"success": True, "comment": comment_fixture}}
            )
        )
        adapter = _build_adapter(handler)
        comment = await adapter.add_comment("uuid-1", "hi")
        assert comment.id == "c1"
        assert comment.body == "hi"
        assert comment.author == "Tester"
        await adapter.close()

    @pytest.mark.asyncio
    async def test_raises_when_failure(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"commentCreate": {"success": False}})
        )
        adapter = _build_adapter(handler)
        with pytest.raises(KeyError):
            await adapter.add_comment("uuid-x", "hi")
        await adapter.close()


class TestProjects:
    @pytest.mark.asyncio
    async def test_get_project(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"project": _PROJECT_FIXTURE})
        )
        adapter = _build_adapter(handler)
        project = await adapter.get_project("p1")
        assert project.id == "p1"
        assert project.name == "Active Sprint"
        assert project.progress == 0.5
        await adapter.close()

    @pytest.mark.asyncio
    async def test_get_project_missing(self) -> None:
        handler = httpx.MockTransport(lambda _request: _ok({"project": None}))
        adapter = _build_adapter(handler)
        with pytest.raises(KeyError):
            await adapter.get_project("missing")
        await adapter.close()

    @pytest.mark.asyncio
    async def test_list_projects(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: _ok({"projects": {"nodes": [_PROJECT_FIXTURE]}})
        )
        adapter = _build_adapter(handler)
        projects = await adapter.list_projects(limit=10)
        assert len(projects) == 1
        assert projects[0].id == "p1"
        await adapter.close()


# ---------- error mapping ----------


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_401_raises_authentication(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(401, json={"error": "unauthorized"})
        )
        adapter = _build_adapter(handler)
        with pytest.raises(InfraAuthenticationError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_403_raises_authentication(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(403, json={"error": "forbidden"})
        )
        adapter = _build_adapter(handler)
        with pytest.raises(InfraAuthenticationError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_429_raises_rate_limited(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(
                429,
                json={"error": "throttled"},
                headers={"Retry-After": "30"},
            )
        )
        adapter = _build_adapter(handler)
        with pytest.raises(InfraRateLimitedError) as exc_info:
            await adapter.connect()
        assert "rate limit" in str(exc_info.value).lower()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_500_raises_connection_error(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(503, json={"error": "down"})
        )
        adapter = _build_adapter(handler)
        with pytest.raises(InfraConnectionError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_400_raises_connection_but_does_not_burn_circuit(self) -> None:
        """Caller-side 4xx must not count toward circuit breaker."""
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(400, json={"error": "bad input"})
        )
        adapter = _build_adapter(handler)
        for _ in range(10):
            with pytest.raises(InfraConnectionError):
                await adapter.connect()
        # Circuit breaker counter must remain at 0 — 4xx are not transient.
        assert adapter._circuit_breaker_failures == 0
        assert adapter._circuit_breaker_open is False
        await adapter.close()

    @pytest.mark.asyncio
    async def test_timeout_raises_infra_timeout(self) -> None:
        def _raise(_request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("boom")

        handler = httpx.MockTransport(_raise)
        adapter = _build_adapter(handler)
        with pytest.raises(InfraTimeoutError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_transport_error_raises_connection(self) -> None:
        def _raise(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns")

        handler = httpx.MockTransport(_raise)
        adapter = _build_adapter(handler)
        with pytest.raises(InfraConnectionError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_graphql_errors_raise_connection(self) -> None:
        """GraphQL errors[] payloads must not count toward circuit breaker."""
        handler = httpx.MockTransport(lambda _request: _err(["something broke"]))
        adapter = _build_adapter(handler)
        for _ in range(10):
            with pytest.raises(InfraConnectionError) as exc_info:
                await adapter.connect()
            assert "something broke" in str(exc_info.value)
        # Caller-side schema/validation errors don't open the circuit.
        assert adapter._circuit_breaker_failures == 0
        assert adapter._circuit_breaker_open is False
        await adapter.close()

    @pytest.mark.asyncio
    async def test_invalid_json_raises_connection(self) -> None:
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(
                200, content=b"not-json", headers={"Content-Type": "text/plain"}
            )
        )
        adapter = _build_adapter(handler)
        with pytest.raises(InfraConnectionError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_api_key_never_in_error_message(self) -> None:
        secret = "lin_api_supersecret_value_xyz"
        handler = httpx.MockTransport(
            lambda _request: httpx.Response(401, json={"error": "unauthorized"})
        )
        adapter = _build_adapter(handler, api_key=secret)
        with pytest.raises(InfraAuthenticationError) as exc_info:
            await adapter.connect()
        assert secret not in str(exc_info.value)
        await adapter.close()


# ---------- circuit breaker ----------


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_repeated_failures_open_circuit(self) -> None:
        # 5 consecutive 503 failures should open the circuit; the 6th call
        # raises InfraUnavailableError without hitting the transport.
        from omnibase_infra.errors import InfraUnavailableError

        handler = httpx.MockTransport(
            lambda _request: httpx.Response(503, json={"error": "down"})
        )
        adapter = _build_adapter(handler)
        for _ in range(5):
            with pytest.raises(InfraConnectionError):
                await adapter.connect()
        with pytest.raises(InfraUnavailableError):
            await adapter.connect()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_success_resets_failure_counter(self) -> None:
        # Alternate success/failure: 3 fails then 1 success should reset the
        # internal failure counter.
        responses: list[httpx.Response] = [
            httpx.Response(503, json={"error": "down"}),
            httpx.Response(503, json={"error": "down"}),
            httpx.Response(503, json={"error": "down"}),
            _ok({"viewer": {"id": "u1", "name": "Tester"}}),
        ]
        idx = {"i": 0}

        def _next(_request: httpx.Request) -> httpx.Response:
            r = responses[idx["i"]]
            idx["i"] += 1
            return r

        handler = httpx.MockTransport(_next)
        adapter = _build_adapter(handler)
        for _ in range(3):
            with pytest.raises(InfraConnectionError):
                await adapter.connect()
        # Successful viewer query closes the circuit again.
        await adapter.connect()
        assert adapter._circuit_breaker_failures == 0
        await adapter.close()
