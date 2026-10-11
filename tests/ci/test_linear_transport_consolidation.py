# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17678: one GraphQL transport, preserving caller retry and identity policy."""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.adapter_project_tracker_linear import (
    AdapterProjectTrackerLinear,
)
from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
    _issue_from_graphql,
)
from omnibase_infra.adapters.ticket.adapter_ticket_service_linear import (
    AdapterTicketLinear,
)
from omnibase_infra.handlers.handler_linear_db_error_reporter import (
    HandlerLinearDbErrorReporter,
)
from omnibase_infra.handlers.models.model_db_error_event import ModelDbErrorEvent
from omnibase_infra.models.github.model_pr_merged_event import ModelPRMergedEvent
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _LinearClient,
)
from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.handlers.handler_in_progress_probe_hygiene import (
    LinearHygieneTransport,
)
from omnibase_infra.nodes.node_merge_gate_effect.handlers.handler_upsert_merge_gate import (
    HandlerUpsertMergeGate,
)
from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_result import (
    ModelMergeGateResult,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.handlers.handler_sync_revert_watchdog import (
    _LinearClient as WatchdogLinearClient,
)
from omnibase_infra.services.post_merge.config import ConfigPostMergeConsumer
from omnibase_infra.services.post_merge.consumer import PostMergeConsumer
from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import EnumFindingSeverity
from omnibase_infra.services.post_merge.model_post_merge_finding import (
    ModelPostMergeFinding,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_OWNER = Path(
    "src/omnibase_infra/adapters/project_tracker/linear_graphql_project_tracker_adapter.py"
)


def _endpoint_lines(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "api.linear.app/graphql" in node.value
    ]


def test_graphql_endpoint_has_one_transport_owner() -> None:
    owners = {
        path.relative_to(_ROOT): lines
        for path in (_ROOT / "src").rglob("*.py")
        if (lines := _endpoint_lines(path.read_text()))
    }
    assert owners == {_OWNER: _endpoint_lines((_ROOT / _OWNER).read_text())}, owners
    assert (
        len(owners[_OWNER]) == 1
    )  # Positive control: the owning transport was scanned.


def test_endpoint_guard_detects_a_reintroduced_caller() -> None:
    assert _endpoint_lines('URL = "https://api.linear.app/graphql"') == [1]
    assert _endpoint_lines('URL = "https://example.com/graphql"') == []


@pytest.mark.asyncio
async def test_shared_transport_preserves_wire_request_and_borrowed_client() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "2"}, json={"errors": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with AdapterLinearGraphQLProjectTracker.graphql_transport(
            api_key="Bearer application-test-token", timeout_seconds=7.0, client=client
        ) as transport:
            response = await transport.post_graphql("query Q { ok }", {"id": "issue"})
        assert not client.is_closed
    assert len(requests) == 1  # No viewer query or implicit retry.
    assert requests[0].headers["Authorization"] == "Bearer application-test-token"
    assert json.loads(requests[0].content) == {
        "query": "query Q { ok }",
        "variables": {"id": "issue"},
    }
    assert requests[0].extensions["timeout"]["read"] == 7.0
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "2"


@pytest.mark.asyncio
async def test_sweep_uses_shared_transport_with_its_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    async def post(
        self: AdapterLinearGraphQLProjectTracker,
        query: str,
        variables: dict[str, object] | None = None,
    ) -> httpx.Response:
        calls.append((self._request_headers["Authorization"], query, variables))
        return httpx.Response(
            200,
            json={"data": {"ok": True}},
            request=httpx.Request("POST", self._endpoint),
        )

    monkeypatch.setattr(AdapterLinearGraphQLProjectTracker, "post_graphql", post)
    sweep = _LinearClient(api_key="explicit-identity", base_delay_seconds=0)
    assert await sweep._query("query Q { ok }", {"id": "issue"}) == {"ok": True}
    assert calls == [("explicit-identity", "query Q { ok }", {"id": "issue"})]


@pytest.mark.parametrize(
    "caller",
    [
        "discovery",
        "ticket",
        "hygiene",
        "watchdog",
        "reporter",
        "merge_gate",
        "post_merge",
    ],
)
@pytest.mark.asyncio
async def test_other_runtime_callers_use_shared_transport(
    caller: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Intercept the real adapter seam, preserving each caller's domain result."""
    issue_id = uuid4()
    issue = {
        "id": str(issue_id),
        "identifier": "TEST-1",
        "url": "https://linear.app/test/1",
    }
    data = {
        "teams": {"nodes": [{"id": "team", "name": "Test", "key": "TEST"}]},
        "issueSearch": {"nodes": [issue]},
        "issueCreate": {"success": True, "issue": issue},
        "ok": True,
    }
    calls: list[tuple[str, dict[str, object] | None]] = []

    async def post(
        self: AdapterLinearGraphQLProjectTracker,
        query: str,
        variables: dict[str, object] | None = None,
    ) -> httpx.Response:
        assert self._request_headers["Authorization"] == "explicit-identity"
        calls.append((query, variables))
        return httpx.Response(
            200, json={"data": data}, request=httpx.Request("POST", self._endpoint)
        )

    monkeypatch.setattr(AdapterLinearGraphQLProjectTracker, "post_graphql", post)
    if caller == "discovery":
        tracker = AdapterProjectTrackerLinear(api_key="explicit-identity")
        try:
            assert (await tracker.list_teams())[0].key == "TEST"
        finally:
            await tracker.close()
    elif caller == "ticket":
        tickets = AdapterTicketLinear(linear_api_key="explicit-identity")
        try:
            assert (await tickets.get_ticket("TEST-1"))["id"] == str(issue_id)
        finally:
            await tickets.close()
    elif caller == "hygiene":
        assert (
            await LinearHygieneTransport(api_key="explicit-identity").query(
                "query Q { ok }", {}
            )
            == data
        )
    elif caller == "watchdog":
        assert (
            await WatchdogLinearClient(api_key="explicit-identity")._query(
                "query Q { ok }", {}, 7.0
            )
            == data
        )
    elif caller == "reporter":
        reporter = HandlerLinearDbErrorReporter(
            linear_api_key="explicit-identity", linear_team_id="team"
        )
        event = ModelDbErrorEvent(
            error_message="test failure",
            fingerprint="test",
            first_seen_at=datetime.now(UTC),
            service="test",
        )
        assert await reporter._create_linear_issue(event) == (issue_id, issue["url"])
        assert calls[0][1]["priority"] == 3
    elif caller == "merge_gate":
        handler = HandlerUpsertMergeGate(
            linear_api_key="explicit-identity", linear_team_id="team"
        )
        payload = ModelMergeGateResult(
            gate_id=uuid4(),
            pr_ref="OmniNode-ai/omnibase_infra#1",
            head_sha="a" * 40,
            base_sha="b" * 40,
            decision="QUARANTINE",
            tier="tier-a",
            decided_at=datetime.now(UTC),
        )
        await handler._open_quarantine_ticket(payload, uuid4())
        assert calls[0][1]["priority"] == 1
    else:
        consumer = PostMergeConsumer(
            ConfigPostMergeConsumer(
                kafka_bootstrap_servers="test:9092",
                linear_api_key="explicit-identity",
                linear_team_id="team",
            )
        )
        event = ModelPRMergedEvent(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=1,
            base_ref="dev",
            head_ref="test",
            merge_sha="a" * 40,
            author="test",
        )
        finding = ModelPostMergeFinding(
            stage=EnumCheckStage.CONTRACT_SWEEP,
            severity=EnumFindingSeverity.HIGH,
            title="test finding",
            description="test description",
        )
        assert await consumer._create_linear_ticket(event, finding) == "TEST-1"
    assert len(calls) == 1


@pytest.mark.live_contact("tests/ci/fixtures/linear_graphql_issue_omn17678.json")
def test_shared_adapter_reads_recorded_linear_issue(
    recorded_response: dict[str, object],
) -> None:
    response = recorded_response["response"]
    assert isinstance(response, dict)
    assert "errors" not in response
    data = response["data"]
    assert isinstance(data, dict)
    raw_issue = data["issue"]
    assert isinstance(raw_issue, dict)

    issue = _issue_from_graphql(raw_issue)

    assert issue.id == "91584acf-d9b2-4f66-9ee1-910d8385ccb9"
    assert issue.identifier == "OMN-17678"
    assert issue.title == raw_issue["title"]
    assert issue.url == "https://linear.app/omninode/issue/OMN-17678"
    assert issue.state == "Backlog"
    assert issue.priority == "0"
    assert issue.created_at == datetime(2026, 9, 3, 0, 45, 53, 60000, tzinfo=UTC)
    assert issue.updated_at == issue.created_at
