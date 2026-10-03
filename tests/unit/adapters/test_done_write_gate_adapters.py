# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The generic Linear adapters refuse a Done write without a bound receipt (OMN-20368).

On dev ``AdapterLinearGraphQLProjectTracker.update_issue`` forwarded any state
write straight to ``issueUpdate``, Done included. These tests fail there and
pass with the gate; a write to any other state is unchanged.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.adapters.ticket.adapter_ticket_service_linear import (
    AdapterTicketLinear,
)
from omnibase_infra.handlers.done_write_receipt_guard import (
    DoneWriteReceiptGuard,
    DoneWriteRefusedError,
    is_done_state,
)

pytestmark = pytest.mark.unit

_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)
_DESCRIPTION = (
    "## Acceptance Criteria\n- **AC1**: the adapter refuses an unbound Done\n"
)
_DONE_STATE_ID = "state-done"
_STARTED_STATE_ID = "state-started"
_WORKFLOW_STATES = {
    _DONE_STATE_ID: {"id": _DONE_STATE_ID, "name": "Done", "type": "completed"},
    _STARTED_STATE_ID: {
        "id": _STARTED_STATE_ID,
        "name": "In Progress",
        "type": "started",
    },
}
_ISSUE = {
    "id": "uuid-1",
    "identifier": "OMN-1",
    "title": "Test issue",
    "description": _DESCRIPTION,
    "priority": 2,
    "url": "https://linear.app/omninode/issue/OMN-1",
    "createdAt": "2026-04-27T00:00:00.000Z",
    "updatedAt": "2026-04-27T01:00:00.000Z",
    "state": {"name": "In Progress", "type": "started"},
    "team": {"id": "t1", "name": "Omninode"},
}


def _verdict(*, bind: bool) -> dict[str, object]:
    return {
        "status": "verified",
        "total_checks": 1,
        "verified_count": 1,
        "failed_count": 0,
        "non_probative_count": 0,
        "checks": [
            {
                "evidence_id": "t1",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC1"] if bind else [],
            }
        ],
    }


class _Verifier:
    """A dod_verify runner that records how often it was asked."""

    def __init__(self, verdict: dict[str, object] | None) -> None:
        self.calls: list[str] = []
        self._verdict = verdict

    async def __call__(
        self, ticket_id: str, cwd: str, timeout: float
    ) -> tuple[dict[str, object] | None, int, str]:
        self.calls.append(ticket_id)
        if self._verdict is None:
            return None, -1, "Timeout running dod_verify"
        return {"result_model": _STATE_MODEL, "result": self._verdict}, 0, ""


def _tracker(
    verifier: _Verifier, requests: list[dict[str, object]]
) -> AdapterLinearGraphQLProjectTracker:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        query = body["query"]
        variables = body.get("variables", {})
        if "workflowState" in query:
            state = _WORKFLOW_STATES.get(variables["id"])
            return httpx.Response(200, json={"data": {"workflowState": state}})
        if "issueUpdate" in query:
            return httpx.Response(
                200, json={"data": {"issueUpdate": {"success": True, "issue": _ISSUE}}}
            )
        return httpx.Response(200, json={"data": {"issue": _ISSUE}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return AdapterLinearGraphQLProjectTracker(
        api_key="test-key",
        client=client,
        done_write_guard=DoneWriteReceiptGuard(run_dod_verify=verifier),
    )


def _writes(requests: list[dict[str, object]]) -> list[dict[str, object]]:
    return [r for r in requests if "issueUpdate" in str(r["query"])]


class TestProjectTrackerDoneWrite:
    async def test_done_write_with_no_bound_receipt_is_refused_and_nothing_is_written(
        self,
    ) -> None:
        requests: list[dict[str, object]] = []
        verifier = _Verifier(_verdict(bind=False))
        adapter = _tracker(verifier, requests)
        with pytest.raises(DoneWriteRefusedError) as refused:
            await adapter.update_issue("uuid-1", {"stateId": _DONE_STATE_ID})
        assert "AC1" in str(refused.value)
        assert verifier.calls == ["OMN-1"]
        assert _writes(requests) == []

    async def test_done_write_when_the_verifier_cannot_be_read_is_refused(self) -> None:
        requests: list[dict[str, object]] = []
        adapter = _tracker(_Verifier(None), requests)
        with pytest.raises(DoneWriteRefusedError):
            await adapter.update_issue("uuid-1", {"stateId": _DONE_STATE_ID})
        assert _writes(requests) == []

    async def test_done_write_with_a_bound_pass_receipt_goes_through(self) -> None:
        requests: list[dict[str, object]] = []
        adapter = _tracker(_Verifier(_verdict(bind=True)), requests)
        issue = await adapter.update_issue("uuid-1", {"stateId": _DONE_STATE_ID})
        assert issue.identifier == "OMN-1"
        assert len(_writes(requests)) == 1

    async def test_a_name_shaped_done_target_is_gated_too(self) -> None:
        requests: list[dict[str, object]] = []
        adapter = _tracker(_Verifier(_verdict(bind=False)), requests)
        with pytest.raises(DoneWriteRefusedError):
            await adapter.update_issue("uuid-1", {"state": "Done"})
        assert _writes(requests) == []

    async def test_a_state_id_that_cannot_be_resolved_is_refused(self) -> None:
        requests: list[dict[str, object]] = []
        verifier = _Verifier(_verdict(bind=True))
        adapter = _tracker(verifier, requests)
        with pytest.raises(DoneWriteRefusedError):
            await adapter.update_issue("uuid-1", {"stateId": "state-unknown"})
        assert _writes(requests) == []

    async def test_other_state_writes_are_unchanged(self) -> None:
        requests: list[dict[str, object]] = []
        verifier = _Verifier(None)
        adapter = _tracker(verifier, requests)
        await adapter.update_issue("uuid-1", {"stateId": _STARTED_STATE_ID})
        assert len(_writes(requests)) == 1
        assert verifier.calls == []

    async def test_writes_that_name_no_state_are_unchanged(self) -> None:
        requests: list[dict[str, object]] = []
        verifier = _Verifier(None)
        adapter = _tracker(verifier, requests)
        await adapter.update_issue("uuid-1", {"title": "renamed"})
        assert len(_writes(requests)) == 1
        assert verifier.calls == []
        assert not any("workflowState" in str(r["query"]) for r in requests)


class TestTicketServiceDoneWrite:
    """``AdapterTicketLinear.update_ticket_status`` is a stub today; its gate is not."""

    @staticmethod
    def _adapter(verifier: _Verifier) -> AdapterTicketLinear:
        adapter = AdapterTicketLinear(
            linear_api_key="test-key",
            done_write_guard=DoneWriteReceiptGuard(run_dod_verify=verifier),
        )
        return adapter

    async def test_done_with_no_bound_receipt_is_refused_before_the_stub(self) -> None:
        verifier = _Verifier(_verdict(bind=False))
        adapter = self._adapter(verifier)
        with (
            patch.object(
                adapter,
                "get_ticket",
                AsyncMock(
                    return_value={"identifier": "OMN-1", "description": _DESCRIPTION}
                ),
            ),
            pytest.raises(DoneWriteRefusedError),
        ):
            await adapter.update_ticket_status("OMN-1", "Done")
        assert verifier.calls == ["OMN-1"]

    async def test_done_with_a_bound_receipt_reaches_the_stub(self) -> None:
        adapter = self._adapter(_Verifier(_verdict(bind=True)))
        with (
            patch.object(
                adapter,
                "get_ticket",
                AsyncMock(
                    return_value={"identifier": "OMN-1", "description": _DESCRIPTION}
                ),
            ),
            pytest.raises(NotImplementedError, match="update_ticket_status"),
        ):
            await adapter.update_ticket_status("OMN-1", "Done")

    async def test_other_statuses_never_consult_the_verifier(self) -> None:
        verifier = _Verifier(None)
        adapter = self._adapter(verifier)
        with pytest.raises(NotImplementedError, match="update_ticket_status"):
            await adapter.update_ticket_status("OMN-1", "In Progress")
        assert verifier.calls == []


@pytest.mark.parametrize(
    ("name", "state_type", "expected"),
    [
        ("Done", None, True),
        ("done", None, True),
        (None, "completed", True),
        ("Shipped", "completed", True),
        ("In Progress", "started", False),
        ("Canceled", "canceled", False),
        ("Duplicate", "canceled", False),
        (None, None, False),
    ],
)
def test_is_done_state(
    name: str | None, state_type: str | None, expected: bool
) -> None:
    assert is_done_state(name, state_type) is expected
