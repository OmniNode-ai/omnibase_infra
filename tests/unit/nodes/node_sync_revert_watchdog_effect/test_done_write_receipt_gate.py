# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The watchdog's re-flip writes Done, so it needs a bound PASS receipt (OMN-20368).

On dev the watchdog re-flipped any human-set prior Done the moment it saw an
automation revert, with no receipt in sight. These tests fail there and pass
with the gate.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from omnibase_infra.handlers.done_write_receipt_guard import (
    DoneWriteReceiptGuard,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.handlers.handler_sync_revert_watchdog import (
    HandlerSyncRevertWatchdog,
)
from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.enum_sync_revert_watchdog_decision import (
    EnumSyncRevertWatchdogDecision,
)

from .test_handler_sync_revert_watchdog import (
    FakeLinearClient,
    _automation_revert_entry,
    _human_set_done_entry,
    _issue_stub,
    _now,
    _request,
)

pytestmark = pytest.mark.unit

_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)
_DESCRIPTION = "## Acceptance Criteria\n- **AC1**: the ticket is restored\n"


class _DescribedLinear(FakeLinearClient):
    async def fetch_issue_description(self, issue_id, timeout):
        return _DESCRIPTION, ""


def _verdict(*, bind: bool, status: str = "verified") -> dict[str, object]:
    return {
        "status": status,
        "total_checks": 1,
        "verified_count": 1 if status == "verified" else 0,
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


def _guard(result: dict[str, object] | None) -> DoneWriteReceiptGuard:
    async def run(ticket_id: str, cwd: str, timeout: float):
        if result is None:
            return None, -1, "Timeout running dod_verify"
        return {"result_model": _STATE_MODEL, "result": result}, 0, ""

    return DoneWriteReceiptGuard(run_dod_verify=run)


def _linear() -> FakeLinearClient:
    now = _now()
    history = [
        _human_set_done_entry(now - timedelta(days=1)),
        _automation_revert_entry(now - timedelta(seconds=5)),
    ]
    return _DescribedLinear(
        issues=[_issue_stub()], history_by_issue={"issue-1": history}
    )


async def test_revert_with_no_bound_receipt_is_left_alone_and_says_why() -> None:
    linear = _linear()
    handler = HandlerSyncRevertWatchdog(
        linear_client=linear, done_write_guard=_guard(_verdict(bind=False))
    )
    result = await handler.handle(_request(apply=True))
    outcome = result.outcomes[0]
    assert outcome.decision == EnumSyncRevertWatchdogDecision.SKIPPED_NO_BOUND_RECEIPT
    assert "AC1" in outcome.reason
    assert linear.state_updates == []
    assert linear.comments_posted == []
    assert result.tickets_reflipped == 0
    assert result.tickets_skipped == 1


async def test_dry_run_previews_the_refusal_too() -> None:
    linear = _linear()
    handler = HandlerSyncRevertWatchdog(
        linear_client=linear, done_write_guard=_guard(_verdict(bind=False))
    )
    result = await handler.handle(_request(apply=False))
    assert result.outcomes[0].decision == (
        EnumSyncRevertWatchdogDecision.SKIPPED_NO_BOUND_RECEIPT
    )


async def test_unreadable_verifier_is_a_refusal_not_a_flip() -> None:
    linear = _linear()
    handler = HandlerSyncRevertWatchdog(
        linear_client=linear, done_write_guard=_guard(None)
    )
    result = await handler.handle(_request(apply=True))
    assert result.outcomes[0].decision == (
        EnumSyncRevertWatchdogDecision.SKIPPED_NO_BOUND_RECEIPT
    )
    assert linear.state_updates == []


async def test_bound_pass_receipt_reflips() -> None:
    linear = _linear()
    handler = HandlerSyncRevertWatchdog(
        linear_client=linear, done_write_guard=_guard(_verdict(bind=True))
    )
    result = await handler.handle(_request(apply=True))
    assert result.outcomes[0].decision == EnumSyncRevertWatchdogDecision.REFLIPPED
    assert len(linear.state_updates) == 1
    assert result.tickets_reflipped == 1
