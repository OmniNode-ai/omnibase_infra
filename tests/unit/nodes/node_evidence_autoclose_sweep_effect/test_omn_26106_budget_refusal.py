# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26106: a verifier that refused to start is not a gap in the ticket.

Hosted ``dod_verify`` refuses to start without ``DOD_VERIFY_START_BUDGET_S``
(omnimarket OMN-25856) and returns ``status=unresolved``,
``unresolved_cause=dispatch_budget_not_declared`` and ``total_checks=0``. The
sweep read that as ``0/0 ACs verified, 0 failed`` and wrote it to Linear as an
evidence gap about the ticket. No check ran, so the run learned nothing about
the criteria and must say so, writing nothing.
"""

import pytest

from .test_omn_16808_comment_idempotency import (
    _handler,
    _merged_pr,
    _request,
    _skill_result,
    _StatefulLinear,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_REFUSAL = (
    "DISPATCH_BUDGET_NOT_DECLARED: OMN-26106 was refused before any check ran: "
    "DOD_VERIFY_START_BUDGET_S is absent, malformed or not below the dispatch "
    "deadline."
)


@pytest.mark.parametrize("apply", [False, True])
async def test_budget_refusal_writes_no_gap_comment(apply: bool) -> None:
    receipt = _skill_result(total=0, verified=0, failed=0)
    verdict = receipt["result"]
    verdict["status"] = "unresolved"
    verdict["unresolved_cause"] = "dispatch_budget_not_declared"
    verdict["error_message"] = _REFUSAL
    receipt["result_model"] = (
        "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
    )
    receipt["result"] = {"terminal_payload": verdict, "workflow_result": "unresolved"}
    handler = _handler([receipt], linear := _StatefulLinear(), [_merged_pr(7001)])

    result = await handler.handle(_request(apply=apply))

    outcome = result.outcomes[0]
    assert outcome.decision.value == "skipped_dod_verify"
    assert "dispatch_budget_not_declared" in outcome.reason
    assert "DOD_VERIFY_START_BUDGET_S" in outcome.reason
    assert result.tickets_gap_posted == result.tickets_flipped == 0
    assert not outcome.linear_comment_posted
    assert linear.comments == linear.state_updates == []
