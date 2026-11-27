# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26106: replay hosted dod_verify's budget refusal through the sweep handler.

The recorded verdict is what hosted ``dod_verify`` answered the evidence close
sweep when it declared no ``DOD_VERIFY_START_BUDGET_S``: ``unresolved`` with
``total_checks=0``. The sweep must record ``skipped_dod_verify`` and write
nothing to Linear, instead of posting ``0/0 ACs verified`` as an evidence gap.
"""

from typing import Any

import pytest

from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16808_comment_idempotency import (
    _handler,
    _merged_pr,
    _request,
    _skill_result,
    _StatefulLinear,
)


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/dod_verify_budget_refusal_omn26106.json"
)
@pytest.mark.asyncio
async def test_recorded_budget_refusal_writes_nothing_to_linear(
    recorded_response: dict[str, Any],
) -> None:
    recorded = recorded_response["response"]
    receipt = _skill_result(total=0, verified=0, failed=0)
    verdict = receipt["result"]
    verdict.update(recorded)
    receipt["result_model"] = (
        "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
    )
    receipt["result"] = {"terminal_payload": verdict, "workflow_result": "unresolved"}
    linear = _StatefulLinear()
    handler = _handler([receipt], linear, [_merged_pr(7001)])

    result = await handler.handle(_request(apply=True))

    outcome = result.outcomes[0]
    assert outcome.decision.value == "skipped_dod_verify"
    assert recorded["unresolved_cause"] in outcome.reason
    assert not outcome.linear_comment_posted
    assert linear.comments == linear.state_updates == []
