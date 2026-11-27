# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26106: a refused dod_verify start survives to the receipt JSON as a skip.

Drives the handler with the verdict hosted ``dod_verify`` returns when it was
given no ``DOD_VERIFY_START_BUDGET_S`` (``unresolved``,
``dispatch_budget_not_declared``, no checks), serialises the result as receipt
mode does and parses it back: the ticket is named ``skipped_dod_verify``,
nothing is posted and nothing is flipped.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16808_comment_idempotency import (
    _handler,
    _merged_pr,
    _request,
    _skill_result,
    _StatefulLinear,
)

pytestmark = pytest.mark.integration


async def test_budget_refusal_is_a_skip_in_the_round_tripped_receipt() -> None:
    receipt = _skill_result(total=0, verified=0, failed=0)
    verdict = receipt["result"]
    verdict["status"] = "unresolved"
    verdict["unresolved_cause"] = "dispatch_budget_not_declared"
    verdict["error_message"] = (
        "DISPATCH_BUDGET_NOT_DECLARED: refused before any check ran: "
        "DOD_VERIFY_START_BUDGET_S is absent."
    )
    receipt["result_model"] = (
        "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
    )
    receipt["result"] = {"terminal_payload": verdict, "workflow_result": "unresolved"}
    linear = _StatefulLinear()
    handler = _handler([receipt], linear, [_merged_pr(7001)])

    result = await handler.handle(_request(apply=True))

    parsed = ModelEvidenceAutocloseSweepResult.model_validate(
        result.model_dump(mode="json")
    )
    assert [o.decision.value for o in parsed.outcomes] == ["skipped_dod_verify"]
    assert parsed.tickets_gap_posted == parsed.tickets_flipped == 0
    assert linear.comments == linear.state_updates == []
