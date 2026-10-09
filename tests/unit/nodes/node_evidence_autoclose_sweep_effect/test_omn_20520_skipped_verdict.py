# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20520: unchanged gaps are deduplicated; skipped runs are not gaps."""

from copy import deepcopy

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)

from .test_omn_16808_comment_idempotency import (
    _handler,
    _merged_pr,
    _request,
    _skill_result,
    _StatefulLinear,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_unchanged_gap_set_across_runs_posts_once() -> None:
    first = _skill_result(total=6, verified=3, failed=3)
    verdict = first["result"]["terminal_payload"]
    verdict["checks"].append({"evidence_id": "missing-proof", "status": "failed"})
    second = deepcopy(first)
    second_verdict = second["result"]["terminal_payload"]
    second_verdict.update(total_checks=9, verified_count=6)
    second_verdict["checks"].reverse()
    linear = _StatefulLinear()

    # Separate handlers model separate scheduled runs, with only Linear history shared.
    results = []
    for receipt in (first, second):
        handler = _handler([receipt], linear, [_merged_pr(7001)])
        results.append(await handler.handle(_request(apply=True)))

    assert len(linear.comments) == 1
    assert (
        results[1].outcomes[0].decision
        == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
    )
    assert linear.state_updates == []


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("behavior_proving", [0, 1])
@pytest.mark.parametrize("reason_source", ["terminal", "check", "cause", "missing"])
async def test_skipped_verdict_reports_reason_without_gap(
    apply: bool, behavior_proving: int, reason_source: str
) -> None:
    receipt = _skill_result(
        total=10, verified=5, failed=0, behavior_proving=behavior_proving
    )
    verdict = receipt["result"]
    verdict["status"] = "skipped"
    verdict["skipped_count"] = 1
    reason = "behavior check exceeded its 30 s execution ceiling"
    if reason_source == "terminal":
        verdict["error_message"] = reason
    elif reason_source in ("check", "cause"):
        verdict["checks"].append(
            {
                "evidence_id": "live-proof",
                "status": "skipped",
                "message": reason,
                "unverifiable_cause": (
                    "check_budget_exceeded" if reason_source == "cause" else None
                ),
            }
        )
    # Non-verified CLI results carry the verdict on the runtime-summary arm.
    receipt["result_model"] = (
        "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
    )
    receipt["result"] = {"terminal_payload": verdict, "workflow_result": "skipped"}
    handler = _handler([receipt], linear := _StatefulLinear(), [_merged_pr(7001)])

    result = await handler.handle(_request(apply=apply))

    outcome = result.outcomes[0]
    assert outcome.decision.value == "skipped_dod_verify"
    assert "skipped" in outcome.reason
    if reason_source == "cause":
        assert "check_budget_exceeded" in outcome.reason
    assert (
        reason if reason_source != "missing" else "no reason supplied"
    ) in outcome.reason
    assert result.tickets_skipped == 1
    assert (
        result.tickets_gap_posted
        == result.tickets_flipped
        == result.tickets_errored
        == 0
    )
    assert not outcome.applied and not outcome.linear_comment_posted
    assert linear.comments == linear.state_updates == []
