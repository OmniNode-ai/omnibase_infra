# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19937: chain-canary verdicts use the board probe event vocabulary."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from omnibase_infra.nodes.node_board_probe_effect.models import (
    EnumBoardProbeOutcome,
    EnumBoardSubjectKind,
)
from omnibase_infra.nodes.node_chain_canary_effect.models import (
    EnumChainCanaryVerdict,
    ModelChainCanaryResult,
    chain_canary_result_event_from,
)

pytestmark = [pytest.mark.unit]

CORRELATION_ID = UUID("11111111-1111-4111-8111-111111111111")
PROBE_CORRELATION_ID = UUID("22222222-2222-4222-8222-222222222222")
FINISHED_AT = datetime(2026, 9, 28, 20, 15, tzinfo=UTC)


def _result(verdict: EnumChainCanaryVerdict) -> ModelChainCanaryResult:
    return ModelChainCanaryResult(
        correlation_id=CORRELATION_ID,
        probe_correlation_id=PROBE_CORRELATION_ID,
        verdict=verdict,
        success=verdict
        in (EnumChainCanaryVerdict.GREEN, EnumChainCanaryVerdict.SKIPPED_DISABLED),
        detail=f"detail for {verdict.value}",
        runtime_command="node_delegate_skill_orchestrator",
    )


def _expected_outcome(verdict: EnumChainCanaryVerdict) -> EnumBoardProbeOutcome:
    if verdict is EnumChainCanaryVerdict.GREEN:
        return EnumBoardProbeOutcome.PASS
    if verdict is EnumChainCanaryVerdict.SKIPPED_DISABLED:
        return EnumBoardProbeOutcome.INDETERMINATE
    return EnumBoardProbeOutcome.FAIL


@pytest.mark.parametrize("verdict", list(EnumChainCanaryVerdict))
def test_every_chain_canary_verdict_maps_fail_closed(
    verdict: EnumChainCanaryVerdict,
) -> None:
    result = _result(verdict)

    event = chain_canary_result_event_from(
        result,
        subject="delegation_chain",
        repo="OmniNode-ai/omnibase_infra",
        sha="b" * 40,
        surface_instance="compose-dev",
        finished_at=FINISHED_AT,
    )

    assert event.check_id == "chain_canary"
    assert event.subject_kind is EnumBoardSubjectKind.CHAIN
    assert event.outcome is _expected_outcome(verdict)
    assert event.execution_id == str(PROBE_CORRELATION_ID)
    assert verdict.value in event.reasons
    assert result.detail in event.reasons
