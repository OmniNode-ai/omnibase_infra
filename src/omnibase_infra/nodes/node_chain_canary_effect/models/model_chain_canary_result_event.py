# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Map a chain-canary receipt into the shared board-probe event (OMN-19937)."""

from __future__ import annotations

from datetime import datetime

from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_probe_outcome import (
    EnumBoardProbeOutcome,
)
from omnibase_infra.nodes.node_board_probe_effect.models.enum_board_subject_kind import (
    EnumBoardSubjectKind,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_board_probe_result_event import (
    ModelBoardProbeResultEvent,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)


def _board_outcome_for(
    verdict: EnumChainCanaryVerdict,
) -> EnumBoardProbeOutcome:
    if verdict is EnumChainCanaryVerdict.GREEN:
        return EnumBoardProbeOutcome.PASS
    if verdict is EnumChainCanaryVerdict.SKIPPED_DISABLED:
        return EnumBoardProbeOutcome.INDETERMINATE
    return EnumBoardProbeOutcome.FAIL


def chain_canary_result_event_from(
    result: ModelChainCanaryResult,
    *,
    subject: str,
    repo: str,
    sha: str,
    surface_instance: str,
    finished_at: datetime,
) -> ModelBoardProbeResultEvent:
    """Map every canary verdict fail-closed into the shared event vocabulary."""
    return ModelBoardProbeResultEvent(
        check_id="chain_canary",
        subject_kind=EnumBoardSubjectKind.CHAIN,
        subject=subject,
        repo=repo,
        sha=sha,
        surface_instance=surface_instance,
        execution_id=str(result.probe_correlation_id),
        outcome=_board_outcome_for(result.verdict),
        reasons=(result.verdict.value, result.detail),
        evidence_items=(),
        finished_at=finished_at,
    )


__all__ = ["chain_canary_result_event_from"]
