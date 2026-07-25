# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``ModelWorkflowReceiptReplayResult`` (OMN-15095)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.verification.workflow_receipt.enum_workflow_receipt_replay_verdict import (
    EnumWorkflowReceiptReplayVerdict,
)
from omnibase_infra.verification.workflow_receipt.model_workflow_receipt_replay_diagnostic import (
    ModelWorkflowReceiptReplayDiagnostic,
)


class ModelWorkflowReceiptReplayResult(BaseModel):
    """PASS/FAIL verdict for one replay-then-diff run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: UUID
    correlation_id: UUID
    verdict: EnumWorkflowReceiptReplayVerdict
    events_replayed: int
    replayed_topic: str
    terminal_event_hash_match: bool
    diagnostics: tuple[ModelWorkflowReceiptReplayDiagnostic, ...] = Field(
        default_factory=tuple
    )


__all__ = ["ModelWorkflowReceiptReplayResult"]
