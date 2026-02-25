# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Payload model for the write-conflict intent.

Drives HandlerWriteConflict: idempotent insert of a single conflict pair
into decision_conflicts via INSERT ON CONFLICT DO NOTHING.

Related Tickets:
    - OMN-2765: NodeDecisionStoreEffect implementation
    - OMN-2764: DB migrations
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelPayloadWriteConflict(BaseModel):
    """Payload for a single decision-conflict write.

    Produced by HandlerWriteDecision (Stage 2) and may also be used
    for direct conflict injection in tests or tooling.

    The pair (decision_min_id, decision_max_id) must satisfy
    decision_min_id < decision_max_id — matching the DB constraint
    chk_conflict_pair_order. The handler normalises the pair if needed.

    Attributes:
        intent_type: Routing key — always "decision_store.write_conflict".
        correlation_id: Correlation UUID from the originating context.
        decision_min_id: Smaller UUID of the conflict pair (after ordering).
        decision_max_id: Larger UUID of the conflict pair (after ordering).
        structural_confidence: Confidence score in [0.000, 1.000] from the
            structural_confidence() pure function.
        final_severity: Resolved severity (HIGH / MEDIUM / LOW).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent_type: Literal["decision_store.write_conflict"] = Field(
        default="decision_store.write_conflict",
        description="Routing key for this intent.",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation UUID from the originating pipeline/session.",
    )
    decision_min_id: UUID = Field(
        ...,
        description="Smaller UUID of the conflict pair.",
    )
    decision_max_id: UUID = Field(
        ...,
        description="Larger UUID of the conflict pair.",
    )
    structural_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Structural confidence score (0.0-1.0).",
    )
    final_severity: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        ...,
        description="Resolved severity for this conflict.",
    )


__all__: list[str] = ["ModelPayloadWriteConflict"]
