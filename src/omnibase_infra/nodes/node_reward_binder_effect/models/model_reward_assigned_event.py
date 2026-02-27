# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""RewardAssignedEvent emitted after an evaluation run.

Published to: ``onex.evt.omnimemory.reward-assigned.v1``

Ticket: OMN-2927
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRewardAssignedEvent(BaseModel):
    """Event emitted after an evaluation run with canonical score vector fields.

    Published to: ``onex.evt.omnimemory.reward-assigned.v1``

    Uses canonical omnibase_core.ModelScoreVector field shapes
    (correctness, safety, cost, latency, maintainability, human_time).

    ``evidence_refs`` must be traceable back to specific ``ModelEvidenceItem.item_id``
    values from the input ``ModelEvidenceBundle``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier.",
    )
    run_id: UUID = Field(
        ...,
        description="Evaluation run ID.",
    )
    correctness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Gate-derived correctness: 1.0 if all gates pass, 0.0 if any fail.",
    )
    safety: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Security, PII, and blacklist gate composite score.",
    )
    cost: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Inverted cost score: lower cost = higher score.",
    )
    latency: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Inverted latency score: lower latency = higher score.",
    )
    maintainability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cyclomatic complexity delta and test coverage composite score.",
    )
    human_time: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Inverted human intervention score: fewer retries/reviews = higher score.",
    )
    evidence_refs: tuple[UUID, ...] = Field(
        default_factory=tuple,
        description="ModelEvidenceItem.item_id values supporting this reward.",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event emission.",
    )


__all__: list[str] = ["ModelRewardAssignedEvent"]
