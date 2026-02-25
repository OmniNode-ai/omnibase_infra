# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""RewardAssignedEvent emitted for each target that received a reward.

Published to: ``onex.evt.omnimemory.reward-assigned.v1``

Ticket: OMN-2552
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRewardAssignedEvent(BaseModel):
    """Event emitted for each target_type that received a reward.

    Published to: ``onex.evt.omnimemory.reward-assigned.v1``

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
    target_id: UUID = Field(
        ...,
        description="ID of the target receiving this reward.",
    )
    target_type: Literal["tool", "model", "pattern", "agent"] = Field(
        ...,
        description="Category of the reward target.",
    )
    composite_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Composite scalar reward in [-1, 1].",
    )
    dimensions: dict[str, float] = Field(
        default_factory=dict,
        description="Named dimension scores from ModelScoreVector.",
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
