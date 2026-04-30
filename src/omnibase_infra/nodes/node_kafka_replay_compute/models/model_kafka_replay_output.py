# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output model for Kafka replay compute."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.projection import ModelSequenceInfo


class ModelKafkaReplayOutput(BaseModel):
    """Replay result and offset proof surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    events_replayed: int = Field(..., ge=0, description="Total replayed events.")
    last_offset_per_topic: dict[str, int] = Field(
        default_factory=dict,
        description="Last replayed offset by topic name across all partitions.",
    )
    sequence_info_per_topic: dict[str, ModelSequenceInfo] = Field(
        default_factory=dict,
        description="ModelSequenceInfo-compatible offset progress per topic-partition.",
    )
    started_at: datetime = Field(..., description="Replay start timestamp.")
    completed_at: datetime = Field(..., description="Replay completion timestamp.")
    correlation_id_chain: list[UUID] = Field(
        default_factory=list,
        description="Correlation IDs recovered from replayed envelopes.",
    )
    failed_event_offsets: list[int] = Field(
        default_factory=list,
        description="Offsets that failed canonical envelope deserialization.",
    )


__all__ = ["ModelKafkaReplayOutput"]
