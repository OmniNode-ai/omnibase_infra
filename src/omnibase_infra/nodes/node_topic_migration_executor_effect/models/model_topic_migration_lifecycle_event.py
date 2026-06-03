# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic-migration lifecycle event model (OMN-12623).

One event per advanced migration phase, emitted by the executor. The event is
the durable fact the ``topic_migration_projection`` reducer materializes; its
``event_id`` + ``sequence`` provide the idempotency keys for replay-safe
projection.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase


class ModelTopicMigrationLifecycleEvent(BaseModel):
    """A single lifecycle transition emitted by the migration executor."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_id: UUID = Field(..., description="Unique id of this lifecycle event")
    correlation_id: UUID = Field(..., description="Trace correlation id")
    migration_ticket: str = Field(
        ..., pattern=r"^OMN-\d+$", description="Migration contract ticket id"
    )
    old_topic: str = Field(..., min_length=1, description="Source topic")
    new_topic: str = Field(..., min_length=1, description="Target topic")
    old_consumer_group: str = Field(..., min_length=1, description="Source group")
    new_consumer_group: str = Field(..., min_length=1, description="Target group")
    phase: EnumMigrationPhase = Field(
        ..., description="Phase the migration entered with this event"
    )
    sequence: int = Field(
        ...,
        ge=0,
        description="Monotonic sequence number for ordering/idempotency",
    )
    new_topic_provisioned: bool = Field(
        default=False,
        description="Whether the new topic was provisioned by this transition",
    )
    retirement_allowed: bool = Field(
        default=False,
        description="Whether the drain-proof gate permitted old-group retirement",
    )
    residual_lag: int = Field(
        default=0,
        ge=0,
        description="Residual old-topic lag observed at this transition",
    )
    detail: str = Field(default="", description="Human-readable transition detail")


__all__ = ["ModelTopicMigrationLifecycleEvent"]
