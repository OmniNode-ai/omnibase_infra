# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection-row model for topic_migration_projection (OMN-12623).

Materializes the current FSM state of a topic migration plus the idempotency
columns (last_applied_event_id/offset/sequence/partition) modeled on
``registration_projections``. The COMPUTE projection node transforms a
:class:`ModelTopicMigrationLifecycleEvent` into one of these rows; the EFFECT
upserts it into ``topic_migration_projection`` keyed on ``migration_ticket``.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase


class ModelTopicMigrationProjectionRow(BaseModel):
    """A single materialized topic-migration projection row.

    ``current_state`` is the FSM phase the migration is in. The
    ``last_applied_*`` columns make the projection replay-safe: an out-of-order
    or duplicate lifecycle event is rejected by the upsert's sequence guard.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    migration_ticket: str = Field(
        ..., pattern=r"^OMN-\d+$", description="Migration contract ticket (PK)"
    )
    old_topic: str = Field(..., min_length=1, description="Source topic")
    new_topic: str = Field(..., min_length=1, description="Target topic")
    old_consumer_group: str = Field(..., min_length=1, description="Source group")
    new_consumer_group: str = Field(..., min_length=1, description="Target group")
    current_state: EnumMigrationPhase = Field(
        ..., description="Current FSM phase of the migration"
    )
    new_topic_provisioned: bool = Field(
        default=False, description="Whether the new topic has been provisioned"
    )
    retirement_allowed: bool = Field(
        default=False, description="Whether the drain-proof gate permits retirement"
    )
    residual_lag: int = Field(
        default=0, ge=0, description="Residual old-topic lag at last transition"
    )

    # Idempotency / ordering columns (mirror registration_projections).
    last_applied_event_id: UUID = Field(
        ..., description="event_id of the last applied lifecycle event"
    )
    last_applied_sequence: int = Field(
        ..., ge=0, description="Sequence of the last applied lifecycle event"
    )
    last_applied_offset: int = Field(
        default=0, ge=0, description="Kafka offset of the last applied event"
    )
    last_applied_partition: str | None = Field(
        default=None, description="Kafka partition of the last applied event"
    )


__all__ = ["ModelTopicMigrationProjectionRow"]
