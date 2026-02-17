# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Invalidation event model for effectiveness measurement updates.

Emitted when new measurement data is written to effectiveness tables,
enabling downstream consumers (dashboards, caches) to refresh.

Related Tickets:
    - OMN-2303: Activate effectiveness consumer and populate measurement tables
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The three known effectiveness measurement tables
_VALID_TABLES: frozenset[str] = frozenset(
    {
        "injection_effectiveness",
        "latency_breakdowns",
        "pattern_hit_rates",
    }
)


class ModelEffectivenessInvalidationEvent(BaseModel):
    """Event emitted when effectiveness measurement data changes.

    Published to Kafka after successful writes to any of the three
    effectiveness tables. Downstream consumers (WebSocket servers,
    dashboard APIs, caches) subscribe to this topic to trigger
    data refresh.

    Attributes:
        event_type: Event type discriminator.
        correlation_id: Correlation ID for tracing.
        tables_affected: Which tables were updated in this write.
        rows_written: Number of rows written in the batch.
        source: Origin of the data (kafka_consumer or batch_compute).
        emitted_at: Timestamp when the event was created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: Literal["effectiveness_data_changed"] = Field(
        default="effectiveness_data_changed",
        description="Event type discriminator",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for tracing",
    )
    tables_affected: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Names of tables that were updated. Possible values: "
            "injection_effectiveness, latency_breakdowns, pattern_hit_rates"
        ),
    )
    rows_written: int = Field(
        ...,
        ge=0,
        description="Total number of rows written across all affected tables",
    )
    source: Literal["kafka_consumer", "batch_compute"] = Field(
        ...,
        description="Origin of the data write",
    )
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when this event was created",
    )

    @field_validator("tables_affected")
    @classmethod
    def _validate_tables_affected(
        cls,
        v: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate that every entry is a known effectiveness table."""
        invalid = set(v) - _VALID_TABLES
        if invalid:
            msg = (
                f"Unknown table(s): {sorted(invalid)}. Allowed: {sorted(_VALID_TABLES)}"
            )
            raise ValueError(msg)
        return v
