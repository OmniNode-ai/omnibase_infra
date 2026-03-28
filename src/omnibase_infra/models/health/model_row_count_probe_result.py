# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Row count probe result model for omnidash projection health checks.

Reports tables with zero rows detected via pg_stat_user_tables,
used by the error triage pipeline to flag empty projection tables
that indicate silent data loss.

Related Tickets:
    - OMN-5653: Wire row count probe
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.model_table_row_count import (
    ModelTableRowCount,
)


class ModelRowCountProbeResult(BaseModel):
    """Result of a row count probe across omnidash projection tables.

    Aggregates pg_stat_user_tables data and flags tables with
    zero rows as potential projection failures.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    event_id: UUID = Field(
        default_factory=uuid4, description="Unique probe event identifier."
    )
    db_display_label: str = Field(
        ..., description="Display label for the database that was probed."
    )
    total_tables: int = Field(..., description="Total number of user tables found.")
    empty_tables: int = Field(..., description="Number of tables with zero live rows.")
    populated_tables: int = Field(
        ..., description="Number of tables with at least one live row."
    )
    empty_table_names: list[str] = Field(
        default_factory=list,
        description="Names of tables with zero live rows.",
    )
    table_details: list[ModelTableRowCount] = Field(
        default_factory=list,
        description="Per-table row count details.",
    )
    healthy: bool = Field(
        ...,
        description="True if no unexpected empty tables were detected.",
    )
    probed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the probe was executed.",
    )


__all__ = [
    "ModelRowCountProbeResult",
]
