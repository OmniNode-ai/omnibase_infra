# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Single-table row count model for projection health probes (OMN-5653).

Represents the row count for one table as reported by pg_stat_user_tables.

Related Tickets:
    - OMN-5653: Wire row count probe
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelTableRowCount(BaseModel):
    """Row count for a single table from pg_stat_user_tables."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    relation_key: str = Field(
        ..., description="Relation (table) identifier from pg_stat_user_tables."
    )
    schema_label: str = Field(
        default="public",
        description="Schema label containing the table.",
    )
    n_live_tup: int = Field(
        ..., description="Approximate live row count from pg_stat_user_tables."
    )
    is_empty: bool = Field(..., description="Whether the table has zero live rows.")


__all__ = [
    "ModelTableRowCount",
]
