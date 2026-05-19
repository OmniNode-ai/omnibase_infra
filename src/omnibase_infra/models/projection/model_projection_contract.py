# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime contract for CQRS projections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_degraded_behavior import EnumDegradedBehavior
from omnibase_infra.models.projection.model_cursor_contract import ModelCursorContract


class ModelProjectionContract(BaseModel):
    """Declares freshness SLA, ordering, and degraded semantics for a projection."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        populate_by_name=True,
    )

    projection_key: str = Field(
        ...,
        alias="projection_name",
        min_length=1,
        description="Unique identifier for this projection.",
    )
    source_topics: tuple[str, ...] = Field(
        ...,
        description="Kafka topics this projection consumes.",
    )
    schema_model: str = Field(
        ...,
        min_length=1,
        description="Fully-qualified Pydantic model name for projection rows.",
    )
    freshness_sla_seconds: int = Field(
        ...,
        gt=0,
        description="Maximum acceptable lag in seconds before the projection is stale.",
    )
    freshness_field: str = Field(
        ...,
        min_length=1,
        description="Column checked to determine projection staleness.",
    )
    freshness_source_table: str = Field(
        ...,
        min_length=1,
        description="Table queried when checking freshness_field.",
    )
    degraded_semantics: EnumDegradedBehavior = Field(
        ...,
        description=(
            "Behaviour when projection freshness SLA is breached. No default; "
            "must be explicit."
        ),
    )
    cursor: ModelCursorContract = Field(
        ...,
        description="Cursor mechanism for this projection.",
    )
    ordering_contract_ref: str | None = Field(
        default=None,
        description="Name of the ordering contract constant, if ordering is defined.",
    )

    @property
    def projection_name(self) -> str:
        """Backwards-compatible name used by projection registry callers."""
        return self.projection_key


__all__ = ["ModelProjectionContract"]
