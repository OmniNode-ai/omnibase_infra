# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Query filter model for injection effectiveness reads.

Defines the filter parameters for querying the injection_effectiveness,
latency_breakdowns, and pattern_hit_rates tables.

Related Tickets:
    - OMN-2078: Golden path: injection metrics + ledger storage
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInjectionEffectivenessQuery(BaseModel):
    """Query filters for injection effectiveness data.

    All fields are optional. When set, they are ANDed together in the
    WHERE clause. When omitted, no filter is applied for that field.

    Attributes:
        session_id: Filter by exact session ID.
        correlation_id: Filter by correlation ID.
        cohort: Filter by A/B test cohort.
        utilization_method: Filter by utilization method.
        start_time: Filter created_at >= start_time.
        end_time: Filter created_at < end_time.
        limit: Maximum rows to return.
        offset: Number of rows to skip for pagination.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID | None = Field(default=None, description="Filter by session ID")
    correlation_id: UUID | None = Field(
        default=None, description="Filter by correlation ID"
    )
    cohort: Literal["control", "treatment"] | None = Field(
        default=None, description="Filter by cohort"
    )
    utilization_method: str | None = Field(
        default=None, description="Filter by utilization method"
    )
    start_time: datetime | None = Field(
        default=None, description="Filter created_at >= start_time"
    )
    end_time: datetime | None = Field(
        default=None, description="Filter created_at < end_time"
    )
    limit: int = Field(default=100, ge=1, le=10000, description="Max rows to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


__all__ = ["ModelInjectionEffectivenessQuery"]
