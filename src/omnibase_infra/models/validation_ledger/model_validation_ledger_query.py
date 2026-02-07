# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger query model for filtering and pagination.

This module defines query parameters for validation event ledger searches.
All filter fields are optional - omitting a field means no filtering on
that dimension. Multiple filters are combined with AND logic.

Ticket: OMN-1908
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelValidationLedgerQuery(BaseModel):
    """Query parameters for validation event ledger searches.

    All filter fields are optional - omitting a field means no filtering
    on that dimension. Multiple filters are combined with AND logic.

    The limit and offset fields enable pagination through large result sets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID | None = Field(
        default=None,
        description="Filter by validation run ID",
    )
    repo_id: str | None = Field(
        default=None,
        description="Filter by repository identifier",
    )
    event_type: str | None = Field(
        default=None,
        description="Filter by event type discriminator",
    )
    start_time: datetime | None = Field(
        default=None,
        description="Filter events at or after this timestamp",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Filter events before this timestamp",
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of entries to return",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of entries to skip for pagination",
    )


__all__ = ["ModelValidationLedgerQuery"]
