"""Ledger query model for searching ledger entries.

This module defines the query parameters for ledger search operations.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelLedgerQuery(BaseModel):
    """Query parameters for ledger searches.

    All filter fields are optional - omitting a field means no filtering
    on that dimension. Multiple filters are combined with AND logic.

    The limit and offset fields enable pagination through large result sets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID | None = Field(
        default=None,
        description="Filter by correlation ID for request tracing",
    )
    event_type: str | None = Field(
        default=None,
        description="Filter by event type identifier",
    )
    topic: str | None = Field(
        default=None,
        description="Filter by Kafka topic",
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
