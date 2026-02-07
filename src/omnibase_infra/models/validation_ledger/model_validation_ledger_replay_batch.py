# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Validation ledger replay batch model for paginated query results.

This module defines the result structure returned by validation event ledger
queries, including entries, total count, and pagination metadata.

Ticket: OMN-1908
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation_ledger.model_validation_ledger_entry import (
    ModelValidationLedgerEntry,
)
from omnibase_infra.models.validation_ledger.model_validation_ledger_query import (
    ModelValidationLedgerQuery,
)


class ModelValidationLedgerReplayBatch(BaseModel):
    """Result of a validation event ledger query.

    Contains the matching entries along with pagination metadata
    and the original query for reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: list[ModelValidationLedgerEntry] = Field(
        ...,
        description="List of matching validation ledger entries",
    )
    total_count: int = Field(
        ...,
        ge=0,
        description="Total number of entries matching the query (before pagination)",
    )
    has_more: bool = Field(
        ...,
        description="True if more entries exist beyond the current page",
    )
    query: ModelValidationLedgerQuery = Field(
        ...,
        description="The query parameters used to generate this result",
    )


__all__ = ["ModelValidationLedgerReplayBatch"]
