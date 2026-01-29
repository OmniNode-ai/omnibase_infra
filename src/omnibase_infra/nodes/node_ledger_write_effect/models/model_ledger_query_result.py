"""Ledger query result model for search responses.

This module defines the result structure returned by ledger search operations.
"""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_ledger_write_effect.models.model_ledger_entry import (
    ModelLedgerEntry,
)
from omnibase_infra.nodes.node_ledger_write_effect.models.model_ledger_query import (
    ModelLedgerQuery,
)


class ModelLedgerQueryResult(BaseModel):
    """Result of a ledger query operation.

    Contains the matching entries along with pagination metadata
    and the original query for reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: list[ModelLedgerEntry] = Field(
        ...,
        description="List of matching ledger entries",
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
    query: ModelLedgerQuery = Field(
        ...,
        description="The query parameters used to generate this result",
    )
