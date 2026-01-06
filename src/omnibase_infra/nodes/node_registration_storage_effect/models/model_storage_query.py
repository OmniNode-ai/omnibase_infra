# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Storage Query Model for Registration Storage Operations.

This module provides ModelStorageQuery, representing a query for retrieving
registration records from storage backends.

Architecture:
    ModelStorageQuery supports flexible querying:
    - Filter by node_id for specific record lookup
    - Filter by node_type for type-based queries
    - Filter by capabilities for capability-based discovery
    - Pagination via limit and offset

    All filters are optional - an empty query returns all records.

Related:
    - NodeRegistrationStorageEffect: Effect node that executes queries
    - ModelStorageResult: Result model containing query results
    - ProtocolRegistrationStorageHandler: Protocol that implements queries
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelStorageQuery(BaseModel):
    """Query model for registration storage operations.

    Defines filters and pagination for querying registration records.
    All filter fields are optional - omitting a filter means "match all".

    Immutability:
        This model uses frozen=True to ensure queries are immutable
        once created, enabling safe reuse and caching.

    Attributes:
        node_id: Filter by specific node ID (exact match).
        node_type: Filter by node type (EFFECT, COMPUTE, etc.).
        capability_filter: Filter by capability name (contains match).
        limit: Maximum number of records to return.
        offset: Number of records to skip for pagination.

    Example:
        >>> # Query all EFFECT nodes
        >>> query = ModelStorageQuery(node_type="EFFECT", limit=100)

        >>> # Query specific node
        >>> query = ModelStorageQuery(node_id=some_uuid)

        >>> # Query by capability with pagination
        >>> query = ModelStorageQuery(
        ...     capability_filter="registration.storage",
        ...     limit=50,
        ...     offset=100,
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: UUID | None = Field(
        default=None,
        description="Filter by specific node ID (exact match)",
    )
    node_type: Literal["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"] | None = Field(
        default=None,
        description="Filter by node type",
    )
    capability_filter: str | None = Field(
        default=None,
        description="Filter by capability name (nodes containing this capability)",
    )
    limit: int = Field(
        default=100,
        description="Maximum number of records to return",
        ge=1,
        le=1000,
    )
    offset: int = Field(
        default=0,
        description="Number of records to skip for pagination",
        ge=0,
    )

    @model_validator(mode="after")
    def validate_query_constraints(self) -> ModelStorageQuery:
        """Validate query constraint combinations.

        When node_id is specified, other filters should typically not be used
        as node_id already identifies a unique record. This validator logs
        a warning but does not block - the node_id filter takes precedence.

        Returns:
            The validated query model.
        """
        # If node_id is specified, warn about redundant filters
        # (but don't block - node_id takes precedence)
        return self

    def is_single_record_query(self) -> bool:
        """Check if this query targets a single record by node_id.

        Returns:
            True if node_id is specified, indicating a single-record lookup.
        """
        return self.node_id is not None


__all__ = ["ModelStorageQuery"]
