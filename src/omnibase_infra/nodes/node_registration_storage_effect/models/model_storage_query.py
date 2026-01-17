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

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_node_kind import EnumNodeKind


class ModelStorageQuery(BaseModel):
    """Query model for registration storage operations.

    Defines filters and pagination for querying registration records.
    All filter fields are optional - omitting a filter means "match all".

    Immutability:
        This model uses frozen=True to ensure queries are immutable
        once created, enabling safe reuse and caching.

    Attributes:
        node_id: Filter by specific node ID (exact match).
        node_type: Filter by node type (EnumNodeKind).
        capability_filter: Filter by capability name (contains match).
        limit: Maximum number of records to return.
        offset: Number of records to skip for pagination.

    Example:
        >>> # Query all EFFECT nodes
        >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
        >>> query = ModelStorageQuery(node_type=EnumNodeKind.EFFECT, limit=100)

        >>> # Query specific node
        >>> query = ModelStorageQuery(node_id=some_uuid)

        >>> # Query by capability with pagination
        >>> query = ModelStorageQuery(
        ...     capability_filter="registration.storage",
        ...     limit=50,
        ...     offset=100,
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID | None = Field(
        default=None,
        description="Filter by specific node ID (exact match)",
    )
    node_type: EnumNodeKind | None = Field(
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

    def is_single_record_query(self) -> bool:
        """Check if this query targets a single record by node_id.

        Returns:
            True if node_id is specified, indicating a single-record lookup.
        """
        return self.node_id is not None


__all__ = ["ModelStorageQuery"]
