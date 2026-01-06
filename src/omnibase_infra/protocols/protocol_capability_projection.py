# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Capability-Based Projection Queries.

This module defines the protocol interface for querying registration projections
by capability fields. Enables fast capability-based node discovery using
GIN-indexed array queries.

Related Tickets:
    - OMN-1134: Registry Projection Extensions for Capabilities
    - OMN-1135: CapabilityQueryService (consumer of this protocol)

Example:
    >>> class CapabilityQueryService:
    ...     def __init__(self, reader: ProtocolCapabilityProjection):
    ...         self._reader = reader
    ...
    ...     async def find_postgres_adapters(self) -> list[ModelRegistrationProjection]:
    ...         return await self._reader.get_by_capability_tag("postgres.storage")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.models.projection.model_registration_projection import (
        ModelRegistrationProjection,
    )


@runtime_checkable
class ProtocolCapabilityProjection(Protocol):
    """Protocol for capability-based projection queries.

    Defines the interface for querying registration projections by capability
    fields. Implementations use GIN-indexed PostgreSQL array queries for
    efficient lookups.

    Methods:
        get_by_capability_tag: Find nodes by capability tag
        get_by_intent_type: Find nodes by intent type they handle
        get_by_protocol: Find nodes implementing a specific protocol
        get_by_contract_type: Find nodes by contract type (effect, compute, etc.)

    Example Implementation:
        class ProjectionReaderRegistration(ProtocolCapabilityProjection):
            async def get_by_capability_tag(
                self, tag: str
            ) -> list[ModelRegistrationProjection]:
                return await self._query(
                    "SELECT * FROM registration_projections "
                    "WHERE capability_tags @> ARRAY[$1]",
                    tag,
                )

    Query Performance:
        All methods use GIN-indexed array queries which provide:
        - O(log n) lookup time for single-element containment
        - Efficient multi-tag queries using array operators
        - Automatic index selection by PostgreSQL query planner
    """

    async def get_by_capability_tag(
        self, tag: str
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations with the specified capability tag.

        Uses GIN index on capability_tags column for efficient lookup.

        Args:
            tag: The capability tag to search for (e.g., "postgres.storage")

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_capability_tag("kafka.consumer")
            >>> for adapter in adapters:
            ...     print(f"{adapter.entity_id}: {adapter.node_type}")
        """
        ...

    async def get_by_intent_type(
        self, intent_type: str
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations that handle the specified intent type.

        Uses GIN index on intent_types column for efficient lookup.

        Args:
            intent_type: The intent type to search for (e.g., "postgres.upsert")

        Returns:
            List of matching registration projections

        Example:
            >>> handlers = await reader.get_by_intent_type("postgres.query")
            >>> for handler in handlers:
            ...     print(f"Can handle postgres.query: {handler.entity_id}")
        """
        ...

    async def get_by_protocol(
        self, protocol_name: str
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations implementing the specified protocol.

        Uses GIN index on protocols column for efficient lookup.

        Args:
            protocol_name: The protocol name (e.g., "ProtocolDatabaseAdapter")

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_protocol("ProtocolEventPublisher")
            >>> print(f"Found {len(adapters)} event publishers")
        """
        ...

    async def get_by_contract_type(
        self, contract_type: str
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations of the specified contract type.

        Uses B-tree index on contract_type column for efficient lookup.

        Args:
            contract_type: The contract type ("effect", "compute", "reducer", "orchestrator")

        Returns:
            List of matching registration projections

        Example:
            >>> effects = await reader.get_by_contract_type("effect")
            >>> print(f"Found {len(effects)} effect nodes")
        """
        ...

    async def get_by_capability_tags_all(
        self, tags: list[str]
    ) -> list[ModelRegistrationProjection]:
        """Find registrations with ALL specified capability tags.

        Uses GIN index with @> (contains all) operator.

        Args:
            tags: List of capability tags that must all be present

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_capability_tags_all(
            ...     ["postgres.storage", "transactions"]
            ... )
        """
        ...

    async def get_by_capability_tags_any(
        self, tags: list[str]
    ) -> list[ModelRegistrationProjection]:
        """Find registrations with ANY of the specified capability tags.

        Uses GIN index with && (overlaps) operator.

        Args:
            tags: List of capability tags, at least one must be present

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_capability_tags_any(
            ...     ["postgres.storage", "mysql.storage", "sqlite.storage"]
            ... )
        """
        ...


__all__: list[str] = ["ProtocolCapabilityProjection"]
