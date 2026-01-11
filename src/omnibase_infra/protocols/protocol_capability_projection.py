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
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumRegistrationState
    from omnibase_infra.models.projection.model_registration_projection import (
        ContractType,
        ModelRegistrationProjection,
    )


@runtime_checkable
class ProtocolCapabilityProjection(Protocol):
    """Protocol for capability-based projection queries.

    Defines the interface for querying registration projections by capability
    fields. Implementations use GIN-indexed PostgreSQL array queries for
    efficient lookups.

    Canonical Implementation:
        The reference implementation of this protocol is
        ``ProjectionReaderRegistration`` in:

            ``omnibase_infra/projectors/projection_reader_registration.py``

        This implementation provides:
        - asyncpg-based PostgreSQL queries with connection pooling
        - Circuit breaker resilience (MixinAsyncCircuitBreaker)
        - Parameterized queries for SQL injection protection
        - Consistent error handling with InfraConnectionError/InfraTimeoutError
        - Optional state filtering on all capability query methods

    Methods:
        get_by_capability_tag: Find nodes by capability tag (e.g., "postgres.storage").
            Uses GIN index on capability_tags column.
        get_by_intent_type: Find nodes by intent type they handle (e.g., "postgres.query").
            Uses GIN index on intent_types column.
        get_by_protocol: Find nodes implementing a specific protocol
            (e.g., "ProtocolEventPublisher"). Uses GIN index on protocols column.
        get_by_contract_type: Find nodes by contract type (effect, compute, reducer,
            orchestrator). Uses B-tree index on contract_type column.
        get_by_capability_tags_all: Find nodes with ALL specified tags.
            Uses GIN index with @> (contains all) operator.
        get_by_capability_tags_any: Find nodes with ANY of the specified tags.
            Uses GIN index with && (overlaps) operator.

    Query Performance:
        All methods use GIN-indexed array queries which provide:
        - O(log n) lookup time for single-element containment
        - Efficient multi-tag queries using array operators
        - Automatic index selection by PostgreSQL query planner

    Example:
        >>> import asyncpg
        >>> from omnibase_infra.projectors import ProjectionReaderRegistration
        >>> from omnibase_infra.enums import EnumRegistrationState
        >>>
        >>> # Create reader with connection pool
        >>> pool = await asyncpg.create_pool(dsn)
        >>> reader = ProjectionReaderRegistration(pool)
        >>>
        >>> # Find all active Kafka consumers
        >>> kafka_nodes = await reader.get_by_capability_tag(
        ...     "kafka.consumer",
        ...     state=EnumRegistrationState.ACTIVE,
        ... )
        >>>
        >>> # Find nodes implementing a protocol
        >>> publishers = await reader.get_by_protocol("ProtocolEventPublisher")
        >>>
        >>> # Find effect nodes
        >>> effects = await reader.get_by_contract_type("effect")

    See Also:
        - ``ProtocolProjectionReader``: Base protocol for all projection readers
        - ``ModelRegistrationProjection``: The projection model returned by queries
        - ``EnumRegistrationState``: FSM states for optional state filtering
    """

    async def get_by_capability_tag(
        self,
        tag: str,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations with the specified capability tag.

        Uses GIN index on capability_tags column for efficient lookup.

        Args:
            tag: The capability tag to search for (e.g., "postgres.storage")
            state: Optional state filter (e.g., EnumRegistrationState.ACTIVE)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_capability_tag("kafka.consumer")
            >>> for adapter in adapters:
            ...     print(f"{adapter.entity_id}: {adapter.node_type}")
        """
        ...

    async def get_by_intent_type(
        self,
        intent_type: str,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations that handle the specified intent type.

        Uses GIN index on intent_types column for efficient lookup.

        Args:
            intent_type: The intent type to search for (e.g., "postgres.upsert")
            state: Optional state filter (e.g., EnumRegistrationState.ACTIVE)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            List of matching registration projections

        Example:
            >>> handlers = await reader.get_by_intent_type("postgres.query")
            >>> for handler in handlers:
            ...     print(f"Can handle postgres.query: {handler.entity_id}")
        """
        ...

    async def get_by_protocol(
        self,
        protocol_name: str,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations implementing the specified protocol.

        Uses GIN index on protocols column for efficient lookup.

        Args:
            protocol_name: The protocol name (e.g., "ProtocolDatabaseAdapter")
            state: Optional state filter (e.g., EnumRegistrationState.ACTIVE)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_protocol("ProtocolEventPublisher")
            >>> print(f"Found {len(adapters)} event publishers")
        """
        ...

    async def get_by_contract_type(
        self,
        contract_type: ContractType,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find all registrations of the specified contract type.

        Uses B-tree index on contract_type column for efficient lookup.

        Args:
            contract_type: The contract type. Must be one of: "effect", "compute",
                "reducer", or "orchestrator"
            state: Optional state filter (e.g., EnumRegistrationState.ACTIVE)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            List of matching registration projections

        Example:
            >>> effects = await reader.get_by_contract_type("effect")
            >>> print(f"Found {len(effects)} effect nodes")
        """
        ...

    async def get_by_capability_tags_all(
        self,
        tags: list[str],
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find registrations with ALL specified capability tags.

        Uses GIN index with @> (contains all) operator.

        Args:
            tags: List of capability tags that must all be present
            state: Optional state filter (e.g., EnumRegistrationState.ACTIVE)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_capability_tags_all(
            ...     ["postgres.storage", "transactions"]
            ... )
        """
        ...

    async def get_by_capability_tags_any(
        self,
        tags: list[str],
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find registrations with ANY of the specified capability tags.

        Uses GIN index with && (overlaps) operator.

        Args:
            tags: List of capability tags, at least one must be present
            state: Optional state filter (e.g., EnumRegistrationState.ACTIVE)
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            List of matching registration projections

        Example:
            >>> adapters = await reader.get_by_capability_tags_any(
            ...     ["postgres.storage", "mysql.storage", "sqlite.storage"]
            ... )
        """
        ...


__all__: list[str] = ["ProtocolCapabilityProjection"]
