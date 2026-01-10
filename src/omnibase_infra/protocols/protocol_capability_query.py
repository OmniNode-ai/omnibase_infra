# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Capability-Based Node Query Service.

Defines the protocol interface for querying nodes by capability rather than
by name. This enables capability-based auto-configuration where consumers
declare what they need, not who provides it.

Core Principle: "I'm interested in what you do, not what you are."

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery
    - OMN-1134: Registry Projection Extensions for Capabilities

Example:
    >>> class MyCapabilityQuery:
    ...     async def find_nodes_by_capability(
    ...         self, capability: str, ...
    ...     ) -> list[ModelRegistrationProjection]:
    ...         # Query registry for nodes with this capability
    ...         ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumRegistrationState
    from omnibase_infra.models.discovery.model_dependency_spec import (
        ModelDependencySpec,
    )
    from omnibase_infra.models.projection.model_registration_projection import (
        ModelRegistrationProjection,
    )


@runtime_checkable
class ProtocolCapabilityQuery(Protocol):
    """Protocol for capability-based node queries.

    Defines the interface for querying registration projections by capability
    attributes (capability tags, intent types, protocols) rather than by name.

    This protocol enables loose coupling between consumers and providers:
    consumers declare what capabilities they need, and the system discovers
    which nodes provide them.

    Canonical Implementation:
        The reference implementation of this protocol is
        ``ServiceCapabilityQuery`` in:

            ``omnibase_infra/services/service_capability_query.py``

        This implementation provides:
        - Delegation to ProjectionReaderRegistration for database queries
        - Contract type filtering
        - Dependency resolution from ModelDependencySpec
        - Node selection strategies (first, random, round-robin, least-loaded)

    Methods:
        find_nodes_by_capability: Find nodes by capability tag.
        find_nodes_by_intent_type: Find effect nodes by intent type handled.
        find_nodes_by_protocol: Find nodes implementing a specific protocol.
        resolve_dependency: Resolve a ModelDependencySpec to a concrete node.

    Example:
        >>> from omnibase_infra.services import ServiceCapabilityQuery
        >>> from omnibase_infra.projectors import ProjectionReaderRegistration
        >>>
        >>> reader = ProjectionReaderRegistration(pool)
        >>> query = ServiceCapabilityQuery(reader)
        >>>
        >>> # Find all Kafka consumers
        >>> kafka_nodes = await query.find_nodes_by_capability("kafka.consumer")
        >>>
        >>> # Find nodes that handle postgres.upsert intent
        >>> handlers = await query.find_nodes_by_intent_type("postgres.upsert")
        >>>
        >>> # Resolve a dependency specification
        >>> spec = ModelDependencySpec(
        ...     name="storage",
        ...     type="node",
        ...     capability="postgres.storage",
        ... )
        >>> node = await query.resolve_dependency(spec)

    See Also:
        - ``ProtocolCapabilityProjection``: Lower-level projection query protocol
        - ``ModelDependencySpec``: Dependency specification model
        - ``ModelRegistrationProjection``: The projection model returned by queries
        - ``EnumRegistrationState``: FSM states for optional state filtering
    """

    async def find_nodes_by_capability(
        self,
        capability: str,
        contract_type: str | None = None,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find nodes that provide a specific capability.

        Queries the registry for nodes with the specified capability tag.
        Optionally filters by contract type and registration state.

        Args:
            capability: Capability tag to search for (e.g., "postgres.storage",
                "kafka.consumer", "consul.registration").
            contract_type: Optional filter by contract type ("effect", "compute",
                "reducer", "orchestrator").
            state: Registration state filter. When None (default), filters to
                EnumRegistrationState.ACTIVE to return only actively registered
                nodes. Pass an explicit EnumRegistrationState value to query
                nodes in other states (e.g., PENDING_REGISTRATION, LIVENESS_EXPIRED).
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            List of matching registration projections. Empty list if no matches.

        Example:
            >>> nodes = await query.find_nodes_by_capability(
            ...     "postgres.storage",
            ...     contract_type="effect",
            ...     state=EnumRegistrationState.ACTIVE,
            ...     correlation_id=uuid4(),
            ... )
            >>> for node in nodes:
            ...     print(f"Found: {node.entity_id}")
        """
        ...

    async def find_nodes_by_intent_type(
        self,
        intent_type: str,
        contract_type: str = "effect",
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find effect nodes that handle a specific intent type.

        Queries the registry for nodes that can handle the specified intent type.
        Typically used to find effect nodes that execute specific intents.

        Args:
            intent_type: Intent type to search for (e.g., "postgres.upsert",
                "consul.register", "kafka.publish").
            contract_type: Filter by contract type (default: "effect").
                Intents are typically handled by effect nodes.
            state: Registration state filter. When None (default), filters to
                EnumRegistrationState.ACTIVE to return only actively registered
                nodes. Pass an explicit EnumRegistrationState value to query
                nodes in other states (e.g., PENDING_REGISTRATION, LIVENESS_EXPIRED).
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            List of matching registration projections. Empty list if no matches.

        Example:
            >>> handlers = await query.find_nodes_by_intent_type(
            ...     "postgres.query",
            ...     contract_type="effect",
            ...     state=EnumRegistrationState.ACTIVE,
            ...     correlation_id=uuid4(),
            ... )
            >>> for handler in handlers:
            ...     print(f"Can handle postgres.query: {handler.entity_id}")
        """
        ...

    async def find_nodes_by_protocol(
        self,
        protocol: str,
        contract_type: str | None = None,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find nodes implementing a specific protocol.

        Queries the registry for nodes that implement the specified protocol.
        Useful for finding nodes that satisfy interface requirements.

        Args:
            protocol: Protocol name to search for (e.g., "ProtocolEventPublisher",
                "ProtocolReducer", "ProtocolDatabaseAdapter").
            contract_type: Optional filter by contract type ("effect", "compute",
                "reducer", "orchestrator").
            state: Registration state filter. When None (default), filters to
                EnumRegistrationState.ACTIVE to return only actively registered
                nodes. Pass an explicit EnumRegistrationState value to query
                nodes in other states (e.g., PENDING_REGISTRATION, LIVENESS_EXPIRED).
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            List of matching registration projections. Empty list if no matches.

        Example:
            >>> adapters = await query.find_nodes_by_protocol(
            ...     "ProtocolEventPublisher",
            ...     state=EnumRegistrationState.ACTIVE,
            ...     correlation_id=uuid4(),
            ... )
            >>> print(f"Found {len(adapters)} event publishers")
        """
        ...

    async def resolve_dependency(
        self,
        dependency_spec: ModelDependencySpec,
        correlation_id: UUID | None = None,
    ) -> ModelRegistrationProjection | None:
        """Resolve a dependency specification to a concrete node.

        Uses the dependency specification to query the registry and select
        a node that matches the specified capability criteria.

        Resolution Strategy:
            1. If capability specified -> find by capability
            2. If intent_types specified -> find by intent type
            3. If protocol specified -> find by protocol
            4. Apply selection strategy to choose among matches

        Args:
            dependency_spec: Dependency specification from contract.
                Contains capability filters and selection strategy.
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            Resolved node registration, or None if not found.

        Example:
            >>> spec = ModelDependencySpec(
            ...     name="storage",
            ...     type="node",
            ...     capability="postgres.storage",
            ...     contract_type="effect",
            ...     selection_strategy="round_robin",
            ... )
            >>> node = await query.resolve_dependency(spec, correlation_id=uuid4())
            >>> if node:
            ...     print(f"Resolved to: {node.entity_id}")
        """
        ...


__all__: list[str] = ["ProtocolCapabilityQuery"]
