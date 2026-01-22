# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Capability Query Service.

Provides high-level service for querying the registry by capability rather
than by name. This enables capability-based auto-configuration where nodes
declare what they need, not who provides it.

Core Principle: "I'm interested in what you do, not what you are."

Coroutine Safety:
    This service uses MixinAsyncCircuitBreaker for coroutine-safe circuit
    breaker protection. The circuit breaker provides defense-in-depth on top
    of the underlying ProjectionReaderRegistration's circuit breaker.

Circuit Breaker:
    This service implements its own circuit breaker as a defense-in-depth layer.
    While ProjectionReaderRegistration also has circuit breakers, having them
    at the service layer provides:
    - Service-level failure isolation
    - Independent failure thresholds tuned for query patterns
    - Protection against issues beyond database connectivity

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery for capability-based discovery
    - OMN-1134: Registry Projection Extensions for Capabilities

Example:
    >>> from omnibase_infra.services import ServiceCapabilityQuery
    >>> from omnibase_infra.projectors import ProjectionReaderRegistration
    >>>
    >>> reader = ProjectionReaderRegistration(pool)
    >>> query = ServiceCapabilityQuery(reader)
    >>>
    >>> # Find all postgres storage providers
    >>> nodes = await query.find_nodes_by_capability("postgres.storage")
    >>>
    >>> # Resolve a dependency spec
    >>> spec = ModelDependencySpec(name="db", type="node", capability="postgres.storage")
    >>> node = await query.resolve_dependency(spec)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import (
    EnumInfraTransportType,
    EnumRegistrationState,
    EnumSelectionStrategy,
)
from omnibase_infra.errors import ProtocolConfigurationError, RuntimeHostError
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.discovery.model_dependency_spec import ModelDependencySpec
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.services.service_node_selector import ServiceNodeSelector

if TYPE_CHECKING:
    from omnibase_infra.projectors.projection_reader_registration import (
        ProjectionReaderRegistration,
    )

logger = logging.getLogger(__name__)


class ServiceCapabilityQuery(MixinAsyncCircuitBreaker):
    """Queries registry for nodes by capability, not by name.

    Core Principle: "I'm interested in what you do, not what you are."

    This service wraps ProjectionReaderRegistration to provide high-level
    capability-based discovery. Instead of hardcoding module paths, consumers
    can declare what capabilities they need and the system discovers which
    nodes provide them.

    Query Methods:
        - find_nodes_by_capability: Find by capability tag
        - find_nodes_by_intent_type: Find by single intent type handled
        - find_nodes_by_intent_types: Find by multiple intent types (bulk query)
        - find_nodes_by_protocol: Find by protocol implemented

    Dependency Resolution:
        The resolve_dependency method takes a ModelDependencySpec and:
        1. Determines the discovery strategy (capability/intent/protocol)
        2. Queries the registry for matching nodes
        3. Applies the selection strategy to choose one node

    Circuit Breaker:
        This service implements MixinAsyncCircuitBreaker for defense-in-depth
        protection. While ProjectionReaderRegistration has its own circuit
        breakers, the service-level circuit breaker provides:
        - Independent failure isolation at the service layer
        - Service-specific failure thresholds (5 failures, 60s reset)
        - Protection against issues beyond database connectivity

    Design Notes:
        - All queries delegate to ProjectionReaderRegistration
        - Service-level circuit breaker provides defense-in-depth
        - Node selection is handled by ServiceNodeSelector
        - Round-robin state is maintained per-service instance

    Example:
        >>> reader = ProjectionReaderRegistration(pool)
        >>> query = ServiceCapabilityQuery(reader)
        >>>
        >>> # Find active Kafka consumers
        >>> nodes = await query.find_nodes_by_capability(
        ...     "kafka.consumer",
        ...     state=EnumRegistrationState.ACTIVE,
        ... )
        >>>
        >>> # Find nodes that handle postgres.upsert intent
        >>> handlers = await query.find_nodes_by_intent_type(
        ...     "postgres.upsert",
        ...     contract_type="effect",
        ... )

    Raises:
        InfraConnectionError: If database connection fails
        InfraTimeoutError: If query times out
        InfraUnavailableError: If circuit breaker is open
        RuntimeHostError: For other database errors

    Note:
        Dependency Injection Pattern: This service is a leaf infrastructure
        service that receives its dependencies directly via constructor
        parameters rather than resolving them from a container. Unlike
        orchestrators that use ``container.service_registry.resolve_service()``
        to obtain services dynamically, leaf services like this one are
        instantiated with concrete dependencies (projection_reader, node_selector)
        and are themselves resolved by higher-level components.
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        container: ModelONEXContainer | None = None,
        node_selector: ServiceNodeSelector | None = None,
    ) -> None:
        """Initialize the capability query service.

        Args:
            projection_reader: The projection reader for database queries.
                Must be initialized with an asyncpg connection pool.
            container: Optional ONEX container for dependency injection.
                Passed to default ServiceNodeSelector if node_selector is None.
            node_selector: Optional node selector for selection strategies.
                If None, creates a new ServiceNodeSelector instance.

        Example:
            >>> pool = await asyncpg.create_pool(dsn)
            >>> reader = ProjectionReaderRegistration(pool)
            >>> query = ServiceCapabilityQuery(reader)
        """
        # Initialize circuit breaker for defense-in-depth protection
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="capability-query",
            transport_type=EnumInfraTransportType.DATABASE,
        )

        self._projection_reader = projection_reader
        self._container = container
        self._node_selector = node_selector or ServiceNodeSelector(container=container)

    async def find_nodes_by_capability(
        self,
        capability: str,
        contract_type: str | None = None,
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find nodes that provide a specific capability.

        Queries the registry for nodes with the specified capability tag.
        Results can be filtered by contract type and registration state.

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop. The circuit breaker state is
            protected by an asyncio.Lock.

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

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> nodes = await query.find_nodes_by_capability(
            ...     "postgres.storage",
            ...     contract_type="effect",
            ...     state=EnumRegistrationState.ACTIVE,
            ... )
            >>> for node in nodes:
            ...     print(f"Found: {node.entity_id} - {node.node_type}")
        """
        correlation_id = self._ensure_correlation_id(correlation_id)
        state = state or EnumRegistrationState.ACTIVE

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="find_nodes_by_capability",
                correlation_id=correlation_id,
            )

        logger.debug(
            "Finding nodes by capability",
            extra={
                "capability": capability,
                "contract_type": contract_type,
                "state": str(state),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Query by capability tag
            results = await self._projection_reader.get_by_capability_tag(
                tag=capability,
                state=state,
                correlation_id=correlation_id,
            )

            results = self._filter_by_contract_type(results, contract_type)

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Capability query completed",
                extra={
                    "capability": capability,
                    "result_count": len(results),
                    "correlation_id": str(correlation_id),
                },
            )

            return results

        except RuntimeHostError:
            # Record failure and re-raise infrastructure errors as-is
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_capability",
                    correlation_id=correlation_id,
                )
            raise
        except Exception as e:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_capability",
                    correlation_id=correlation_id,
                )
            # Wrap unexpected exceptions with RuntimeHostError
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="find_nodes_by_capability",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Capability query failed: {type(e).__name__}",
                context=context,
                capability=capability,
            ) from e

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

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop. The circuit breaker state is
            protected by an asyncio.Lock.

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

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> handlers = await query.find_nodes_by_intent_type(
            ...     "postgres.query",
            ...     contract_type="effect",
            ...     state=EnumRegistrationState.ACTIVE,
            ... )
            >>> for handler in handlers:
            ...     print(f"Can handle postgres.query: {handler.entity_id}")
        """
        correlation_id = self._ensure_correlation_id(correlation_id)
        state = state or EnumRegistrationState.ACTIVE

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="find_nodes_by_intent_type",
                correlation_id=correlation_id,
            )

        logger.debug(
            "Finding nodes by intent type",
            extra={
                "intent_type": intent_type,
                "contract_type": contract_type,
                "state": str(state),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Query by intent type
            results = await self._projection_reader.get_by_intent_type(
                intent_type=intent_type,
                state=state,
                correlation_id=correlation_id,
            )

            results = self._filter_by_contract_type(results, contract_type)

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Intent type query completed",
                extra={
                    "intent_type": intent_type,
                    "result_count": len(results),
                    "correlation_id": str(correlation_id),
                },
            )

            return results

        except RuntimeHostError:
            # Record failure and re-raise infrastructure errors as-is
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_intent_type",
                    correlation_id=correlation_id,
                )
            raise
        except Exception as e:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_intent_type",
                    correlation_id=correlation_id,
                )
            # Wrap unexpected exceptions with RuntimeHostError
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="find_nodes_by_intent_type",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Intent type query failed: {type(e).__name__}",
                context=context,
                intent_type=intent_type,
            ) from e

    async def find_nodes_by_intent_types(
        self,
        intent_types: list[str],
        contract_type: str = "effect",
        state: EnumRegistrationState | None = None,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Find effect nodes that handle ANY of the specified intent types.

        Bulk query method that retrieves nodes matching any intent type in a single
        database call. This is more efficient than calling find_nodes_by_intent_type
        repeatedly for each intent type.

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop. The circuit breaker state is
            protected by an asyncio.Lock.

        Performance Note:
            This method reduces N database queries to 1 query when resolving
            dependencies with multiple intent types. For N intent types:
            - Previous: N sequential database calls
            - Now: 1 bulk query using SQL array overlap

        Args:
            intent_types: List of intent types to search for (e.g.,
                ["postgres.upsert", "postgres.query", "postgres.delete"]).
            contract_type: Filter by contract type (default: "effect").
                Intents are typically handled by effect nodes.
            state: Registration state filter. When None (default), filters to
                EnumRegistrationState.ACTIVE to return only actively registered
                nodes. Pass an explicit EnumRegistrationState value to query
                nodes in other states (e.g., PENDING_REGISTRATION, LIVENESS_EXPIRED).
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            List of matching registration projections. Empty list if no matches
            or if intent_types list is empty.

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> handlers = await query.find_nodes_by_intent_types(
            ...     ["postgres.query", "postgres.upsert", "postgres.delete"],
            ...     contract_type="effect",
            ...     state=EnumRegistrationState.ACTIVE,
            ... )
            >>> for handler in handlers:
            ...     print(f"Can handle postgres intents: {handler.entity_id}")
        """
        correlation_id = self._ensure_correlation_id(correlation_id)
        if not intent_types:
            return []

        state = state or EnumRegistrationState.ACTIVE

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="find_nodes_by_intent_types",
                correlation_id=correlation_id,
            )

        logger.debug(
            "Finding nodes by intent types (bulk)",
            extra={
                "intent_types": intent_types,
                "intent_count": len(intent_types),
                "contract_type": contract_type,
                "state": str(state),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Query by intent types (bulk)
            results = await self._projection_reader.get_by_intent_types(
                intent_types=intent_types,
                state=state,
                correlation_id=correlation_id,
            )

            results = self._filter_by_contract_type(results, contract_type)

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Intent types query completed (bulk)",
                extra={
                    "intent_types": intent_types,
                    "result_count": len(results),
                    "correlation_id": str(correlation_id),
                },
            )

            return results

        except RuntimeHostError:
            # Record failure and re-raise infrastructure errors as-is
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_intent_types",
                    correlation_id=correlation_id,
                )
            raise
        except Exception as e:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_intent_types",
                    correlation_id=correlation_id,
                )
            # Wrap unexpected exceptions with RuntimeHostError
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="find_nodes_by_intent_types",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Intent types query failed: {type(e).__name__}",
                context=context,
                intent_types=intent_types,
            ) from e

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

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop. The circuit breaker state is
            protected by an asyncio.Lock.

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

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors

        Example:
            >>> adapters = await query.find_nodes_by_protocol(
            ...     "ProtocolEventPublisher",
            ...     state=EnumRegistrationState.ACTIVE,
            ... )
            >>> print(f"Found {len(adapters)} event publishers")
        """
        correlation_id = self._ensure_correlation_id(correlation_id)
        state = state or EnumRegistrationState.ACTIVE

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation="find_nodes_by_protocol",
                correlation_id=correlation_id,
            )

        logger.debug(
            "Finding nodes by protocol",
            extra={
                "protocol": protocol,
                "contract_type": contract_type,
                "state": str(state),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Query by protocol
            results = await self._projection_reader.get_by_protocol(
                protocol_name=protocol,
                state=state,
                correlation_id=correlation_id,
            )

            results = self._filter_by_contract_type(results, contract_type)

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Protocol query completed",
                extra={
                    "protocol": protocol,
                    "result_count": len(results),
                    "correlation_id": str(correlation_id),
                },
            )

            return results

        except RuntimeHostError:
            # Record failure and re-raise infrastructure errors as-is
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_protocol",
                    correlation_id=correlation_id,
                )
            raise
        except Exception as e:
            # Record failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="find_nodes_by_protocol",
                    correlation_id=correlation_id,
                )
            # Wrap unexpected exceptions with RuntimeHostError
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="find_nodes_by_protocol",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Protocol query failed: {type(e).__name__}",
                context=context,
                protocol=protocol,
            ) from e

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
            2. If intent_types specified -> find by intent types (bulk query)
            3. If protocol specified -> find by protocol
            4. Apply selection strategy from spec to choose among matches

        Coroutine Safety:
            This method is coroutine-safe when called concurrently from multiple
            coroutines within the same event loop. Uses circuit-breaker-protected
            query methods internally.

        Args:
            dependency_spec: Dependency specification from contract.
                Contains capability filters and selection strategy.
            correlation_id: Optional correlation ID for distributed tracing.
                When provided, included in all log messages for request tracking.

        Returns:
            Resolved node registration, or None if not found.

        Note:
            Performance: When resolving by intent types, this method uses a bulk
            query that retrieves all matching nodes in a single database call using
            SQL array overlap. This reduces N queries to 1 query regardless of the
            number of intent types in the dependency spec.

        Example:
            >>> spec = ModelDependencySpec(
            ...     name="storage",
            ...     type="node",
            ...     capability="postgres.storage",
            ...     contract_type="effect",
            ...     selection_strategy="round_robin",
            ... )
            >>> node = await query.resolve_dependency(spec)
            >>> if node:
            ...     print(f"Resolved to: {node.entity_id}")
            ... else:
            ...     print("No node found for capability")
        """
        correlation_id = self._ensure_correlation_id(correlation_id)
        logger.debug(
            "Resolving dependency",
            extra={
                "dependency_name": dependency_spec.name,
                "dependency_type": dependency_spec.type,
                "capability": dependency_spec.capability,
                "intent_types": dependency_spec.intent_types,
                "protocol": dependency_spec.protocol,
                "selection_strategy": dependency_spec.selection_strategy,
                "correlation_id": str(correlation_id),
            },
        )

        # Determine state filter
        state = self._parse_state(dependency_spec.state, correlation_id)

        # Find candidates based on discovery strategy
        candidates: list[ModelRegistrationProjection] = []

        if dependency_spec.has_capability_filter():
            # Assert for type narrowing - has_capability_filter guarantees not None
            assert dependency_spec.capability is not None
            candidates = await self.find_nodes_by_capability(
                capability=dependency_spec.capability,
                contract_type=dependency_spec.contract_type,
                state=state,
                correlation_id=correlation_id,
            )

        elif dependency_spec.has_intent_filter():
            # Use bulk query for multiple intent types (single database call)
            intent_types = dependency_spec.intent_types
            if intent_types:
                candidates = await self.find_nodes_by_intent_types(
                    intent_types=intent_types,
                    contract_type=dependency_spec.contract_type or "effect",
                    state=state,
                    correlation_id=correlation_id,
                )

        elif dependency_spec.has_protocol_filter():
            # Assert for type narrowing - has_protocol_filter guarantees not None
            assert dependency_spec.protocol is not None
            candidates = await self.find_nodes_by_protocol(
                protocol=dependency_spec.protocol,
                contract_type=dependency_spec.contract_type,
                state=state,
                correlation_id=correlation_id,
            )

        # No matches found
        if not candidates:
            logger.debug(
                "No candidates found for dependency",
                extra={
                    "dependency_name": dependency_spec.name,
                    "correlation_id": str(correlation_id),
                },
            )
            return None

        # Apply selection strategy
        strategy = self._parse_selection_strategy(
            dependency_spec.selection_strategy, correlation_id
        )
        selected = await self._node_selector.select(
            candidates=candidates,
            strategy=strategy,
            selection_key=dependency_spec.name,
            correlation_id=correlation_id,
        )

        if selected:
            logger.debug(
                "Dependency resolved",
                extra={
                    "dependency_name": dependency_spec.name,
                    "selected_entity_id": str(selected.entity_id),
                    "total_candidates": len(candidates),
                    "strategy": dependency_spec.selection_strategy,
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "No node selected for dependency",
                extra={
                    "dependency_name": dependency_spec.name,
                    "correlation_id": str(correlation_id),
                },
            )

        return selected

    def _ensure_correlation_id(self, correlation_id: UUID | None) -> UUID:
        """Ensure correlation ID is present, generating one if missing.

        Args:
            correlation_id: Optional correlation ID from caller. Accepts UUID
                or None.

        Returns:
            The provided correlation ID, or a newly generated UUID4.
        """
        return correlation_id or uuid4()

    def _parse_state(
        self,
        state_value: EnumRegistrationState | str | None,
        correlation_id: UUID | None = None,
    ) -> EnumRegistrationState:
        """Parse state value to EnumRegistrationState.

        Accepts either an EnumRegistrationState enum value (returned as-is),
        a string representation (parsed to enum), or None (defaults to ACTIVE).

        Args:
            state_value: State as EnumRegistrationState, string (e.g., "ACTIVE", "active"),
                or None. When None, defaults to EnumRegistrationState.ACTIVE.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            EnumRegistrationState value.

        Raises:
            ProtocolConfigurationError: If state string is not a valid state or
                if an unexpected error occurs during parsing.
        """
        # Handle None - default to ACTIVE
        if state_value is None:
            return EnumRegistrationState.ACTIVE

        # If already an enum, return directly
        if isinstance(state_value, EnumRegistrationState):
            return state_value

        # Parse string to enum
        try:
            return EnumRegistrationState(state_value.lower())
        except ValueError as e:
            valid_states = [s.value for s in EnumRegistrationState]
            raise ProtocolConfigurationError(
                f"Invalid registration state '{state_value}'. "
                f"Valid states are: {valid_states}",
                details={
                    "state": state_value,
                    "valid_states": valid_states,
                    "correlation_id": correlation_id,
                },
            ) from e
        except Exception as e:
            # Catch any other unexpected errors (e.g., AttributeError if state_value
            # is an unexpected type that passed type checking)
            raise ProtocolConfigurationError(
                f"Failed to parse registration state: {type(e).__name__}",
                details={
                    "state": str(state_value),
                    "error": str(e),
                    "correlation_id": correlation_id,
                },
            ) from e

    def _parse_selection_strategy(
        self,
        strategy: str | EnumSelectionStrategy,
        correlation_id: UUID | None = None,
    ) -> EnumSelectionStrategy:
        """Parse selection strategy to enum.

        Accepts either an EnumSelectionStrategy enum value (returned as-is)
        or a string representation (parsed to enum).

        Args:
            strategy: Strategy as EnumSelectionStrategy or string
                (e.g., "first", "round_robin").
            correlation_id: Optional correlation ID for tracing.

        Returns:
            EnumSelectionStrategy value.

        Raises:
            ProtocolConfigurationError: If strategy string is not a valid strategy
                or if an unexpected error occurs during parsing.
        """
        # If already an enum, return directly
        if isinstance(strategy, EnumSelectionStrategy):
            return strategy

        # Parse string to enum
        try:
            return EnumSelectionStrategy(strategy.lower())
        except ValueError as e:
            valid_strategies = [s.value for s in EnumSelectionStrategy]
            raise ProtocolConfigurationError(
                f"Invalid selection strategy '{strategy}'. "
                f"Valid strategies are: {valid_strategies}",
                details={
                    "strategy": strategy,
                    "valid_strategies": valid_strategies,
                    "correlation_id": correlation_id,
                },
            ) from e
        except Exception as e:
            # Catch any other unexpected errors (e.g., AttributeError if strategy
            # is an unexpected type that passed type checking)
            raise ProtocolConfigurationError(
                f"Failed to parse selection strategy: {type(e).__name__}",
                details={
                    "strategy": str(strategy),
                    "error": str(e),
                    "correlation_id": correlation_id,
                },
            ) from e

    def _filter_by_contract_type(
        self,
        results: list[ModelRegistrationProjection],
        contract_type: str | None,
    ) -> list[ModelRegistrationProjection]:
        """Filter results by contract type if specified.

        Args:
            results: List of registration projections to filter.
            contract_type: Optional contract type to filter by. If None, returns
                results unchanged.

        Returns:
            Filtered list of projections matching the contract type, or original
            list if contract_type is None.
        """
        if contract_type is None:
            return results
        return [r for r in results if r.contract_type == contract_type]


__all__: list[str] = ["ServiceCapabilityQuery"]
