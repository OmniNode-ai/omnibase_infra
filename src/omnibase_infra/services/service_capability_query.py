# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Capability Query Service.

Provides high-level service for querying the registry by capability rather
than by name. This enables capability-based auto-configuration where nodes
declare what they need, not who provides it.

Core Principle: "I'm interested in what you do, not what you are."

Coroutine Safety:
    This service is stateless (except for node selector round-robin state)
    and delegates database operations to ProjectionReaderRegistration, which
    handles coroutine safety and circuit breaker protection.

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

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.discovery.model_dependency_spec import ModelDependencySpec
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.services.enum_selection_strategy import EnumSelectionStrategy
from omnibase_infra.services.service_node_selector import ServiceNodeSelector

if TYPE_CHECKING:
    from omnibase_infra.projectors.projection_reader_registration import (
        ProjectionReaderRegistration,
    )

logger = logging.getLogger(__name__)


class ServiceCapabilityQuery:
    """Queries registry for nodes by capability, not by name.

    Core Principle: "I'm interested in what you do, not what you are."

    This service wraps ProjectionReaderRegistration to provide high-level
    capability-based discovery. Instead of hardcoding module paths, consumers
    can declare what capabilities they need and the system discovers which
    nodes provide them.

    Query Methods:
        - find_nodes_by_capability: Find by capability tag
        - find_nodes_by_intent_type: Find by intent type handled
        - find_nodes_by_protocol: Find by protocol implemented

    Dependency Resolution:
        The resolve_dependency method takes a ModelDependencySpec and:
        1. Determines the discovery strategy (capability/intent/protocol)
        2. Queries the registry for matching nodes
        3. Applies the selection strategy to choose one node

    Design Notes:
        - All queries delegate to ProjectionReaderRegistration
        - Circuit breaker protection is inherited from the reader
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
    """

    def __init__(
        self,
        projection_reader: ProjectionReaderRegistration,
        node_selector: ServiceNodeSelector | None = None,
    ) -> None:
        """Initialize the capability query service.

        Args:
            projection_reader: The projection reader for database queries.
                Must be initialized with an asyncpg connection pool.
            node_selector: Optional node selector for selection strategies.
                If None, creates a new ServiceNodeSelector instance.

        Example:
            >>> pool = await asyncpg.create_pool(dsn)
            >>> reader = ProjectionReaderRegistration(pool)
            >>> query = ServiceCapabilityQuery(reader)
        """
        self._projection_reader = projection_reader
        self._node_selector = node_selector or ServiceNodeSelector()

    async def find_nodes_by_capability(
        self,
        capability: str,
        contract_type: str | None = None,
        state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ) -> list[ModelRegistrationProjection]:
        """Find nodes that provide a specific capability.

        Queries the registry for nodes with the specified capability tag.
        Results can be filtered by contract type and registration state.

        Args:
            capability: Capability tag to search for (e.g., "postgres.storage",
                "kafka.consumer", "consul.registration").
            contract_type: Optional filter by contract type ("effect", "compute",
                "reducer", "orchestrator").
            state: Filter by registration state (default: ACTIVE).

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
        logger.debug(
            "Finding nodes by capability",
            extra={
                "capability": capability,
                "contract_type": contract_type,
                "state": str(state),
            },
        )

        # Query by capability tag
        results = await self._projection_reader.get_by_capability_tag(
            tag=capability,
            state=state,
        )

        # Filter by contract_type if specified
        if contract_type is not None:
            results = [r for r in results if r.contract_type == contract_type]

        logger.debug(
            "Capability query completed",
            extra={
                "capability": capability,
                "result_count": len(results),
            },
        )

        return results

    async def find_nodes_by_intent_type(
        self,
        intent_type: str,
        contract_type: str = "effect",
        state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ) -> list[ModelRegistrationProjection]:
        """Find effect nodes that handle a specific intent type.

        Queries the registry for nodes that can handle the specified intent type.
        Typically used to find effect nodes that execute specific intents.

        Args:
            intent_type: Intent type to search for (e.g., "postgres.upsert",
                "consul.register", "kafka.publish").
            contract_type: Filter by contract type (default: "effect").
                Intents are typically handled by effect nodes.
            state: Filter by registration state (default: ACTIVE).

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
        logger.debug(
            "Finding nodes by intent type",
            extra={
                "intent_type": intent_type,
                "contract_type": contract_type,
                "state": str(state),
            },
        )

        # Query by intent type
        results = await self._projection_reader.get_by_intent_type(
            intent_type=intent_type,
            state=state,
        )

        # Filter by contract_type if specified
        if contract_type is not None:
            results = [r for r in results if r.contract_type == contract_type]

        logger.debug(
            "Intent type query completed",
            extra={
                "intent_type": intent_type,
                "result_count": len(results),
            },
        )

        return results

    async def find_nodes_by_protocol(
        self,
        protocol: str,
        contract_type: str | None = None,
        state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ) -> list[ModelRegistrationProjection]:
        """Find nodes implementing a specific protocol.

        Queries the registry for nodes that implement the specified protocol.
        Useful for finding nodes that satisfy interface requirements.

        Args:
            protocol: Protocol name to search for (e.g., "ProtocolEventPublisher",
                "ProtocolReducer", "ProtocolDatabaseAdapter").
            contract_type: Optional filter by contract type ("effect", "compute",
                "reducer", "orchestrator").
            state: Filter by registration state (default: ACTIVE).

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
        logger.debug(
            "Finding nodes by protocol",
            extra={
                "protocol": protocol,
                "contract_type": contract_type,
                "state": str(state),
            },
        )

        # Query by protocol
        results = await self._projection_reader.get_by_protocol(
            protocol_name=protocol,
            state=state,
        )

        # Filter by contract_type if specified
        if contract_type is not None:
            results = [r for r in results if r.contract_type == contract_type]

        logger.debug(
            "Protocol query completed",
            extra={
                "protocol": protocol,
                "result_count": len(results),
            },
        )

        return results

    async def resolve_dependency(
        self,
        dependency_spec: ModelDependencySpec,
    ) -> ModelRegistrationProjection | None:
        """Resolve a dependency specification to a concrete node.

        Uses the dependency specification to query the registry and select
        a node that matches the specified capability criteria.

        Resolution Strategy:
            1. If capability specified -> find by capability
            2. If intent_types specified -> find by intent type (uses first intent)
            3. If protocol specified -> find by protocol
            4. Apply selection strategy from spec to choose among matches

        Args:
            dependency_spec: Dependency specification from contract.
                Contains capability filters and selection strategy.

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
            >>> node = await query.resolve_dependency(spec)
            >>> if node:
            ...     print(f"Resolved to: {node.entity_id}")
            ... else:
            ...     print("No node found for capability")
        """
        logger.debug(
            "Resolving dependency",
            extra={
                "dependency_name": dependency_spec.name,
                "dependency_type": dependency_spec.type,
                "capability": dependency_spec.capability,
                "intent_types": dependency_spec.intent_types,
                "protocol": dependency_spec.protocol,
                "selection_strategy": dependency_spec.selection_strategy,
            },
        )

        # Determine state filter
        state = self._parse_state(dependency_spec.state)

        # Find candidates based on discovery strategy
        candidates: list[ModelRegistrationProjection] = []

        if dependency_spec.has_capability_filter():
            candidates = await self.find_nodes_by_capability(
                capability=dependency_spec.capability,  # type: ignore[arg-type]
                contract_type=dependency_spec.contract_type,
                state=state,
            )

        elif dependency_spec.has_intent_filter():
            # Use first intent type for discovery
            intent_types = dependency_spec.intent_types
            if intent_types:
                for intent_type in intent_types:
                    nodes = await self.find_nodes_by_intent_type(
                        intent_type=intent_type,
                        contract_type=dependency_spec.contract_type or "effect",
                        state=state,
                    )
                    # Add nodes not already in candidates
                    existing_ids = {c.entity_id for c in candidates}
                    for node in nodes:
                        if node.entity_id not in existing_ids:
                            candidates.append(node)
                            existing_ids.add(node.entity_id)

        elif dependency_spec.has_protocol_filter():
            candidates = await self.find_nodes_by_protocol(
                protocol=dependency_spec.protocol,  # type: ignore[arg-type]
                contract_type=dependency_spec.contract_type,
                state=state,
            )

        # No filter specified - cannot resolve
        if not candidates and not dependency_spec.has_any_filter():
            logger.warning(
                "Dependency spec has no capability filter",
                extra={"dependency_name": dependency_spec.name},
            )
            return None

        # No matches found
        if not candidates:
            logger.debug(
                "No candidates found for dependency",
                extra={"dependency_name": dependency_spec.name},
            )
            return None

        # Apply selection strategy
        strategy = self._parse_selection_strategy(dependency_spec.selection_strategy)
        selected = self._node_selector.select(
            candidates=candidates,
            strategy=strategy,
            selection_key=dependency_spec.name,
        )

        if selected:
            logger.debug(
                "Dependency resolved",
                extra={
                    "dependency_name": dependency_spec.name,
                    "selected_entity_id": str(selected.entity_id),
                    "total_candidates": len(candidates),
                    "strategy": dependency_spec.selection_strategy,
                },
            )
        else:
            logger.debug(
                "No node selected for dependency",
                extra={"dependency_name": dependency_spec.name},
            )

        return selected

    def _parse_state(self, state_str: str) -> EnumRegistrationState:
        """Parse state string to EnumRegistrationState.

        Args:
            state_str: State string (e.g., "ACTIVE", "active").

        Returns:
            EnumRegistrationState value.
        """
        try:
            return EnumRegistrationState(state_str.lower())
        except ValueError:
            logger.warning(
                "Unknown state value, defaulting to ACTIVE",
                extra={"state": state_str},
            )
            return EnumRegistrationState.ACTIVE

    def _parse_selection_strategy(self, strategy_str: str) -> EnumSelectionStrategy:
        """Parse selection strategy string to enum.

        Args:
            strategy_str: Strategy string (e.g., "first", "round_robin").

        Returns:
            EnumSelectionStrategy value.
        """
        try:
            return EnumSelectionStrategy(strategy_str)
        except ValueError:
            logger.warning(
                "Unknown selection strategy, defaulting to FIRST",
                extra={"strategy": strategy_str},
            )
            return EnumSelectionStrategy.FIRST


__all__: list[str] = ["ServiceCapabilityQuery"]
