# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent Handler - Temporary demo wiring for intent graph operations.

Wraps HandlerGraph to provide intent-specific graph operations for the demo.
This is temporary hardcoded routing that will be replaced by contract-driven
handler routing in production.

Supported Operations:
    - intent.store: Store an intent as a graph node with label "Intent"
    - intent.query_session: Query intents by session_id property
    - intent.query_distribution: Get intent count/statistics

Note:
    This is TEMPORARY demo wiring. Keep it simple and focused on the demo use case.
    Production implementation should use contract-driven handler routing.
"""

# TODO(OMN-1515): Remove demo wiring after intent routing is contract-driven

from __future__ import annotations

import logging
from uuid import UUID, uuid4

from omnibase_core.container import ModelONEXContainer
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.handler_graph import HandlerGraph
from omnibase_infra.mixins import MixinEnvelopeExtraction

logger = logging.getLogger(__name__)

HANDLER_ID_INTENT: str = "intent-handler"
_SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "intent.store",
        "intent.query_session",
        "intent.query_distribution",
    }
)


class HandlerIntent(MixinEnvelopeExtraction):  # DEMO ONLY
    """Intent handler wrapping HandlerGraph for intent-specific operations.

    This handler provides a simplified interface for storing and querying
    intents in the graph database. It wraps HandlerGraph and translates
    intent-specific operations to graph operations.

    Note:
        This is temporary demo wiring. The handler assumes HandlerGraph
        is already initialized and passed via config during initialize().

    Idempotency:
        - intent.store: NOT idempotent (creates new node each call)
        - intent.query_session: Idempotent (read-only query)
        - intent.query_distribution: Idempotent (read-only aggregation)
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerIntent with ONEX container for dependency injection.

        Args:
            container: ONEX container for dependency injection.
        """
        self._container = container
        self._graph_handler: HandlerGraph | None = None
        self._initialized: bool = False

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the intent handler.

        Args:
            config: Configuration dict containing:
                - graph_handler: Pre-initialized HandlerGraph instance (required)

        Raises:
            RuntimeHostError: If graph_handler is missing or invalid.
        """
        init_correlation_id = uuid4()

        logger.info(
            "Initializing %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

        graph_handler = config.get("graph_handler")
        if not isinstance(graph_handler, HandlerGraph):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="initialize",
                target_name="intent_handler",
                correlation_id=init_correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'graph_handler' in config - "
                "must be an initialized HandlerGraph instance",
                context=ctx,
            )

        self._graph_handler = graph_handler
        self._initialized = True

        logger.info(
            "%s initialized successfully",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(init_correlation_id),
            },
        )

    async def shutdown(self) -> None:
        """Shutdown the intent handler.

        Note:
            This handler does not own the graph handler, so we do not
            shut it down here. The caller is responsible for managing
            the graph handler lifecycle.
        """
        self._graph_handler = None
        self._initialized = False
        logger.info("HandlerIntent shutdown complete")

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute intent operation from envelope.

        Args:
            envelope: Request envelope containing:
                - operation: Intent operation (intent.store, intent.query_session, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            dict containing operation result with:
                - success: bool indicating operation success
                - data: Operation-specific result data
                - correlation_id: UUID string for tracing

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
        """
        correlation_id = self._extract_correlation_id(envelope)

        if not self._initialized or self._graph_handler is None:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="execute",
                target_name="intent_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HandlerIntent not initialized. Call initialize() first.",
                context=ctx,
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="execute",
                target_name="intent_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=ctx,
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation=operation,
                target_name="intent_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Operation '{operation}' not supported. "
                f"Available: {', '.join(sorted(_SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation=operation,
                target_name="intent_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=ctx,
            )

        # Route to appropriate handler method
        if operation == "intent.store":
            return await self._store_intent(payload, correlation_id)
        elif operation == "intent.query_session":
            return await self._query_session(payload, correlation_id)
        else:  # intent.query_distribution
            return await self._query_distribution(correlation_id)

    async def _store_intent(
        self, payload: dict[str, object], correlation_id: UUID
    ) -> dict[str, object]:
        """Store an intent as a graph node.

        Args:
            payload: Intent data to store. Should contain:
                - intent_type: Type of intent (required)
                - session_id: Session identifier (optional)
                - Additional properties as needed

        Returns:
            dict with created node details.
        """
        # Note: _graph_handler is guaranteed non-None by execute() validation
        assert self._graph_handler is not None  # Type narrowing for mypy

        # Extract intent properties - use JsonType for graph compatibility
        properties: dict[str, JsonType] = {
            "correlation_id": str(correlation_id),
        }

        # Copy all payload properties to node properties
        for key, value in payload.items():
            # Convert non-primitive types to strings for graph storage
            # NOTE: Using tuple form for isinstance to avoid union validator flag
            if isinstance(value, (str, int, float, bool)) or value is None:
                properties[key] = value
            else:
                properties[key] = str(value)

        # Create the intent node
        node = await self._graph_handler.create_node(
            labels=["Intent"],
            properties=properties,
        )

        return {
            "success": True,
            "data": {
                "node_id": node.id,
                "element_id": node.element_id,
                "labels": node.labels,
                "properties": node.properties,
            },
            "correlation_id": str(correlation_id),
        }

    async def _query_session(
        self, payload: dict[str, object], correlation_id: UUID
    ) -> dict[str, object]:
        """Query intents by session_id.

        Args:
            payload: Query parameters. Should contain:
                - session_id: Session identifier to filter by (required)

        Returns:
            dict with matching intent nodes.
        """
        # Note: _graph_handler is guaranteed non-None by execute() validation
        assert self._graph_handler is not None  # Type narrowing for mypy

        session_id = payload.get("session_id")
        if not session_id:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="intent.query_session",
                target_name="intent_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Missing 'session_id' in payload",
                context=ctx,
            )

        # Query intents by session_id
        # SECURITY: Using parameterized query ($session_id) to prevent Cypher injection
        query = """
        MATCH (i:Intent {session_id: $session_id})
        RETURN i, elementId(i) as eid, id(i) as nid
        ORDER BY i.created_at DESC
        """

        result = await self._graph_handler.execute_query(
            query=query,
            parameters={"session_id": str(session_id)},
        )

        # Transform records to intent data
        intents = []
        for record in result.records:
            node = record.get("i")
            if node:
                intents.append(
                    {
                        "node_id": str(record.get("nid", "")),
                        "element_id": str(record.get("eid", "")),
                        "properties": dict(node) if isinstance(node, dict) else {},
                    }
                )

        return {
            "success": True,
            "data": {
                "session_id": str(session_id),
                "intents": intents,
                "count": len(intents),
            },
            "correlation_id": str(correlation_id),
        }

    async def _query_distribution(self, correlation_id: UUID) -> dict[str, object]:
        """Query intent distribution/statistics.

        Returns:
            dict with intent statistics including counts by intent_type.
        """
        # Note: _graph_handler is guaranteed non-None by execute() validation
        assert self._graph_handler is not None  # Type narrowing for mypy

        # Query total count
        count_query = "MATCH (i:Intent) RETURN count(i) as total"
        count_result = await self._graph_handler.execute_query(query=count_query)

        total_count = 0
        if count_result.records:
            raw_total = count_result.records[0].get("total", 0)
            total_count = int(raw_total) if isinstance(raw_total, int | float) else 0

        # Query distribution by intent_type
        distribution_query = """
        MATCH (i:Intent)
        RETURN i.intent_type as intent_type, count(i) as count
        ORDER BY count DESC
        """
        distribution_result = await self._graph_handler.execute_query(
            query=distribution_query
        )

        # Build distribution dict
        distribution: dict[str, int] = {}
        for record in distribution_result.records:
            intent_type = record.get("intent_type")
            raw_count = record.get("count", 0)
            if intent_type:
                count_val = int(raw_count) if isinstance(raw_count, int | float) else 0
                distribution[str(intent_type)] = count_val

        return {
            "success": True,
            "data": {
                "total_count": total_count,
                "distribution": distribution,
            },
            "correlation_id": str(correlation_id),
        }

    def describe(self) -> dict[str, object]:
        """Return handler metadata and capabilities.

        Returns:
            dict containing handler information.
        """
        return {
            "handler_id": HANDLER_ID_INTENT,
            "handler_type": "intent_handler",
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "initialized": self._initialized,
            "version": "0.1.0-demo",
        }


__all__: list[str] = ["HandlerIntent", "HANDLER_ID_INTENT"]
