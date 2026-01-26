# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Graph Database Handler - Implements ProtocolGraphDatabaseHandler from omnibase_spi.

Provides backend-agnostic graph database operations via the neo4j async driver,
supporting Memgraph and Neo4j through Bolt protocol with Cypher queries.

Protocol Implementation:
    Implements ProtocolGraphDatabaseHandler from omnibase_spi.protocols.storage,
    providing typed graph operations with models from omnibase_core.models.graph.

Supported Operations:
    - execute_query: Execute parameterized Cypher queries
    - execute_query_batch: Transactional batch query execution
    - create_node: Create nodes with labels and properties
    - create_relationship: Create typed relationships between nodes
    - delete_node: Delete nodes with optional cascade (DETACH DELETE)
    - delete_relationship: Delete relationships by ID
    - traverse: Graph traversal with configurable depth and filters
    - health_check: Connection health monitoring
    - describe: Handler metadata introspection

Security:
    - All queries use parameterization to prevent injection attacks
    - Credentials are treated as secrets and never logged
    - Health check responses sanitize error messages

Circuit Breaker Pattern:
    Uses MixinAsyncCircuitBreaker for fault tolerance with automatic
    recovery from transient failures.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from uuid import UUID, uuid4

from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import (
    AuthError,
    ConstraintError,
    Neo4jError,
    ServiceUnavailable,
    TransactionError,
)

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_core.models.graph import (
    ModelGraphBatchResult,
    ModelGraphDatabaseNode,
    ModelGraphDeleteResult,
    ModelGraphHandlerMetadata,
    ModelGraphHealthStatus,
    ModelGraphQueryCounters,
    ModelGraphQueryResult,
    ModelGraphQuerySummary,
    ModelGraphRelationship,
    ModelGraphTraversalFilters,
    ModelGraphTraversalResult,
)
from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType, EnumResponseStatus
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.handlers.models.graph import (
    ModelGraphExecutePayload,
    ModelGraphHandlerPayload,
    ModelGraphQueryPayload,
    ModelGraphRecord,
)
from omnibase_infra.handlers.models.model_graph_handler_response import (
    ModelGraphHandlerResponse,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker, MixinEnvelopeExtraction
from omnibase_infra.utils.util_env_parsing import parse_env_float
from omnibase_spi.protocols.storage import ProtocolGraphDatabaseHandler

logger = logging.getLogger(__name__)

HANDLER_ID_GRAPH: str = "graph-handler"

SUPPORTED_OPERATIONS: frozenset[str] = frozenset(
    {
        "graph.execute_query",
        "graph.execute_query_batch",
        "graph.create_node",
        "graph.create_relationship",
        "graph.delete_node",
        "graph.delete_relationship",
        "graph.traverse",
    }
)

_DEFAULT_TIMEOUT_SECONDS: float = parse_env_float(
    "ONEX_GRAPH_TIMEOUT",
    30.0,
    min_value=0.1,
    max_value=3600.0,
    transport_type=EnumInfraTransportType.GRAPH,
    service_name="graph_handler",
)
_DEFAULT_POOL_SIZE: int = 50
_HEALTH_CACHE_SECONDS: float = 10.0

# Cypher label validation: alphanumeric and underscore only
# This prevents injection attacks via malicious label values
_CYPHER_LABEL_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class HandlerGraph(
    MixinAsyncCircuitBreaker, MixinEnvelopeExtraction, ProtocolGraphDatabaseHandler
):
    """Graph database handler implementing ProtocolGraphDatabaseHandler.

    Provides typed graph database operations using neo4j async driver,
    supporting Memgraph and Neo4j via Bolt protocol with Cypher queries.

    Protocol Compliance:
        Implements all methods from ProtocolGraphDatabaseHandler:
        - handler_type property returning "graph_database"
        - supports_transactions property returning True
        - initialize(), shutdown() lifecycle methods
        - execute_query(), execute_query_batch() query methods
        - create_node(), create_relationship() creation methods
        - delete_node(), delete_relationship() deletion methods
        - traverse() graph traversal method
        - health_check(), describe() introspection methods

    Security Policy:
        Credentials are treated as secrets and never logged or exposed in errors.

    Circuit Breaker Pattern:
        Uses MixinAsyncCircuitBreaker for fault tolerance with automatic
        recovery after transient failures.

    Example:
        ```python
        handler = HandlerGraph()
        await handler.initialize(
            connection_uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )

        result = await handler.execute_query(
            query="MATCH (n:Person {name: $name}) RETURN n",
            parameters={"name": "Alice"},
        )

        await handler.shutdown()
        ```
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize HandlerGraph with ONEX container for dependency injection.

        Args:
            container: ONEX container providing dependency injection for
                services, configuration, and runtime context.

        Note:
            The container is stored for interface compliance with the standard ONEX
            handler pattern (def __init__(self, container: ModelONEXContainer)) and
            to enable future DI-based service resolution. Currently, the handler
            operates independently but the container parameter ensures API
            consistency across all handlers.
        """
        self._container = container
        self._driver: AsyncDriver | None = None
        self._connection_uri: str = ""
        self._database: str = "memgraph"
        self._timeout: float = _DEFAULT_TIMEOUT_SECONDS
        self._pool_size: int = _DEFAULT_POOL_SIZE
        self._initialized: bool = False
        self._cached_health: ModelGraphHealthStatus | None = None
        self._health_cache_time: float = 0.0

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            String "graph_database" as defined by ProtocolGraphDatabaseHandler.
        """
        return "graph_database"

    @property
    def supports_transactions(self) -> bool:
        """Return whether this handler supports transactional operations.

        Returns:
            True - Neo4j/Memgraph support ACID transactions.
        """
        return True

    async def initialize(  # type: ignore[override]
        self,
        connection_uri: str,
        auth: tuple[str, str] | None = None,
        *,
        options: Mapping[str, JsonType] | None = None,
    ) -> None:
        """Initialize the graph database connection.

        Establishes connection to the graph database using the provided URI
        and authentication credentials. Configures connection pools and
        validates connectivity.

        Args:
            connection_uri: Database connection URI (e.g., "bolt://localhost:7687").
            auth: Optional tuple of (username, password) for authentication.
            options: Additional connection parameters:
                - max_connection_pool_size: Maximum connections in pool (default: 50)
                - database: Database name (default: "memgraph")
                - timeout_seconds: Operation timeout (default: 30.0)
                - encrypted: Whether to use TLS/SSL encryption

        Raises:
            RuntimeHostError: If configuration is invalid.
            InfraConnectionError: If connection to graph database fails.
            InfraAuthenticationError: If authentication fails.
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

        self._connection_uri = connection_uri
        opts = dict(options) if options else {}

        # Extract configuration options
        pool_raw = opts.get("max_connection_pool_size", _DEFAULT_POOL_SIZE)
        self._pool_size = (
            int(pool_raw)
            if isinstance(pool_raw, int | float | str)
            else _DEFAULT_POOL_SIZE
        )

        timeout_raw = opts.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        self._timeout = (
            float(timeout_raw)
            if isinstance(timeout_raw, int | float | str)
            else _DEFAULT_TIMEOUT_SECONDS
        )

        database_raw = opts.get("database", "memgraph")
        self._database = str(database_raw) if database_raw else "memgraph"

        encrypted = opts.get("encrypted", False)

        # Create async driver
        try:
            self._driver = AsyncGraphDatabase.driver(
                connection_uri,
                auth=auth,
                max_connection_pool_size=self._pool_size,
                encrypted=bool(encrypted) if encrypted else False,
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            self._initialized = True
            logger.info(
                "%s initialized successfully",
                self.__class__.__name__,
                extra={
                    "handler": self.__class__.__name__,
                    "correlation_id": str(init_correlation_id),
                },
            )
        except AuthError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="initialize",
                target_name="graph_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraAuthenticationError(
                "Graph database authentication failed", context=ctx
            ) from e
        except ServiceUnavailable as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="initialize",
                target_name="graph_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraConnectionError(
                "Failed to connect to graph database", context=ctx
            ) from e
        except Exception as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="initialize",
                target_name="graph_handler",
                correlation_id=init_correlation_id,
            )
            raise InfraConnectionError(
                f"Connection failed: {type(e).__name__}", context=ctx
            ) from e

        # Initialize circuit breaker
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="graph",
            transport_type=EnumInfraTransportType.GRAPH,
        )

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Close database connections and release resources.

        Gracefully shuts down the handler by closing all active connections
        and releasing resources.

        Args:
            timeout_seconds: Maximum time to wait for shutdown. Defaults to 30.0.
        """
        correlation_id = uuid4()
        logger.info(
            "Shutting down %s",
            self.__class__.__name__,
            extra={
                "handler": self.__class__.__name__,
                "correlation_id": str(correlation_id),
                "timeout_seconds": timeout_seconds,
            },
        )

        if self._driver is not None:
            try:
                await self._driver.close()
            except Exception as e:
                logger.warning(
                    "Error during driver close: %s",
                    type(e).__name__,
                    extra={"correlation_id": str(correlation_id)},
                )
            self._driver = None

        self._initialized = False
        self._cached_health = None
        logger.info(
            "%s shutdown complete",
            self.__class__.__name__,
            extra={"correlation_id": str(correlation_id)},
        )

    def _ensure_initialized(
        self, operation: str, correlation_id: object
    ) -> AsyncDriver:
        """Ensure handler is initialized and return driver.

        Args:
            operation: Name of the operation being performed.
            correlation_id: Correlation ID for error context.

        Returns:
            The initialized AsyncDriver.

        Raises:
            RuntimeHostError: If handler is not initialized.
        """
        if not self._initialized or self._driver is None:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation=operation,
                target_name="graph_handler",
                correlation_id=correlation_id
                if isinstance(correlation_id, UUID)
                else None,
            )
            raise RuntimeHostError(
                "HandlerGraph not initialized. Call initialize() first.", context=ctx
            )
        return self._driver

    async def execute_query(
        self,
        query: str,
        parameters: Mapping[str, JsonType] | None = None,
    ) -> ModelGraphQueryResult:
        """Execute a Cypher query and return typed results.

        Security:
            Uses parameterized queries to prevent injection attacks.
            NEVER construct queries via string concatenation with user input.

        Args:
            query: The Cypher query string.
            parameters: Optional mapping of query parameters.

        Returns:
            ModelGraphQueryResult with records, summary, counters, and execution time.

        Raises:
            RuntimeHostError: If handler not initialized or query invalid.
            InfraConnectionError: If query execution fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("execute_query", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("execute_query", correlation_id)

        params = dict(parameters) if parameters else {}
        start_time = time.perf_counter()

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, params)
                records_data = await result.data()
                summary = await result.consume()

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return ModelGraphQueryResult(
                records=list(records_data),
                summary=ModelGraphQuerySummary(
                    query_type=summary.query_type or "unknown",
                    database=self._database,
                    contains_updates=summary.counters.contains_updates,
                ),
                counters=ModelGraphQueryCounters(
                    nodes_created=summary.counters.nodes_created,
                    nodes_deleted=summary.counters.nodes_deleted,
                    relationships_created=summary.counters.relationships_created,
                    relationships_deleted=summary.counters.relationships_deleted,
                    properties_set=summary.counters.properties_set,
                    labels_added=summary.counters.labels_added,
                    labels_removed=summary.counters.labels_removed,
                ),
                execution_time_ms=execution_time_ms,
            )
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="execute_query",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute_query")
            raise InfraConnectionError(
                f"Query execution failed: {type(e).__name__}", context=ctx
            ) from e

    async def execute_query_batch(  # type: ignore[override]
        self,
        queries: list[tuple[str, Mapping[str, JsonType] | None]],
        transaction: bool = True,
    ) -> ModelGraphBatchResult:
        """Execute multiple queries, optionally within a transaction.

        Args:
            queries: List of (query, parameters) tuples to execute.
            transaction: If True, execute all queries atomically. Defaults to True.

        Returns:
            ModelGraphBatchResult with individual results and success status.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If batch execution fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("execute_query_batch", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("execute_query_batch", correlation_id)

        results: list[ModelGraphQueryResult] = []
        rollback_occurred = False
        start_time = time.perf_counter()

        try:
            if transaction and self.supports_transactions:
                # Execute all queries in a single transaction
                async with driver.session(database=self._database) as session:
                    tx = await session.begin_transaction()
                    try:
                        for query, params in queries:
                            query_start = time.perf_counter()
                            tx_result = await tx.run(
                                query, dict(params) if params else {}
                            )
                            records_data = await tx_result.data()
                            summary = await tx_result.consume()
                            query_time_ms = (time.perf_counter() - query_start) * 1000

                            results.append(
                                ModelGraphQueryResult(
                                    records=list(records_data),
                                    summary=ModelGraphQuerySummary(
                                        query_type=summary.query_type or "unknown",
                                        database=self._database,
                                        contains_updates=summary.counters.contains_updates,
                                    ),
                                    counters=ModelGraphQueryCounters(
                                        nodes_created=summary.counters.nodes_created,
                                        nodes_deleted=summary.counters.nodes_deleted,
                                        relationships_created=summary.counters.relationships_created,
                                        relationships_deleted=summary.counters.relationships_deleted,
                                        properties_set=summary.counters.properties_set,
                                        labels_added=summary.counters.labels_added,
                                        labels_removed=summary.counters.labels_removed,
                                    ),
                                    execution_time_ms=query_time_ms,
                                )
                            )
                        await tx.commit()
                    except Exception:
                        await tx.rollback()
                        rollback_occurred = True
                        raise
            else:
                # Execute queries individually without transaction
                for query, params in queries:
                    # Type assertion: execute_query returns ModelGraphQueryResult in this handler
                    query_result: ModelGraphQueryResult = await self.execute_query(
                        query, params
                    )  # type: ignore[assignment]
                    results.append(query_result)

            return ModelGraphBatchResult(
                results=results,
                success=True,
                transaction_id=correlation_id if transaction else None,
                rollback_occurred=rollback_occurred,
            )
        except TransactionError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="execute_query_batch",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute_query_batch")
            raise InfraConnectionError(
                f"Batch transaction failed: {type(e).__name__}", context=ctx
            ) from e
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="execute_query_batch",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("execute_query_batch")
            raise InfraConnectionError(
                f"Batch execution failed: {type(e).__name__}", context=ctx
            ) from e

    def _validate_cypher_labels(
        self, labels: list[str], operation: str, correlation_id: UUID
    ) -> None:
        """Validate that all labels are safe for Cypher queries.

        Labels are embedded directly in Cypher queries (not parameterized),
        so they must be validated to prevent injection attacks.

        Args:
            labels: List of labels to validate.
            operation: Operation name for error context.
            correlation_id: Correlation ID for error context.

        Raises:
            RuntimeHostError: If any label contains unsafe characters.
        """
        for label in labels:
            if not _CYPHER_LABEL_PATTERN.match(label):
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.GRAPH,
                    operation=operation,
                    target_name="graph_handler",
                    correlation_id=correlation_id,
                )
                raise RuntimeHostError(
                    f"Invalid label '{label}': labels must start with a letter or "
                    f"underscore and contain only alphanumeric characters and underscores",
                    context=ctx,
                )

    async def create_node(
        self,
        labels: list[str],
        properties: Mapping[str, JsonType],
    ) -> ModelGraphDatabaseNode:
        """Create a new node in the graph.

        Args:
            labels: List of labels to assign to the node.
            properties: Mapping of property key-value pairs.

        Returns:
            ModelGraphDatabaseNode with the created node's details.

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If node creation fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("create_node", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("create_node", correlation_id)

        # Validate labels to prevent Cypher injection
        self._validate_cypher_labels(labels, "create_node", correlation_id)

        # Build Cypher query with labels
        labels_str = ":".join(labels) if labels else ""
        label_clause = f":{labels_str}" if labels_str else ""
        query = f"CREATE (n{label_clause} $props) RETURN n, elementId(n) as eid, id(n) as nid"

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, {"props": dict(properties)})
                record = await result.single()
                await result.consume()

                if record is None:
                    ctx = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.GRAPH,
                        operation="create_node",
                        target_name="graph_handler",
                        correlation_id=correlation_id,
                    )
                    raise RuntimeHostError(
                        "Node creation returned no result", context=ctx
                    )

                node = record["n"]
                element_id = str(record["eid"])
                node_id = str(record["nid"])

                return ModelGraphDatabaseNode(
                    id=node_id,
                    element_id=element_id,
                    labels=list(node.labels),
                    properties=dict(node.items()),
                )
        except ConstraintError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="create_node",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Node creation failed: constraint violation", context=ctx
            ) from e
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="create_node",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("create_node")
            raise InfraConnectionError(
                f"Node creation failed: {type(e).__name__}", context=ctx
            ) from e

    async def create_relationship(
        self,
        from_node_id: str | int,
        to_node_id: str | int,
        relationship_type: str,
        properties: Mapping[str, JsonType] | None = None,
    ) -> ModelGraphRelationship:
        """Create a relationship between two nodes.

        Args:
            from_node_id: Identifier of the source node.
            to_node_id: Identifier of the target node.
            relationship_type: Type of the relationship (e.g., "KNOWS").
            properties: Optional relationship properties.

        Returns:
            ModelGraphRelationship with the created relationship's details.

        Raises:
            RuntimeHostError: If handler not initialized or nodes don't exist.
            InfraConnectionError: If relationship creation fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("create_relationship", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("create_relationship", correlation_id)

        # Validate relationship type to prevent Cypher injection
        if not _CYPHER_LABEL_PATTERN.match(relationship_type):
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="create_relationship",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Invalid relationship_type '{relationship_type}': must start with a "
                f"letter or underscore and contain only alphanumeric characters and underscores",
                context=ctx,
            )

        # Determine if IDs are element IDs (strings with colons) or internal IDs
        from_is_element_id = isinstance(from_node_id, str) and ":" in from_node_id
        to_is_element_id = isinstance(to_node_id, str) and ":" in to_node_id

        # Build appropriate match clauses
        if from_is_element_id:
            from_match = "MATCH (a) WHERE elementId(a) = $from_id"
        else:
            from_match = "MATCH (a) WHERE id(a) = $from_id"

        if to_is_element_id:
            to_match = "MATCH (b) WHERE elementId(b) = $to_id"
        else:
            to_match = "MATCH (b) WHERE id(b) = $to_id"

        props = dict(properties) if properties else {}
        query = f"""
        {from_match}
        {to_match}
        CREATE (a)-[r:{relationship_type} $props]->(b)
        RETURN r, elementId(r) as eid, id(r) as rid,
               elementId(a) as start_eid, elementId(b) as end_eid
        """

        params: dict[str, object] = {
            "from_id": int(from_node_id) if not from_is_element_id else from_node_id,
            "to_id": int(to_node_id) if not to_is_element_id else to_node_id,
            "props": props,
        }

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, params)
                record = await result.single()
                await result.consume()

                if record is None:
                    ctx = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.GRAPH,
                        operation="create_relationship",
                        target_name="graph_handler",
                        correlation_id=correlation_id,
                    )
                    raise RuntimeHostError(
                        "Relationship creation failed: nodes not found", context=ctx
                    )

                rel = record["r"]
                element_id = str(record["eid"])
                rel_id = str(record["rid"])
                start_eid = str(record["start_eid"])
                end_eid = str(record["end_eid"])

                return ModelGraphRelationship(
                    id=rel_id,
                    element_id=element_id,
                    type=rel.type,
                    properties=dict(rel.items()),
                    start_node_id=start_eid,
                    end_node_id=end_eid,
                )
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="create_relationship",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("create_relationship")
            raise InfraConnectionError(
                f"Relationship creation failed: {type(e).__name__}", context=ctx
            ) from e

    async def delete_node(
        self,
        node_id: str | int,
        detach: bool = False,
    ) -> ModelGraphDeleteResult:
        """Delete a node from the graph.

        Args:
            node_id: Identifier of the node to delete.
            detach: If True, delete all relationships first (DETACH DELETE).

        Returns:
            ModelGraphDeleteResult with deletion status and counts.

        Raises:
            RuntimeHostError: If handler not initialized or node has relationships.
            InfraConnectionError: If deletion fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("delete_node", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("delete_node", correlation_id)

        is_element_id = isinstance(node_id, str) and ":" in str(node_id)
        start_time = time.perf_counter()

        if is_element_id:
            match_clause = "MATCH (n) WHERE elementId(n) = $node_id"
        else:
            match_clause = "MATCH (n) WHERE id(n) = $node_id"

        # Count relationships before delete if detaching
        rel_count = 0
        if detach:
            count_query = (
                f"{match_clause} OPTIONAL MATCH (n)-[r]-() RETURN count(r) as cnt"
            )
            try:
                async with driver.session(database=self._database) as session:
                    result = await session.run(
                        count_query,
                        {"node_id": node_id if is_element_id else int(node_id)},
                    )
                    record = await result.single()
                    await result.consume()
                    if record:
                        rel_count = record["cnt"]
            except Neo4jError:
                pass  # Best effort count

        delete_keyword = "DETACH DELETE" if detach else "DELETE"
        query = f"{match_clause} {delete_keyword} n RETURN count(n) as deleted"

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(
                    query,
                    {"node_id": node_id if is_element_id else int(node_id)},
                )
                record = await result.single()
                await result.consume()

                execution_time_ms = (time.perf_counter() - start_time) * 1000
                deleted = record["deleted"] if record else 0

                return ModelGraphDeleteResult(
                    success=deleted > 0,
                    node_id=str(node_id),
                    relationships_deleted=rel_count if detach else 0,
                    execution_time_ms=execution_time_ms,
                )
        except ConstraintError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="delete_node",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "Cannot delete node with relationships. Use detach=True.", context=ctx
            ) from e
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="delete_node",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("delete_node")
            raise InfraConnectionError(
                f"Node deletion failed: {type(e).__name__}", context=ctx
            ) from e

    async def delete_relationship(
        self,
        relationship_id: str | int,
    ) -> ModelGraphDeleteResult:
        """Delete a relationship from the graph.

        Args:
            relationship_id: Identifier of the relationship to delete.

        Returns:
            ModelGraphDeleteResult with deletion status.

        Raises:
            RuntimeHostError: If handler not initialized.
            InfraConnectionError: If deletion fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("delete_relationship", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("delete_relationship", correlation_id)

        is_element_id = isinstance(relationship_id, str) and ":" in str(relationship_id)
        start_time = time.perf_counter()

        if is_element_id:
            query = """
            MATCH ()-[r]-()
            WHERE elementId(r) = $rel_id
            DELETE r
            RETURN count(r) as deleted
            """
        else:
            query = """
            MATCH ()-[r]-()
            WHERE id(r) = $rel_id
            DELETE r
            RETURN count(r) as deleted
            """

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(
                    query,
                    {
                        "rel_id": relationship_id
                        if is_element_id
                        else int(relationship_id)
                    },
                )
                record = await result.single()
                await result.consume()

                execution_time_ms = (time.perf_counter() - start_time) * 1000
                deleted = record["deleted"] if record else 0

                return ModelGraphDeleteResult(
                    success=deleted > 0,
                    node_id=None,  # This is relationship deletion, not node
                    relationships_deleted=deleted,
                    execution_time_ms=execution_time_ms,
                )
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="delete_relationship",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("delete_relationship")
            raise InfraConnectionError(
                f"Relationship deletion failed: {type(e).__name__}", context=ctx
            ) from e

    async def traverse(
        self,
        start_node_id: str | int,
        relationship_types: list[str] | None = None,
        direction: str = "outgoing",
        max_depth: int = 1,
        filters: ModelGraphTraversalFilters | None = None,
    ) -> ModelGraphTraversalResult:
        """Traverse the graph from a starting node.

        Args:
            start_node_id: Identifier of the node to start from.
            relationship_types: Optional list of relationship types to follow.
            direction: Direction to traverse ("outgoing", "incoming", "both").
            max_depth: Maximum traversal depth. Defaults to 1.
            filters: Optional traversal filters for labels and properties.

        Returns:
            ModelGraphTraversalResult with discovered nodes, relationships, and paths.

        Raises:
            RuntimeHostError: If handler not initialized or invalid parameters.
            InfraConnectionError: If traversal fails.
        """
        correlation_id = uuid4()
        driver = self._ensure_initialized("traverse", correlation_id)

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("traverse", correlation_id)

        is_element_id = isinstance(start_node_id, str) and ":" in str(start_node_id)
        start_time = time.perf_counter()

        # Validate relationship types to prevent Cypher injection
        if relationship_types:
            for rel_type in relationship_types:
                if not _CYPHER_LABEL_PATTERN.match(rel_type):
                    ctx = ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.GRAPH,
                        operation="traverse",
                        target_name="graph_handler",
                        correlation_id=correlation_id,
                    )
                    raise RuntimeHostError(
                        f"Invalid relationship_type '{rel_type}': must start with a "
                        f"letter or underscore and contain only alphanumeric characters "
                        f"and underscores",
                        context=ctx,
                    )

        # Validate filter labels to prevent Cypher injection
        if filters and filters.node_labels:
            self._validate_cypher_labels(
                filters.node_labels, "traverse", correlation_id
            )

        # Build match clause for start node
        if is_element_id:
            start_match = "MATCH (start) WHERE elementId(start) = $start_id"
        else:
            start_match = "MATCH (start) WHERE id(start) = $start_id"

        # Build relationship pattern
        rel_types_pattern = ""
        if relationship_types:
            rel_types_pattern = ":" + "|".join(relationship_types)

        # Direction patterns
        if direction == "incoming":
            rel_pattern = f"<-[r{rel_types_pattern}*1..{max_depth}]-"
        elif direction == "both":
            rel_pattern = f"-[r{rel_types_pattern}*1..{max_depth}]-"
        else:  # outgoing (default)
            rel_pattern = f"-[r{rel_types_pattern}*1..{max_depth}]->"

        # Build filter conditions
        filter_conditions: list[str] = []
        if filters:
            if filters.node_labels:
                label_checks = " OR ".join(
                    f"'{lbl}' IN labels(n)" for lbl in filters.node_labels
                )
                filter_conditions.append(f"({label_checks})")
            if filters.node_properties:
                for key, value in filters.node_properties.items():
                    filter_conditions.append(f"n.{key} = ${key}")

        where_clause = ""
        if filter_conditions:
            where_clause = "WHERE " + " AND ".join(filter_conditions)

        query = f"""
        {start_match}
        MATCH p = (start){rel_pattern}(n)
        {where_clause}
        WITH DISTINCT n, relationships(p) as rels, [node in nodes(p) | elementId(node)] as path_ids
        RETURN n, elementId(n) as eid, id(n) as nid, rels, path_ids
        LIMIT 1000
        """

        params: dict[str, object] = {
            "start_id": start_node_id if is_element_id else int(start_node_id),
        }
        if filters and filters.node_properties:
            params.update(filters.node_properties)

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(query, params)
                records = await result.data()
                await result.consume()

            nodes: list[ModelGraphDatabaseNode] = []
            relationships: list[ModelGraphRelationship] = []
            paths: list[list[str]] = []
            seen_node_ids: set[str] = set()
            seen_rel_ids: set[str] = set()
            max_depth_reached = 0

            for record in records:
                node = record["n"]
                element_id = str(record["eid"])
                node_id = str(record["nid"])

                if element_id not in seen_node_ids:
                    seen_node_ids.add(element_id)
                    nodes.append(
                        ModelGraphDatabaseNode(
                            id=node_id,
                            element_id=element_id,
                            labels=list(node.labels),
                            properties=dict(node.items()),
                        )
                    )

                # Process relationships
                for rel in record.get("rels", []):
                    rel_eid = rel.element_id
                    if rel_eid not in seen_rel_ids:
                        seen_rel_ids.add(rel_eid)
                        relationships.append(
                            ModelGraphRelationship(
                                id=str(rel.id),
                                element_id=rel_eid,
                                type=rel.type,
                                properties=dict(rel.items()),
                                start_node_id=rel.start_node.element_id,
                                end_node_id=rel.end_node.element_id,
                            )
                        )

                # Process path
                path_ids = record.get("path_ids", [])
                if path_ids:
                    paths.append(path_ids)
                    max_depth_reached = max(max_depth_reached, len(path_ids) - 1)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return ModelGraphTraversalResult(
                nodes=nodes,
                relationships=relationships,
                paths=paths,
                depth_reached=max_depth_reached,
                execution_time_ms=execution_time_ms,
            )
        except Neo4jError as e:
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.GRAPH,
                operation="traverse",
                target_name="graph_handler",
                correlation_id=correlation_id,
            )
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("traverse")
            raise InfraConnectionError(
                f"Traversal failed: {type(e).__name__}", context=ctx
            ) from e

    async def health_check(self) -> ModelGraphHealthStatus:
        """Check handler health and database connectivity.

        Returns cached results for rapid repeated calls to prevent
        overwhelming the backend.

        Returns:
            ModelGraphHealthStatus with health status and latency.

        Raises:
            RuntimeHostError: If called before initialize().
        """
        correlation_id = uuid4()

        # Return cached result if recent
        current_time = time.time()
        if (
            self._cached_health is not None
            and current_time - self._health_cache_time < _HEALTH_CACHE_SECONDS
        ):
            return self._cached_health

        if not self._initialized or self._driver is None:
            return ModelGraphHealthStatus(
                healthy=False,
                latency_ms=0.0,
                database_version=None,
                connection_count=0,
            )

        start_time = time.perf_counter()

        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run("RETURN 1 as n")
                await result.consume()

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Try to get server info
            version = None
            try:
                server_info = await self._driver.get_server_info()
                version = server_info.agent if server_info else None
            except Exception:
                pass

            health = ModelGraphHealthStatus(
                healthy=True,
                latency_ms=latency_ms,
                database_version=version,
                connection_count=0,  # Neo4j driver doesn't expose pool stats easily
            )

            # Cache the result
            self._cached_health = health
            self._health_cache_time = current_time

            return health
        except Exception as e:
            logger.warning(
                "Health check failed: %s",
                type(e).__name__,
                extra={"correlation_id": str(correlation_id)},
            )
            return ModelGraphHealthStatus(
                healthy=False,
                latency_ms=0.0,
                database_version=None,
                connection_count=0,
            )

    async def describe(self) -> ModelGraphHandlerMetadata:  # type: ignore[override]
        """Return handler metadata and capabilities.

        Returns:
            ModelGraphHandlerMetadata with handler information.

        Note:
            This method is async per protocol specification (v0.5.0+).
        """
        # Determine database type based on connection URI
        database_type = "memgraph"
        if self._connection_uri:
            uri_lower = self._connection_uri.lower()
            if "neo4j" in uri_lower:
                database_type = "neo4j"
            elif "neptune" in uri_lower:
                database_type = "neptune"

        capabilities = [
            "cypher",
            "parameterized_queries",
            "transactions",
            "node_crud",
            "relationship_crud",
            "traversal",
            "batch_operations",
        ]

        return ModelGraphHandlerMetadata(
            handler_type=self.handler_type,
            capabilities=capabilities,
            database_type=database_type,
            supports_transactions=self.supports_transactions,
        )

    async def execute(
        self, envelope: dict[str, object]
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph operation from envelope.

        Dispatches to specialized methods based on operation field.
        This method enables contract-based handler discovery via HandlerPluginLoader.

        Supported operations:
            - graph.execute_query: Execute a Cypher query
            - graph.execute_query_batch: Execute multiple queries in transaction
            - graph.create_node: Create a node with labels and properties
            - graph.create_relationship: Create relationship between nodes
            - graph.delete_node: Delete a node (optionally with DETACH)
            - graph.delete_relationship: Delete a relationship
            - graph.traverse: Traverse graph from starting node

        Args:
            envelope: Request envelope containing:
                - operation: Graph operation (graph.execute_query, etc.)
                - payload: dict with operation-specific parameters
                - correlation_id: Optional correlation ID for tracing
                - envelope_id: Optional envelope ID for causality tracking

        Returns:
            ModelHandlerOutput wrapping the operation result with correlation tracking.

        Raises:
            RuntimeHostError: If handler not initialized or invalid input.
            InfraConnectionError: If graph database connection fails.
            InfraAuthenticationError: If authentication fails.

        Envelope-Based Routing:
            This handler uses envelope-based operation routing. See CLAUDE.md section
            "Intent Model Architecture > Envelope-Based Handler Routing" for the full
            design pattern and how orchestrators translate intents to handler envelopes.
        """
        correlation_id = self._extract_correlation_id(envelope)
        input_envelope_id = self._extract_envelope_id(envelope)

        if not self._initialized or self._driver is None:
            raise RuntimeHostError(
                "HandlerGraph not initialized. Call initialize() first.",
                context=self._error_context("execute", correlation_id),
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope",
                context=self._error_context("execute", correlation_id),
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeHostError(
                "Missing or invalid 'payload' in envelope",
                context=self._error_context(operation, correlation_id),
            )

        # Dispatch table maps operation strings to handler methods
        dispatch_table = {
            "graph.execute_query": self._execute_query_operation,
            "graph.execute_query_batch": self._execute_query_batch_operation,
            "graph.create_node": self._create_node_operation,
            "graph.create_relationship": self._create_relationship_operation,
            "graph.delete_node": self._delete_node_operation,
            "graph.delete_relationship": self._delete_relationship_operation,
            "graph.traverse": self._traverse_operation,
        }

        handler = dispatch_table.get(operation)
        if handler is None:
            raise RuntimeHostError(
                f"Operation '{operation}' not supported. "
                f"Available: {', '.join(sorted(dispatch_table.keys()))}",
                context=self._error_context(operation, correlation_id),
            )

        return await handler(payload, correlation_id, input_envelope_id)

    async def _execute_query_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.execute_query operation.

        Validates that payload contains required 'query' field and optional
        'parameters' dict, then delegates to execute_query() and converts
        the result to ModelHandlerOutput format.

        Args:
            payload: Request payload with 'query' and optional 'parameters'.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping query results.

        Raises:
            RuntimeHostError: If query field missing or parameters invalid.
        """
        query = payload.get("query")
        if not isinstance(query, str):
            raise RuntimeHostError(
                "Missing or invalid 'query' in payload",
                context=self._error_context("graph.execute_query", correlation_id),
            )

        parameters = payload.get("parameters")
        params_dict: Mapping[str, JsonType] | None = None
        if parameters is not None:
            if not isinstance(parameters, dict):
                raise RuntimeHostError(
                    "Invalid 'parameters' in payload - must be a dict",
                    context=self._error_context("graph.execute_query", correlation_id),
                )
            # Type ignore: dict variance - dict[str, object] to Mapping[str, JsonType]
            params_dict = parameters  # type: ignore[assignment]

        try:
            result = await self.execute_query(query, params_dict)
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Query execution failed: {e}",
                context=self._error_context("graph.execute_query", correlation_id),
            ) from e

        # Convert records to ModelGraphRecord format
        # Note: Type ignore needed due to dict variance - dict[str, JsonType] vs dict[str, object]
        records = [
            ModelGraphRecord(data=record)  # type: ignore[arg-type]
            for record in result.records
        ]

        query_payload = ModelGraphQueryPayload(
            cypher=query,
            records=records,
            summary={
                "query_type": result.summary.query_type,
                "database": result.summary.database,
                "contains_updates": result.summary.contains_updates,
                "execution_time_ms": result.execution_time_ms,
                "nodes_created": result.counters.nodes_created,
                "nodes_deleted": result.counters.nodes_deleted,
                "relationships_created": result.counters.relationships_created,
                "relationships_deleted": result.counters.relationships_deleted,
            },
        )

        return self._build_graph_response(
            query_payload, correlation_id, input_envelope_id
        )

    async def _execute_query_batch_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.execute_query_batch operation.

        Validates batch query structure with fail-fast semantics:
        - 'queries' must be a list of dicts
        - Each query dict must have 'query' (str) and optional 'parameters' (dict)
        - 'transaction' must be boolean if provided

        Args:
            payload: Request payload with 'queries' list and optional 'transaction'.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping batch execution results.

        Raises:
            RuntimeHostError: If queries list invalid or any query malformed.
        """
        queries_raw = payload.get("queries")
        if not isinstance(queries_raw, list):
            raise RuntimeHostError(
                "Missing or invalid 'queries' in payload - must be a list",
                context=self._error_context(
                    "graph.execute_query_batch", correlation_id
                ),
            )

        # Convert to list of tuples with fail-fast validation
        queries: list[tuple[str, Mapping[str, JsonType] | None]] = []
        for idx, q in enumerate(queries_raw):
            if not isinstance(q, dict):
                raise RuntimeHostError(
                    f"Query at index {idx} must be a dict, got {type(q).__name__}",
                    context=self._error_context(
                        "graph.execute_query_batch", correlation_id
                    ),
                )
            query_str = q.get("query")
            if not isinstance(query_str, str):
                raise RuntimeHostError(
                    f"Query at index {idx} missing or invalid 'query' field",
                    context=self._error_context(
                        "graph.execute_query_batch", correlation_id
                    ),
                )
            params = q.get("parameters")
            if params is not None and not isinstance(params, dict):
                raise RuntimeHostError(
                    f"Query at index {idx} has invalid 'parameters' - must be dict or null",
                    context=self._error_context(
                        "graph.execute_query_batch", correlation_id
                    ),
                )
            # Type ignore: dict variance - dict[str, object] to Mapping[str, JsonType]
            queries.append(
                (query_str, params if isinstance(params, dict) else None)  # type: ignore[arg-type]
            )

        # Validate transaction is boolean - don't silently coerce other types
        transaction_raw = payload.get("transaction", True)
        if not isinstance(transaction_raw, bool):
            raise RuntimeHostError(
                f"Invalid 'transaction' in payload - must be boolean, "
                f"got {type(transaction_raw).__name__}",
                context=self._error_context(
                    "graph.execute_query_batch", correlation_id
                ),
            )
        transaction = transaction_raw

        try:
            result = await self.execute_query_batch(queries, transaction=transaction)
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Batch query execution failed: {e}",
                context=self._error_context(
                    "graph.execute_query_batch", correlation_id
                ),
            ) from e

        execute_payload = ModelGraphExecutePayload(
            cypher="BATCH",
            counters={
                "success": result.success,
                "rollback_occurred": result.rollback_occurred,
                "query_count": len(result.results),
                "transaction_id": str(result.transaction_id)
                if result.transaction_id
                else None,
            },
            success=result.success,
        )

        return self._build_graph_response(
            execute_payload, correlation_id, input_envelope_id
        )

    async def _create_node_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.create_node operation.

        Validates optional 'labels' (list of strings) and 'properties' (dict),
        then delegates to create_node() method.

        Args:
            payload: Request payload with optional 'labels' and 'properties'.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping created node details.

        Raises:
            RuntimeHostError: If labels or properties have invalid types.
        """
        labels_raw = payload.get("labels")
        if labels_raw is not None:
            if not isinstance(labels_raw, list):
                raise RuntimeHostError(
                    "Invalid 'labels' in payload - must be a list of strings",
                    context=self._error_context("graph.create_node", correlation_id),
                )
            labels_list: list[str] = [str(lbl) for lbl in labels_raw]
        else:
            labels_list = []

        properties_raw = payload.get("properties")
        if properties_raw is not None:
            if not isinstance(properties_raw, dict):
                raise RuntimeHostError(
                    "Invalid 'properties' in payload - must be a dict",
                    context=self._error_context("graph.create_node", correlation_id),
                )
            # Type ignore: dict variance - dict[str, object] to Mapping[str, JsonType]
            props_dict: Mapping[str, JsonType] = properties_raw  # type: ignore[assignment]
        else:
            props_dict = {}

        try:
            result = await self.create_node(labels_list, props_dict)
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Node creation failed: {e}",
                context=self._error_context("graph.create_node", correlation_id),
            ) from e

        labels_str = ":" + ":".join(labels_list) if labels_list else "n"
        execute_payload = ModelGraphExecutePayload(
            cypher=f"CREATE ({labels_str} ...)",
            counters={
                "nodes_created": 1,
                "node_id": result.id,
                "element_id": result.element_id,
                "labels": result.labels,
                "properties": result.properties,
            },
            success=True,
        )

        return self._build_graph_response(
            execute_payload, correlation_id, input_envelope_id
        )

    async def _create_relationship_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.create_relationship operation.

        Validates required fields 'from_node_id', 'to_node_id', 'relationship_type'
        and optional 'properties' dict, then delegates to create_relationship().

        Args:
            payload: Request payload with node IDs, relationship type, and properties.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping created relationship details.

        Raises:
            RuntimeHostError: If required fields missing or properties invalid.
        """
        from_node_id = payload.get("from_node_id")
        to_node_id = payload.get("to_node_id")
        relationship_type = payload.get("relationship_type")

        if from_node_id is None or to_node_id is None or relationship_type is None:
            raise RuntimeHostError(
                "Missing required fields: from_node_id, to_node_id, relationship_type",
                context=self._error_context(
                    "graph.create_relationship", correlation_id
                ),
            )

        properties = payload.get("properties")
        props_dict: Mapping[str, JsonType] | None = None
        if properties is not None:
            if not isinstance(properties, dict):
                raise RuntimeHostError(
                    "Invalid 'properties' in payload - must be a dict",
                    context=self._error_context(
                        "graph.create_relationship", correlation_id
                    ),
                )
            # Type ignore: dict variance - dict[str, object] to Mapping[str, JsonType]
            props_dict = properties  # type: ignore[assignment]

        try:
            result = await self.create_relationship(
                from_node_id=str(from_node_id),
                to_node_id=str(to_node_id),
                relationship_type=str(relationship_type),
                properties=props_dict,
            )
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Relationship creation failed: {e}",
                context=self._error_context(
                    "graph.create_relationship", correlation_id
                ),
            ) from e

        execute_payload = ModelGraphExecutePayload(
            cypher=f"CREATE ()-[:{relationship_type}]->()",
            counters={
                "relationships_created": 1,
                "relationship_id": result.id,
                "element_id": result.element_id,
                "type": result.type,
                "start_node_id": result.start_node_id,
                "end_node_id": result.end_node_id,
            },
            success=True,
        )

        return self._build_graph_response(
            execute_payload, correlation_id, input_envelope_id
        )

    async def _delete_node_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.delete_node operation.

        Validates required 'node_id' and optional 'detach' boolean. The detach
        flag must be explicitly boolean to prevent accidental cascade deletes.

        Args:
            payload: Request payload with 'node_id' and optional 'detach'.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping deletion result.

        Raises:
            RuntimeHostError: If node_id missing or detach is non-boolean.
        """
        node_id = payload.get("node_id")
        if node_id is None:
            raise RuntimeHostError(
                "Missing required field: node_id",
                context=self._error_context("graph.delete_node", correlation_id),
            )

        # Validate detach is boolean - don't silently coerce to prevent accidental deletes
        detach_raw = payload.get("detach", False)
        if not isinstance(detach_raw, bool):
            raise RuntimeHostError(
                f"Invalid 'detach' in payload - must be boolean, "
                f"got {type(detach_raw).__name__}",
                context=self._error_context("graph.delete_node", correlation_id),
            )
        detach = detach_raw

        try:
            result = await self.delete_node(str(node_id), detach=detach)
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Node deletion failed: {e}",
                context=self._error_context("graph.delete_node", correlation_id),
            ) from e

        execute_payload = ModelGraphExecutePayload(
            cypher=f"{'DETACH ' if detach else ''}DELETE (n)",
            counters={
                "nodes_deleted": 1 if result.success else 0,
                "relationships_deleted": result.relationships_deleted,
                "execution_time_ms": result.execution_time_ms,
            },
            success=result.success,
        )

        return self._build_graph_response(
            execute_payload, correlation_id, input_envelope_id
        )

    async def _delete_relationship_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.delete_relationship operation.

        Validates required 'relationship_id' field, then delegates to
        delete_relationship() method.

        Args:
            payload: Request payload with 'relationship_id'.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping deletion result.

        Raises:
            RuntimeHostError: If relationship_id field is missing.
        """
        relationship_id = payload.get("relationship_id")
        if relationship_id is None:
            raise RuntimeHostError(
                "Missing required field: relationship_id",
                context=self._error_context(
                    "graph.delete_relationship", correlation_id
                ),
            )

        try:
            result = await self.delete_relationship(str(relationship_id))
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Relationship deletion failed: {e}",
                context=self._error_context(
                    "graph.delete_relationship", correlation_id
                ),
            ) from e

        execute_payload = ModelGraphExecutePayload(
            cypher="DELETE [r]",
            counters={
                "relationships_deleted": result.relationships_deleted,
                "execution_time_ms": result.execution_time_ms,
            },
            success=result.success,
        )

        return self._build_graph_response(
            execute_payload, correlation_id, input_envelope_id
        )

    async def _traverse_operation(
        self,
        payload: dict[str, object],
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Execute graph.traverse operation.

        Validates traversal parameters with strict type checking:
        - 'start_node_id': required
        - 'relationship_types': optional list of strings
        - 'direction': optional, must be 'outgoing', 'incoming', or 'both'
        - 'max_depth': optional, must be positive integer
        - 'filters': optional dict with 'node_labels' (list) and 'node_properties' (dict)

        Args:
            payload: Request payload with traversal configuration.
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping traversal results (nodes, relationships, paths).

        Raises:
            RuntimeHostError: If start_node_id missing or parameters have invalid types.
        """
        start_node_id = payload.get("start_node_id")
        if start_node_id is None:
            raise RuntimeHostError(
                "Missing required field: start_node_id",
                context=self._error_context("graph.traverse", correlation_id),
            )

        # relationship_types - optional, but if provided must be list
        relationship_types = payload.get("relationship_types")
        rel_types: list[str] | None = None
        if relationship_types is not None:
            if not isinstance(relationship_types, list):
                raise RuntimeHostError(
                    f"Invalid 'relationship_types' - must be list, "
                    f"got {type(relationship_types).__name__}",
                    context=self._error_context("graph.traverse", correlation_id),
                )
            rel_types = [str(rt) for rt in relationship_types]

        # direction - optional with default, but if provided must be valid string
        direction_raw = payload.get("direction")
        if direction_raw is None:
            direction = "outgoing"
        elif not isinstance(direction_raw, str):
            raise RuntimeHostError(
                f"Invalid 'direction' - must be string, "
                f"got {type(direction_raw).__name__}",
                context=self._error_context("graph.traverse", correlation_id),
            )
        elif direction_raw not in ("outgoing", "incoming", "both"):
            raise RuntimeHostError(
                f"Invalid 'direction' value '{direction_raw}' - "
                f"must be 'outgoing', 'incoming', or 'both'",
                context=self._error_context("graph.traverse", correlation_id),
            )
        else:
            direction = direction_raw

        # max_depth - optional with default, but if provided must be positive integer
        max_depth_raw = payload.get("max_depth")
        if max_depth_raw is None:
            max_depth = 1
        elif not isinstance(max_depth_raw, int | float):
            raise RuntimeHostError(
                f"Invalid 'max_depth' - must be int or float, "
                f"got {type(max_depth_raw).__name__}",
                context=self._error_context("graph.traverse", correlation_id),
            )
        else:
            max_depth = int(max_depth_raw)
            if max_depth <= 0:
                raise RuntimeHostError(
                    f"Invalid 'max_depth' value {max_depth} - must be a positive integer",
                    context=self._error_context("graph.traverse", correlation_id),
                )

        # filters - optional, but if provided must be dict with validated fields
        filters = None
        filters_raw = payload.get("filters")
        if filters_raw is not None:
            if not isinstance(filters_raw, dict):
                raise RuntimeHostError(
                    f"Invalid 'filters' - must be dict, "
                    f"got {type(filters_raw).__name__}",
                    context=self._error_context("graph.traverse", correlation_id),
                )

            # Validate node_labels - must be list or None
            node_labels = filters_raw.get("node_labels")
            if node_labels is not None and not isinstance(node_labels, list):
                raise RuntimeHostError(
                    f"Invalid 'filters.node_labels' - must be list, "
                    f"got {type(node_labels).__name__}",
                    context=self._error_context("graph.traverse", correlation_id),
                )

            # Validate node_properties - must be dict or None
            node_properties = filters_raw.get("node_properties")
            if node_properties is not None and not isinstance(node_properties, dict):
                raise RuntimeHostError(
                    f"Invalid 'filters.node_properties' - must be dict, "
                    f"got {type(node_properties).__name__}",
                    context=self._error_context("graph.traverse", correlation_id),
                )

            # Type ignore: list[object] to list[str] - validated above as list
            # Type ignore: dict[str, object] to dict[str, JsonType] - validated above as dict
            filters = ModelGraphTraversalFilters(
                node_labels=node_labels,  # type: ignore[arg-type]
                node_properties=node_properties,  # type: ignore[arg-type]
            )

        try:
            result = await self.traverse(
                start_node_id=str(start_node_id),
                relationship_types=rel_types,
                direction=direction,
                max_depth=max_depth,
                filters=filters,
            )
        except (InfraConnectionError, InfraAuthenticationError, RuntimeHostError):
            # Already has proper context, re-raise as-is
            raise
        except Exception as e:
            raise RuntimeHostError(
                f"Traversal failed: {e}",
                context=self._error_context("graph.traverse", correlation_id),
            ) from e

        # Convert nodes to records
        records = []
        for node in result.nodes:
            records.append(
                ModelGraphRecord(
                    data={
                        "id": node.id,
                        "element_id": node.element_id,
                        "labels": node.labels,
                        "properties": node.properties,
                    }
                )
            )

        query_payload = ModelGraphQueryPayload(
            cypher=f"TRAVERSE from {start_node_id}",
            records=records,
            summary={
                "depth_reached": result.depth_reached,
                "nodes_found": len(result.nodes),
                "relationships_found": len(result.relationships),
                "paths_found": len(result.paths),
                "execution_time_ms": result.execution_time_ms,
            },
        )

        return self._build_graph_response(
            query_payload, correlation_id, input_envelope_id
        )

    def _error_context(
        self, operation: str, correlation_id: UUID
    ) -> ModelInfraErrorContext:
        """Create standardized error context for graph operations.

        Args:
            operation: The operation name (e.g., "graph.execute_query").
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelInfraErrorContext configured for graph handler.
        """
        return ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.GRAPH,
            operation=operation,
            target_name="graph_handler",
            correlation_id=correlation_id,
        )

    def _build_graph_response(
        self,
        typed_payload: ModelGraphQueryPayload | ModelGraphExecutePayload,
        correlation_id: UUID,
        input_envelope_id: UUID,
    ) -> ModelHandlerOutput[ModelGraphHandlerResponse]:
        """Build standardized ModelGraphHandlerResponse wrapped in ModelHandlerOutput.

        This helper method ensures consistent response formatting across all
        graph operations, matching the pattern used by HandlerDb and HandlerConsul.

        Args:
            typed_payload: Strongly-typed payload (query or execute).
            correlation_id: Correlation ID for tracing.
            input_envelope_id: Input envelope ID for causality tracking.

        Returns:
            ModelHandlerOutput wrapping ModelGraphHandlerResponse.
        """
        response = ModelGraphHandlerResponse(
            status=EnumResponseStatus.SUCCESS,
            payload=ModelGraphHandlerPayload(data=typed_payload),
            correlation_id=correlation_id,
        )
        return ModelHandlerOutput.for_compute(
            input_envelope_id=input_envelope_id,
            correlation_id=correlation_id,
            handler_id=HANDLER_ID_GRAPH,
            result=response,
        )


__all__: list[str] = ["HandlerGraph", "HANDLER_ID_GRAPH", "SUPPORTED_OPERATIONS"]
