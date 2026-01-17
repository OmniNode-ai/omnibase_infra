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
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.utils.util_env_parsing import parse_env_float
from omnibase_spi.protocols.storage import ProtocolGraphDatabaseHandler

logger = logging.getLogger(__name__)

HANDLER_ID_GRAPH: str = "graph-handler"
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


class HandlerGraph(MixinAsyncCircuitBreaker, ProtocolGraphDatabaseHandler):
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
                            result = await tx.run(query, dict(params) if params else {})
                            records_data = await result.data()
                            summary = await result.consume()
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
                    result = await self.execute_query(query, params)
                    results.append(result)

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
                pass  # Version info is optional

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


__all__: list[str] = ["HandlerGraph", "HANDLER_ID_GRAPH"]
