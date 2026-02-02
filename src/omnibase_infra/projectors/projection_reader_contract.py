# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Projection Reader Implementation.

Implements projection reads for the contract registry domain to support
Registry API queries. Provides access to contracts and topics stored in
PostgreSQL by the NodeContractRegistryReducer.

Concurrency Safety:
    This implementation is coroutine-safe for concurrent async read operations.
    Uses asyncpg connection pool for connection management, and asyncio.Lock
    (via MixinAsyncCircuitBreaker) for circuit breaker state protection.

    Note: This is not thread-safe. For multi-threaded access, additional
    synchronization would be required.

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1653: Contract registry state materialization
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.projection.model_contract_projection import (
    ModelContractProjection,
)
from omnibase_infra.models.projection.model_topic_projection import (
    ModelTopicProjection,
)
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

logger = logging.getLogger(__name__)


class ProjectionReaderContract(MixinAsyncCircuitBreaker):
    """Contract projection reader implementation using asyncpg.

    Provides read access to contract and topic projections for the Registry API.
    Supports contract lookups, searches, and topic routing queries.

    Circuit Breaker:
        Uses MixinAsyncCircuitBreaker for resilience. Opens after 5 consecutive
        failures and resets after 60 seconds.

    Security:
        All queries use parameterized statements for SQL injection protection.

    Error Handling Pattern:
        All public methods follow a consistent error handling structure:

        1. Create fresh ModelInfraErrorContext per operation (intentionally NOT
           reused to ensure each operation has isolated context with its own
           correlation ID for distributed tracing).

        2. Check circuit breaker before database operation.

        3. Map exceptions consistently:
           - asyncpg.PostgresConnectionError -> InfraConnectionError
           - asyncpg.QueryCanceledError -> InfraTimeoutError
           - Generic Exception -> RuntimeHostError

        4. Record circuit breaker failures for all exception types.

    JSONB Handling:
        The contract_ids field in topics table is stored as JSONB. While asyncpg
        typically returns JSONB as Python lists, some connection configurations
        may return strings. The _row_to_topic_projection method handles both cases.

    Example:
        >>> pool = await asyncpg.create_pool(dsn)
        >>> reader = ProjectionReaderContract(pool)
        >>> contract = await reader.get_contract_by_id("my-node:1.0.0")
        >>> if contract and contract.is_active:
        ...     print(f"Contract hash: {contract.contract_hash}")
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize reader with connection pool.

        Args:
            pool: asyncpg connection pool for database access.
                  Pool should be created by the caller (e.g., from HandlerDb).
        """
        self._pool = pool
        config = ModelCircuitBreakerConfig.from_env(
            service_name="projection_reader.contract",
            transport_type=EnumInfraTransportType.DATABASE,
        )
        self._init_circuit_breaker_from_config(config)

    def _row_to_contract_projection(
        self, row: asyncpg.Record
    ) -> ModelContractProjection:
        """Convert database row to contract projection model.

        Args:
            row: asyncpg Record from query result

        Returns:
            ModelContractProjection instance
        """
        return ModelContractProjection(
            contract_id=row["contract_id"],
            node_name=row["node_name"],
            version_major=row["version_major"],
            version_minor=row["version_minor"],
            version_patch=row["version_patch"],
            contract_hash=row["contract_hash"],
            contract_yaml=row["contract_yaml"],
            registered_at=row["registered_at"],
            deregistered_at=row["deregistered_at"],
            last_seen_at=row["last_seen_at"],
            is_active=row["is_active"],
            last_event_topic=row["last_event_topic"],
            last_event_partition=row["last_event_partition"],
            last_event_offset=row["last_event_offset"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_topic_projection(self, row: asyncpg.Record) -> ModelTopicProjection:
        """Convert database row to topic projection model.

        Args:
            row: asyncpg Record from query result

        Returns:
            ModelTopicProjection instance

        Note:
            Handles JSONB contract_ids which may be returned as string or list
            depending on connection configuration.
        """
        # Parse contract_ids from JSONB
        contract_ids_data = row["contract_ids"]
        if isinstance(contract_ids_data, str):
            try:
                contract_ids_data = json.loads(contract_ids_data)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Failed to parse contract_ids JSON for topic %s/%s: %s. "
                    "Using empty list.",
                    row["topic_suffix"],
                    row["direction"],
                    str(e),
                )
                contract_ids_data = []

        return ModelTopicProjection(
            topic_suffix=row["topic_suffix"],
            direction=row["direction"],
            contract_ids=contract_ids_data or [],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            is_active=row["is_active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ============================================================
    # Contract Query Methods
    # ============================================================

    async def get_contract_by_id(
        self,
        contract_id: str,
        correlation_id: UUID | None = None,
    ) -> ModelContractProjection | None:
        """Get contract by ID.

        Point lookup for a single contract by its natural key.

        Args:
            contract_id: Contract ID (e.g., "my-node:1.0.0")
            correlation_id: Optional correlation ID for tracing

        Returns:
            Contract projection if exists, None otherwise

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> contract = await reader.get_contract_by_id("my-node:1.0.0")
            >>> if contract:
            ...     print(f"Active: {contract.is_active}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_contract_by_id",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_contract_by_id", corr_id)

        query_sql = """
            SELECT * FROM contracts
            WHERE contract_id = $1
        """

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, contract_id)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if row is None:
                return None

            return self._row_to_contract_projection(row)

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_contract_by_id", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for contract lookup",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_contract_by_id", corr_id)
            raise InfraTimeoutError(
                "Contract lookup timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_contract_by_id", corr_id)
            raise RuntimeHostError(
                f"Failed to get contract by ID: {type(e).__name__}",
                context=ctx,
            ) from e

    async def list_active_contracts(
        self,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> list[ModelContractProjection]:
        """List active contracts with pagination.

        Args:
            limit: Maximum results to return (default: 100)
            offset: Number of results to skip (default: 0)
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of active contract projections

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> contracts = await reader.list_active_contracts(limit=50, offset=0)
            >>> for c in contracts:
            ...     print(f"{c.contract_id}: {c.node_name}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="list_active_contracts",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("list_active_contracts", corr_id)

        query_sql = """
            SELECT * FROM contracts
            WHERE is_active = TRUE
            ORDER BY last_seen_at DESC
            LIMIT $1 OFFSET $2
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, limit, offset)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_contract_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("list_active_contracts", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for active contracts query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("list_active_contracts", corr_id)
            raise InfraTimeoutError(
                "Active contracts query timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("list_active_contracts", corr_id)
            raise RuntimeHostError(
                f"Failed to list active contracts: {type(e).__name__}",
                context=ctx,
            ) from e

    async def list_contracts_by_node_name(
        self,
        node_name: str,
        include_inactive: bool = False,
        correlation_id: UUID | None = None,
    ) -> list[ModelContractProjection]:
        """List all contracts for a node name.

        Retrieves all versions of a contract by node name. Useful for
        checking available versions of a node.

        Args:
            node_name: ONEX node name to search for
            include_inactive: Whether to include deregistered contracts
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of contract projections ordered by version descending

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> contracts = await reader.list_contracts_by_node_name("my-node")
            >>> for c in contracts:
            ...     print(f"{c.version_string}: active={c.is_active}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="list_contracts_by_node_name",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("list_contracts_by_node_name", corr_id)

        if include_inactive:
            query_sql = """
                SELECT * FROM contracts
                WHERE node_name = $1
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            """
            params = [node_name]
        else:
            query_sql = """
                SELECT * FROM contracts
                WHERE node_name = $1 AND is_active = TRUE
                ORDER BY version_major DESC, version_minor DESC, version_patch DESC
            """
            params = [node_name]

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_contract_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "list_contracts_by_node_name", corr_id
                )
            raise InfraConnectionError(
                "Failed to connect to database for node name query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "list_contracts_by_node_name", corr_id
                )
            raise InfraTimeoutError(
                "Node name query timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "list_contracts_by_node_name", corr_id
                )
            raise RuntimeHostError(
                f"Failed to list contracts by node name: {type(e).__name__}",
                context=ctx,
            ) from e

    async def search_contracts(
        self,
        query: str,
        limit: int = 100,
        correlation_id: UUID | None = None,
    ) -> list[ModelContractProjection]:
        """Search contracts by node name.

        Performs case-insensitive search on node_name field.

        Args:
            query: Search query string
            limit: Maximum results to return (default: 100)
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of matching contract projections

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> contracts = await reader.search_contracts("registry")
            >>> for c in contracts:
            ...     print(f"{c.node_name}: {c.contract_id}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="search_contracts",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("search_contracts", corr_id)

        query_sql = """
            SELECT * FROM contracts
            WHERE node_name ILIKE '%' || $1 || '%'
            ORDER BY is_active DESC, last_seen_at DESC
            LIMIT $2
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, query, limit)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_contract_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("search_contracts", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for contract search",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("search_contracts", corr_id)
            raise InfraTimeoutError(
                "Contract search timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("search_contracts", corr_id)
            raise RuntimeHostError(
                f"Failed to search contracts: {type(e).__name__}",
                context=ctx,
            ) from e

    async def count_contracts_by_status(
        self,
        correlation_id: UUID | None = None,
    ) -> dict[str, int]:
        """Count contracts by active/inactive status.

        Returns aggregated counts for monitoring and dashboards.

        Args:
            correlation_id: Optional correlation ID for tracing

        Returns:
            Dict with 'active' and 'inactive' counts

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> counts = await reader.count_contracts_by_status()
            >>> print(f"Active: {counts['active']}, Inactive: {counts['inactive']}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="count_contracts_by_status",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("count_contracts_by_status", corr_id)

        query_sql = """
            SELECT is_active, COUNT(*) as count
            FROM contracts
            GROUP BY is_active
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            result: dict[str, int] = {"active": 0, "inactive": 0}
            for row in rows:
                if row["is_active"]:
                    result["active"] = row["count"]
                else:
                    result["inactive"] = row["count"]

            return result

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("count_contracts_by_status", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for contract count",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("count_contracts_by_status", corr_id)
            raise InfraTimeoutError(
                "Contract count query timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("count_contracts_by_status", corr_id)
            raise RuntimeHostError(
                f"Failed to count contracts: {type(e).__name__}",
                context=ctx,
            ) from e

    # ============================================================
    # Topic Query Methods
    # ============================================================

    async def list_topics(
        self,
        direction: str | None = None,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> list[ModelTopicProjection]:
        """List topics with optional direction filter.

        Args:
            direction: Optional filter by direction ('publish' or 'subscribe')
            limit: Maximum results to return (default: 100)
            offset: Number of results to skip (default: 0)
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of topic projections

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> topics = await reader.list_topics(direction="publish")
            >>> for t in topics:
            ...     print(f"{t.topic_suffix}: {t.contract_count} contracts")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="list_topics",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("list_topics", corr_id)

        if direction is not None:
            query_sql = """
                SELECT * FROM topics
                WHERE direction = $1
                ORDER BY last_seen_at DESC
                LIMIT $2 OFFSET $3
            """
            params = [direction, limit, offset]
        else:
            query_sql = """
                SELECT * FROM topics
                ORDER BY last_seen_at DESC
                LIMIT $1 OFFSET $2
            """
            params = [limit, offset]

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, *params)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_topic_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("list_topics", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for topics query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("list_topics", corr_id)
            raise InfraTimeoutError(
                "Topics query timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("list_topics", corr_id)
            raise RuntimeHostError(
                f"Failed to list topics: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_topic(
        self,
        topic_suffix: str,
        direction: str,
        correlation_id: UUID | None = None,
    ) -> ModelTopicProjection | None:
        """Get topic by suffix and direction.

        Point lookup for a single topic by its composite primary key.

        Args:
            topic_suffix: Topic suffix without environment prefix
            direction: Direction ('publish' or 'subscribe')
            correlation_id: Optional correlation ID for tracing

        Returns:
            Topic projection if exists, None otherwise

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> topic = await reader.get_topic(
            ...     "onex.evt.platform.contract-registered.v1",
            ...     "publish"
            ... )
            >>> if topic:
            ...     print(f"Contracts: {topic.contract_ids}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_topic",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_topic", corr_id)

        query_sql = """
            SELECT * FROM topics
            WHERE topic_suffix = $1 AND direction = $2
        """

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, topic_suffix, direction)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if row is None:
                return None

            return self._row_to_topic_projection(row)

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_topic", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for topic lookup",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_topic", corr_id)
            raise InfraTimeoutError(
                "Topic lookup timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_topic", corr_id)
            raise RuntimeHostError(
                f"Failed to get topic: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_topics_by_contract(
        self,
        contract_id: str,
        correlation_id: UUID | None = None,
    ) -> list[ModelTopicProjection]:
        """Get all topics referenced by a contract.

        Uses GIN index on contract_ids JSONB array for efficient lookup.

        Args:
            contract_id: Contract ID to search for
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of topic projections that reference the contract

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> topics = await reader.get_topics_by_contract("my-node:1.0.0")
            >>> for t in topics:
            ...     print(f"{t.direction}: {t.topic_suffix}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_topics_by_contract",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_topics_by_contract", corr_id)

        # Use ? operator to check if JSONB array contains the contract_id
        query_sql = """
            SELECT * FROM topics
            WHERE contract_ids ? $1
            ORDER BY direction, topic_suffix
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, contract_id)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_topic_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_topics_by_contract", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for topics by contract query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_topics_by_contract", corr_id)
            raise InfraTimeoutError(
                "Topics by contract query timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_topics_by_contract", corr_id)
            raise RuntimeHostError(
                f"Failed to get topics by contract: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_contracts_by_topic(
        self,
        topic_suffix: str,
        correlation_id: UUID | None = None,
    ) -> list[ModelContractProjection]:
        """Get all contracts that reference a topic.

        Finds all contracts that publish to or subscribe from a given topic.
        Uses the topics table to get contract IDs, then fetches full contracts.

        Args:
            topic_suffix: Topic suffix to search for
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of contract projections that reference the topic

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> contracts = await reader.get_contracts_by_topic(
            ...     "onex.evt.platform.contract-registered.v1"
            ... )
            >>> for c in contracts:
            ...     print(f"{c.contract_id}: {c.node_name}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_contracts_by_topic",
            target_name="projection_reader.contract",
            correlation_id=corr_id,
        )

        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_contracts_by_topic", corr_id)

        # Join topics with contracts using JSONB array unnest
        # This gets all contracts referenced by any direction of the topic
        query_sql = """
            SELECT DISTINCT c.*
            FROM topics t
            CROSS JOIN LATERAL jsonb_array_elements_text(t.contract_ids) AS cid
            JOIN contracts c ON c.contract_id = cid
            WHERE t.topic_suffix = $1
            ORDER BY c.contract_id
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, topic_suffix)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_contract_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_contracts_by_topic", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for contracts by topic query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_contracts_by_topic", corr_id)
            raise InfraTimeoutError(
                "Contracts by topic query timed out",
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                ),
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_contracts_by_topic", corr_id)
            raise RuntimeHostError(
                f"Failed to get contracts by topic: {type(e).__name__}",
                context=ctx,
            ) from e


__all__: list[str] = ["ProjectionReaderContract"]
