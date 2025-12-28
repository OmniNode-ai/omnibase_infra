# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Projection Reader Implementation.

Implements projection reads for the registration domain to support
orchestrator state queries. Orchestrators read current state using
projections only - never scanning Kafka topics.

Concurrency Safety:
    This implementation is coroutine-safe for concurrent async read operations.
    Uses asyncpg connection pool for connection management, and asyncio.Lock
    (via MixinAsyncCircuitBreaker) for circuit breaker state protection.

    Note: This is not thread-safe. For multi-threaded access, additional
    synchronization would be required.

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-930 (C0): Projection Reader Protocol
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

logger = logging.getLogger(__name__)


class ProjectionReaderRegistration(MixinAsyncCircuitBreaker):
    """Registration projection reader implementation using asyncpg.

    Provides read access to registration projections for orchestrators.
    Supports entity lookups, state queries, and deadline scans for
    timeout handling.

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

        This pattern ensures predictable error behavior and enables consistent
        error handling by callers across all reader methods.

    JSONB Handling:
        The capabilities field is stored as JSONB in PostgreSQL. While asyncpg
        typically returns JSONB as Python dicts, some connection configurations
        may return strings. The _row_to_projection method handles both cases
        using json.loads() for string fallback.

    Example:
        >>> pool = await asyncpg.create_pool(dsn)
        >>> reader = ProjectionReaderRegistration(pool)
        >>> proj = await reader.get_entity_state(node_id, "registration")
        >>> if proj and proj.current_state.is_active():
        ...     print("Node is active")
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize reader with connection pool.

        Args:
            pool: asyncpg connection pool for database access.
                  Pool should be created by the caller (e.g., from DbHandler).
        """
        self._pool = pool
        config = ModelCircuitBreakerConfig.from_env(
            service_name="projection_reader.registration",
            transport_type=EnumInfraTransportType.DATABASE,
        )
        self._init_circuit_breaker_from_config(config)

    def _row_to_projection(self, row: asyncpg.Record) -> ModelRegistrationProjection:
        """Convert database row to projection model.

        Args:
            row: asyncpg Record from query result

        Returns:
            ModelRegistrationProjection instance
        """
        # Parse capabilities from JSONB.
        # asyncpg typically returns JSONB as Python dicts, but connection
        # configuration (e.g., custom type codecs) may return strings.
        # Handle both cases for robustness.
        capabilities_data = row["capabilities"]
        if isinstance(capabilities_data, str):
            capabilities_data = json.loads(capabilities_data)
        capabilities = ModelNodeCapabilities.model_validate(capabilities_data)

        return ModelRegistrationProjection(
            entity_id=row["entity_id"],
            domain=row["domain"],
            current_state=EnumRegistrationState(row["current_state"]),
            node_type=row["node_type"],
            node_version=row["node_version"],
            capabilities=capabilities,
            ack_deadline=row["ack_deadline"],
            liveness_deadline=row["liveness_deadline"],
            last_heartbeat_at=row["last_heartbeat_at"],
            ack_timeout_emitted_at=row["ack_timeout_emitted_at"],
            liveness_timeout_emitted_at=row["liveness_timeout_emitted_at"],
            last_applied_event_id=row["last_applied_event_id"],
            last_applied_offset=row["last_applied_offset"],
            last_applied_sequence=row["last_applied_sequence"],
            last_applied_partition=row["last_applied_partition"],
            registered_at=row["registered_at"],
            updated_at=row["updated_at"],
            correlation_id=row["correlation_id"],
        )

    async def get_entity_state(
        self,
        entity_id: UUID,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> ModelRegistrationProjection | None:
        """Get current projection for entity.

        Point lookup for a single entity's current state.
        Primary method for orchestrators to check entity state.

        Args:
            entity_id: Node UUID
            domain: Domain namespace (default: "registration")
            correlation_id: Optional correlation ID for tracing

        Returns:
            Projection if exists, None otherwise

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If query times out
            RuntimeHostError: For other database errors

        Example:
            >>> proj = await reader.get_entity_state(node_id)
            >>> if proj and proj.current_state.is_active():
            ...     route_work_to_node(node_id)
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_entity_state",
            target_name="projection_reader.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_entity_state", corr_id)

        query_sql = """
            SELECT * FROM registration_projections
            WHERE entity_id = $1 AND domain = $2
        """

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, entity_id, domain)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if row is None:
                return None

            return self._row_to_projection(row)

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_entity_state", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for entity state lookup",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_entity_state", corr_id)
            raise InfraTimeoutError(
                "Entity state lookup timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_entity_state", corr_id)
            raise RuntimeHostError(
                f"Failed to get entity state: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_registration_status(
        self,
        entity_id: UUID,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> EnumRegistrationState | None:
        """Get current registration state (convenience method).

        Lightweight method that returns only the FSM state without
        the full projection. Useful for quick state checks.

        Args:
            entity_id: Node UUID
            domain: Domain namespace (default: "registration")
            correlation_id: Optional correlation ID for tracing

        Returns:
            Current FSM state if exists, None otherwise

        Example:
            >>> state = await reader.get_registration_status(node_id)
            >>> if state == EnumRegistrationState.ACTIVE:
            ...     print("Node is active")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_registration_status",
            target_name="projection_reader.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_registration_status", corr_id)

        query_sql = """
            SELECT current_state FROM registration_projections
            WHERE entity_id = $1 AND domain = $2
        """

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, entity_id, domain)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if row is None:
                return None

            return EnumRegistrationState(row["current_state"])

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_registration_status", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for status lookup",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_registration_status", corr_id)
            raise InfraTimeoutError(
                "Registration status lookup timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_registration_status", corr_id)
            raise RuntimeHostError(
                f"Failed to get registration status: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_by_state(
        self,
        state: EnumRegistrationState,
        domain: str = "registration",
        limit: int = 100,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Query projections by state.

        Find all projections with a specific FSM state.

        Args:
            state: FSM state to filter by
            domain: Domain namespace (default: "registration")
            limit: Maximum results to return (default: 100)
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of matching projections

        Example:
            >>> active = await reader.get_by_state(EnumRegistrationState.ACTIVE)
            >>> for proj in active:
            ...     print(f"Active node: {proj.entity_id}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_by_state",
            target_name="projection_reader.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_by_state", corr_id)

        query_sql = """
            SELECT * FROM registration_projections
            WHERE domain = $1 AND current_state = $2
            ORDER BY updated_at DESC
            LIMIT $3
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, domain, state.value, limit)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_by_state", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for state query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_by_state", corr_id)
            raise InfraTimeoutError(
                "State query timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("get_by_state", corr_id)
            raise RuntimeHostError(
                f"Failed to query by state: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_overdue_ack_registrations(
        self,
        now: datetime,
        domain: str = "registration",
        limit: int = 100,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Get registrations with overdue ack deadlines (not yet emitted).

        Per C2: Returns entities where:
        - ack_deadline < now
        - ack_timeout_emitted_at IS NULL
        - current_state requires ack (ACCEPTED or AWAITING_ACK)

        Used by orchestrators during RuntimeTick processing to find
        registrations that need ack timeout events emitted.

        Args:
            now: Current time (injected by runtime)
            domain: Domain namespace (default: "registration")
            limit: Maximum results to return (default: 100)
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of registrations needing ack timeout events

        Example:
            >>> overdue = await reader.get_overdue_ack_registrations(datetime.now(UTC))
            >>> for proj in overdue:
            ...     emit_ack_timeout_event(proj.entity_id)
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_overdue_ack_registrations",
            target_name="projection_reader.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("get_overdue_ack_registrations", corr_id)

        # States that require ack
        ack_states = [
            EnumRegistrationState.ACCEPTED.value,
            EnumRegistrationState.AWAITING_ACK.value,
        ]

        query_sql = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND ack_deadline < $2
              AND ack_timeout_emitted_at IS NULL
              AND current_state = ANY($3)
            ORDER BY ack_deadline ASC
            LIMIT $4
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, domain, now, ack_states, limit)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "get_overdue_ack_registrations", corr_id
                )
            raise InfraConnectionError(
                "Failed to connect to database for overdue ack query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "get_overdue_ack_registrations", corr_id
                )
            raise InfraTimeoutError(
                "Overdue ack registrations query timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "get_overdue_ack_registrations", corr_id
                )
            raise RuntimeHostError(
                f"Failed to query overdue ack registrations: {type(e).__name__}",
                context=ctx,
            ) from e

    async def get_overdue_liveness_registrations(
        self,
        now: datetime,
        domain: str = "registration",
        limit: int = 100,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Get registrations with overdue liveness deadlines (not yet emitted).

        Per C2: Returns entities where:
        - liveness_deadline < now
        - liveness_timeout_emitted_at IS NULL
        - current_state = ACTIVE

        Used by orchestrators during RuntimeTick processing to find
        active registrations that have missed their liveness deadline.

        Args:
            now: Current time (injected by runtime)
            domain: Domain namespace (default: "registration")
            limit: Maximum results to return (default: 100)
            correlation_id: Optional correlation ID for tracing

        Returns:
            List of registrations needing liveness timeout events

        Example:
            >>> overdue = await reader.get_overdue_liveness_registrations(datetime.now(UTC))
            >>> for proj in overdue:
            ...     emit_liveness_expired_event(proj.entity_id)
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="get_overdue_liveness_registrations",
            target_name="projection_reader.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                "get_overdue_liveness_registrations", corr_id
            )

        query_sql = """
            SELECT * FROM registration_projections
            WHERE domain = $1
              AND liveness_deadline < $2
              AND liveness_timeout_emitted_at IS NULL
              AND current_state = $3
            ORDER BY liveness_deadline ASC
            LIMIT $4
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    query_sql,
                    domain,
                    now,
                    EnumRegistrationState.ACTIVE.value,
                    limit,
                )

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            return [self._row_to_projection(row) for row in rows]

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "get_overdue_liveness_registrations", corr_id
                )
            raise InfraConnectionError(
                "Failed to connect to database for overdue liveness query",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "get_overdue_liveness_registrations", corr_id
                )
            raise InfraTimeoutError(
                "Overdue liveness registrations query timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "get_overdue_liveness_registrations", corr_id
                )
            raise RuntimeHostError(
                f"Failed to query overdue liveness registrations: {type(e).__name__}",
                context=ctx,
            ) from e

    async def count_by_state(
        self,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> dict[EnumRegistrationState, int]:
        """Count projections by state.

        Aggregates projection counts for each FSM state.
        Useful for monitoring and dashboard metrics.

        Args:
            domain: Domain namespace (default: "registration")
            correlation_id: Optional correlation ID for tracing

        Returns:
            Dict mapping state to count

        Example:
            >>> counts = await reader.count_by_state()
            >>> print(f"Active nodes: {counts.get(EnumRegistrationState.ACTIVE, 0)}")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="count_by_state",
            target_name="projection_reader.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("count_by_state", corr_id)

        query_sql = """
            SELECT current_state, COUNT(*) as count
            FROM registration_projections
            WHERE domain = $1
            GROUP BY current_state
        """

        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query_sql, domain)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            result: dict[EnumRegistrationState, int] = {}
            for row in rows:
                state = EnumRegistrationState(row["current_state"])
                result[state] = row["count"]

            return result

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("count_by_state", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for state count",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("count_by_state", corr_id)
            raise InfraTimeoutError(
                "State count query timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("count_by_state", corr_id)
            raise RuntimeHostError(
                f"Failed to count by state: {type(e).__name__}",
                context=ctx,
            ) from e


__all__: list[str] = ["ProjectionReaderRegistration"]
