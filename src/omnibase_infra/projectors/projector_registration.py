# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Projector Implementation.

Implements projection persistence for the registration domain with:
- Offset-based idempotency (reject stale updates)
- Parameterized queries for SQL injection protection
- Circuit breaker resilience pattern

Thread Safety:
    This implementation is thread-safe for concurrent persist calls.
    Uses asyncpg connection pool for connection management.

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)

logger = logging.getLogger(__name__)

# Path to SQL schema file (relative to this module)
_SCHEMA_FILE = (
    Path(__file__).parent.parent / "schemas" / "schema_registration_projection.sql"
)


class ProjectorRegistration(MixinAsyncCircuitBreaker):
    """Registration projector implementation using asyncpg.

    Persists registration projections to PostgreSQL with idempotency and
    ordering guarantees. Uses atomic upsert with sequence comparison to
    reject stale updates.

    Circuit Breaker:
        Uses MixinAsyncCircuitBreaker for resilience. Opens after 5 consecutive
        failures and resets after 60 seconds.

    Security:
        All queries use parameterized statements for SQL injection protection.
        DSN and credentials are never logged or exposed in errors.

    Example:
        >>> pool = await asyncpg.create_pool(dsn)
        >>> projector = ProjectorRegistration(pool)
        >>> await projector.initialize_schema()
        >>> result = await projector.persist(
        ...     projection=proj,
        ...     entity_id=proj.entity_id,
        ...     domain="registration",
        ...     sequence_info=seq,
        ... )
        >>> if result:
        ...     print("Projection applied")
        ... else:
        ...     print("Stale update rejected")
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialize projector with connection pool.

        Args:
            pool: asyncpg connection pool for database access.
                  Pool should be created by the caller (e.g., from DbAdapter).
        """
        self._pool = pool
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name="projector.registration",
            transport_type=EnumInfraTransportType.DATABASE,
        )

    async def initialize_schema(self, correlation_id: UUID | None = None) -> None:
        """Initialize projection schema (create table if not exists).

        Executes the SQL schema file to create the registration_projections
        table and associated indexes. This operation is idempotent.

        Args:
            correlation_id: Optional correlation ID for tracing

        Raises:
            RuntimeHostError: If schema file cannot be read
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If schema creation times out
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="initialize_schema",
            target_name="projector.registration",
            correlation_id=corr_id,
        )

        # Read schema file
        if not _SCHEMA_FILE.exists():
            raise RuntimeHostError(
                f"Schema file not found: {_SCHEMA_FILE}",
                context=ctx,
            )

        schema_sql = _SCHEMA_FILE.read_text()

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("initialize_schema", corr_id)

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(schema_sql)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.info(
                "Registration projection schema initialized",
                extra={"correlation_id": str(corr_id)},
            )

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("initialize_schema", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for schema initialization",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("initialize_schema", corr_id)
            raise InfraTimeoutError(
                "Schema initialization timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("initialize_schema", corr_id)
            raise RuntimeHostError(
                f"Failed to initialize schema: {type(e).__name__}",
                context=ctx,
            ) from e

    async def persist(
        self,
        projection: ModelRegistrationProjection,
        entity_id: UUID,
        domain: str,
        sequence_info: ModelSequenceInfo,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Persist projection with ordering enforcement.

        Atomically persists the projection using an upsert with sequence
        comparison. Rejects stale updates where the incoming sequence is
        older than or equal to the current projection's sequence.

        Args:
            projection: Projection model to persist
            entity_id: Entity identifier (partition key)
            domain: Domain namespace
            sequence_info: Sequence info for ordering validation
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if persisted successfully, False if stale (rejected)

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If persistence times out
            RuntimeHostError: For other database errors

        Example:
            >>> result = await projector.persist(proj, proj.entity_id, "registration", seq)
            >>> if result:
            ...     print("Applied")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="persist",
            target_name="projector.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("persist", corr_id)

        # Build upsert query with sequence comparison
        # The WHERE clause in the ON CONFLICT update ensures we only update
        # if the incoming sequence is newer than the existing one
        upsert_sql = """
            INSERT INTO registration_projections (
                entity_id, domain, current_state, node_type, node_version,
                capabilities, ack_deadline, liveness_deadline,
                ack_timeout_emitted_at, liveness_timeout_emitted_at,
                last_applied_event_id, last_applied_offset,
                last_applied_sequence, last_applied_partition,
                registered_at, updated_at, correlation_id
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (entity_id, domain) DO UPDATE SET
                current_state = EXCLUDED.current_state,
                node_type = EXCLUDED.node_type,
                node_version = EXCLUDED.node_version,
                capabilities = EXCLUDED.capabilities,
                ack_deadline = EXCLUDED.ack_deadline,
                liveness_deadline = EXCLUDED.liveness_deadline,
                ack_timeout_emitted_at = EXCLUDED.ack_timeout_emitted_at,
                liveness_timeout_emitted_at = EXCLUDED.liveness_timeout_emitted_at,
                last_applied_event_id = EXCLUDED.last_applied_event_id,
                last_applied_offset = EXCLUDED.last_applied_offset,
                last_applied_sequence = EXCLUDED.last_applied_sequence,
                last_applied_partition = EXCLUDED.last_applied_partition,
                updated_at = EXCLUDED.updated_at,
                correlation_id = EXCLUDED.correlation_id
            WHERE
                -- Only update if incoming sequence is newer
                EXCLUDED.last_applied_offset > registration_projections.last_applied_offset
                OR (
                    EXCLUDED.last_applied_sequence IS NOT NULL
                    AND (
                        registration_projections.last_applied_sequence IS NULL
                        OR EXCLUDED.last_applied_sequence > registration_projections.last_applied_sequence
                    )
                )
            RETURNING entity_id
        """

        # Prepare parameters
        params = (
            entity_id,
            domain,
            projection.current_state.value,  # Convert enum to string
            projection.node_type,
            projection.node_version,
            projection.capabilities.model_dump_json(),  # JSONB as JSON string
            projection.ack_deadline,
            projection.liveness_deadline,
            projection.ack_timeout_emitted_at,
            projection.liveness_timeout_emitted_at,
            projection.last_applied_event_id,
            sequence_info.offset or sequence_info.sequence,  # Use offset if available
            sequence_info.sequence if sequence_info.offset is None else None,
            sequence_info.partition,
            projection.registered_at,
            projection.updated_at,
            corr_id,
        )

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(upsert_sql, *params)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # If we got a row back, the upsert succeeded (not stale)
            if result:
                logger.debug(
                    "Projection persisted",
                    extra={
                        "entity_id": str(entity_id),
                        "domain": domain,
                        "sequence": sequence_info.sequence,
                        "correlation_id": str(corr_id),
                    },
                )
                return True

            # No row returned means the WHERE clause rejected (stale)
            logger.debug(
                "Stale projection rejected",
                extra={
                    "entity_id": str(entity_id),
                    "domain": domain,
                    "sequence": sequence_info.sequence,
                    "correlation_id": str(corr_id),
                },
            )
            return False

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("persist", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for projection persistence",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("persist", corr_id)
            raise InfraTimeoutError(
                "Projection persistence timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("persist", corr_id)
            raise RuntimeHostError(
                f"Failed to persist projection: {type(e).__name__}",
                context=ctx,
            ) from e

    async def is_stale(
        self,
        entity_id: UUID,
        domain: str,
        sequence_info: ModelSequenceInfo,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Check if sequence info is stale compared to current projection.

        Used for pre-check before processing to avoid unnecessary work.
        This is a point-in-time check; always rely on persist() result
        for correctness.

        Args:
            entity_id: Entity identifier
            domain: Domain namespace
            sequence_info: Incoming sequence to check
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if incoming sequence is stale, False otherwise
            Returns False if no projection exists (not stale)

        Example:
            >>> if await projector.is_stale(entity_id, "registration", seq):
            ...     print("Skip processing - stale event")
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="is_stale",
            target_name="projector.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("is_stale", corr_id)

        query_sql = """
            SELECT last_applied_offset, last_applied_sequence
            FROM registration_projections
            WHERE entity_id = $1 AND domain = $2
        """

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query_sql, entity_id, domain)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if row is None:
                # No existing projection - not stale
                return False

            current_offset = row["last_applied_offset"]
            current_sequence = row["last_applied_sequence"]

            # Build current sequence info for comparison
            current_seq = ModelSequenceInfo(
                sequence=current_sequence or current_offset,
                offset=current_offset,
            )

            return sequence_info.is_stale_compared_to(current_seq)

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("is_stale", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for staleness check",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("is_stale", corr_id)
            raise RuntimeHostError(
                f"Failed to check staleness: {type(e).__name__}",
                context=ctx,
            ) from e


__all__: list[str] = ["ProjectorRegistration"]
