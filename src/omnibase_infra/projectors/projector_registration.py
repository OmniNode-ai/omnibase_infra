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
from datetime import datetime
from pathlib import Path
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

# Path to SQL schema file.
# Resolution: projector_registration.py → projectors/ → omnibase_infra/ → schemas/
# This navigates up two levels from this file to reach the schemas directory,
# which is a sibling of the projectors directory within omnibase_infra.
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
                  Pool should be created by the caller (e.g., from DbHandler).
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
            entity_id: Entity identifier (partition key). Passed as a separate
                parameter (rather than using projection.entity_id) for API
                clarity and to support protocol implementations where the
                entity_id may come from external context (e.g., message routing
                metadata) rather than the projection payload itself.
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
        # CORRELATION ID DISTINCTION:
        # - corr_id: Infrastructure-level tracing for THIS method invocation
        #   (circuit breaker, error context, logging). Generated if not provided.
        # - projection.correlation_id: Business-level tracing stored with the
        #   projection data, representing the original event's correlation chain.
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

        # ==========================================================================
        # UPSERT WITH ORDERING ENFORCEMENT (Stale Update Rejection)
        # ==========================================================================
        #
        # This query performs an atomic INSERT-or-UPDATE with sequence validation.
        # If a row with the same (entity_id, domain) exists, the ON CONFLICT clause
        # triggers an UPDATE - but ONLY if the WHERE clause evaluates to true.
        #
        # KEY CONCEPT: EXCLUDED
        # ---------------------
        # In PostgreSQL's ON CONFLICT clause, EXCLUDED is a special pseudo-table
        # that contains the values from the INSERT statement that triggered the
        # conflict. For example, EXCLUDED.current_state refers to the $3 parameter
        # (the new state we're trying to insert), while registration_projections.*
        # refers to the existing row in the table.
        #
        # KEY CONCEPT: Stale Update Rejection
        # ------------------------------------
        # "Stale" means the incoming event is older than what we've already processed.
        # The WHERE clause compares sequence numbers to ensure we never regress state.
        # If WHERE evaluates to false, PostgreSQL skips the UPDATE entirely - the
        # existing row remains unchanged, and RETURNING returns no rows.
        #
        # ORDERING MODES (mutually exclusive):
        # ------------------------------------
        # 1. KAFKA-BASED: When partition IS NOT NULL, we use offset for ordering.
        #    Kafka guarantees ordering within a partition, so (partition, offset)
        #    provides a total order for events affecting the same entity.
        #
        # 2. SEQUENCE-BASED: When partition IS NULL (non-Kafka transports like HTTP),
        #    we fall back to a generic sequence number for ordering validation.
        #
        # The conditions are mutually exclusive - exactly one branch will match
        # based on whether the incoming event has a partition value.
        # ==========================================================================
        upsert_sql = """
            INSERT INTO registration_projections (
                entity_id, domain, current_state, node_type, node_version,
                capabilities, ack_deadline, liveness_deadline, last_heartbeat_at,
                ack_timeout_emitted_at, liveness_timeout_emitted_at,
                last_applied_event_id, last_applied_offset,
                last_applied_sequence, last_applied_partition,
                registered_at, updated_at, correlation_id
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
            )
            ON CONFLICT (entity_id, domain) DO UPDATE SET
                -- EXCLUDED.* refers to values from the INSERT that triggered conflict
                current_state = EXCLUDED.current_state,
                node_type = EXCLUDED.node_type,
                node_version = EXCLUDED.node_version,
                capabilities = EXCLUDED.capabilities,
                ack_deadline = EXCLUDED.ack_deadline,
                liveness_deadline = EXCLUDED.liveness_deadline,
                last_heartbeat_at = EXCLUDED.last_heartbeat_at,
                ack_timeout_emitted_at = EXCLUDED.ack_timeout_emitted_at,
                liveness_timeout_emitted_at = EXCLUDED.liveness_timeout_emitted_at,
                last_applied_event_id = EXCLUDED.last_applied_event_id,
                last_applied_offset = EXCLUDED.last_applied_offset,
                last_applied_sequence = EXCLUDED.last_applied_sequence,
                last_applied_partition = EXCLUDED.last_applied_partition,
                updated_at = EXCLUDED.updated_at,
                correlation_id = EXCLUDED.correlation_id
            WHERE
                -- Stale update rejection: only proceed if incoming event is NEWER.
                -- The two branches below are MUTUALLY EXCLUSIVE based on partition.
                (
                    -- MODE 1: Kafka-based ordering (has partition).
                    -- Use offset comparison within the partition context.
                    EXCLUDED.last_applied_partition IS NOT NULL
                    AND EXCLUDED.last_applied_offset > registration_projections.last_applied_offset
                )
                OR (
                    -- MODE 2: Sequence-based ordering (no partition, e.g., HTTP transport).
                    -- Use generic sequence number for non-Kafka event sources.
                    EXCLUDED.last_applied_partition IS NULL
                    AND EXCLUDED.last_applied_sequence IS NOT NULL
                    AND (
                        -- Allow update if no previous sequence (first event for this entity)
                        registration_projections.last_applied_sequence IS NULL
                        -- Or if incoming sequence is strictly greater (newer event)
                        OR EXCLUDED.last_applied_sequence > registration_projections.last_applied_sequence
                    )
                )
            RETURNING entity_id
            -- RETURNING: If WHERE was false (stale), no rows returned; if true, entity_id returned
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
            projection.last_heartbeat_at,
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
        # corr_id is for infrastructure-level tracing of THIS method invocation.
        # See persist() for full correlation ID distinction explanation.
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
            SELECT last_applied_offset, last_applied_sequence, last_applied_partition
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
            current_partition = row["last_applied_partition"]

            # Build current sequence info for comparison.
            # We include partition to ensure the comparison logic matches persist():
            # - When partition is set: uses Kafka-based ordering (partition + offset)
            # - When partition is None: uses generic sequence-based ordering
            # This alignment prevents false staleness results from mode mismatch.
            current_seq = ModelSequenceInfo(
                sequence=current_sequence or current_offset,
                offset=current_offset,
                partition=current_partition,
            )

            return sequence_info.is_stale_compared_to(current_seq)

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("is_stale", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for staleness check",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("is_stale", corr_id)
            raise InfraTimeoutError(
                "Staleness check timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("is_stale", corr_id)
            raise RuntimeHostError(
                f"Failed to check staleness: {type(e).__name__}",
                context=ctx,
            ) from e

    async def update_ack_timeout_marker(
        self,
        entity_id: UUID,
        domain: str,
        emitted_at: datetime,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Update ack timeout emission marker to prevent duplicate events.

        Sets the ack_timeout_emitted_at column to mark that an ack timeout
        event has been emitted for this entity. This prevents duplicate
        emission during restarts or retry scenarios.

        This method is called AFTER successful event publish to ensure
        exactly-once semantics. If publish succeeded but this fails,
        the event may be duplicated on retry (at-least-once delivery).

        Args:
            entity_id: Node UUID to update
            domain: Domain namespace
            emitted_at: Timestamp when the timeout event was emitted
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if marker was updated, False if entity not found

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If update times out
            RuntimeHostError: For other database errors

        Example:
            >>> await projector.update_ack_timeout_marker(
            ...     entity_id=node_id,
            ...     domain="registration",
            ...     emitted_at=datetime.now(UTC),
            ... )
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="update_ack_timeout_marker",
            target_name="projector.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("update_ack_timeout_marker", corr_id)

        update_sql = """
            UPDATE registration_projections
            SET ack_timeout_emitted_at = $3,
                updated_at = $3
            WHERE entity_id = $1 AND domain = $2
            RETURNING entity_id
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(update_sql, entity_id, domain, emitted_at)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if result:
                logger.debug(
                    "Ack timeout marker updated",
                    extra={
                        "entity_id": str(entity_id),
                        "domain": domain,
                        "emitted_at": emitted_at.isoformat(),
                        "correlation_id": str(corr_id),
                    },
                )
                return True

            logger.warning(
                "Entity not found for ack timeout marker update",
                extra={
                    "entity_id": str(entity_id),
                    "domain": domain,
                    "correlation_id": str(corr_id),
                },
            )
            return False

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("update_ack_timeout_marker", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for marker update",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("update_ack_timeout_marker", corr_id)
            raise InfraTimeoutError(
                "Marker update timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("update_ack_timeout_marker", corr_id)
            raise RuntimeHostError(
                f"Failed to update ack timeout marker: {type(e).__name__}",
                context=ctx,
            ) from e

    async def update_liveness_timeout_marker(
        self,
        entity_id: UUID,
        domain: str,
        emitted_at: datetime,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Update liveness timeout emission marker to prevent duplicate events.

        Sets the liveness_timeout_emitted_at column to mark that a liveness
        expiration event has been emitted for this entity. This prevents
        duplicate emission during restarts or retry scenarios.

        This method is called AFTER successful event publish to ensure
        exactly-once semantics. If publish succeeded but this fails,
        the event may be duplicated on retry (at-least-once delivery).

        Args:
            entity_id: Node UUID to update
            domain: Domain namespace
            emitted_at: Timestamp when the expiration event was emitted
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if marker was updated, False if entity not found

        Raises:
            InfraConnectionError: If database connection fails
            InfraTimeoutError: If update times out
            RuntimeHostError: For other database errors

        Example:
            >>> await projector.update_liveness_timeout_marker(
            ...     entity_id=node_id,
            ...     domain="registration",
            ...     emitted_at=datetime.now(UTC),
            ... )
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="update_liveness_timeout_marker",
            target_name="projector.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("update_liveness_timeout_marker", corr_id)

        update_sql = """
            UPDATE registration_projections
            SET liveness_timeout_emitted_at = $3,
                updated_at = $3
            WHERE entity_id = $1 AND domain = $2
            RETURNING entity_id
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(update_sql, entity_id, domain, emitted_at)

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if result:
                logger.debug(
                    "Liveness timeout marker updated",
                    extra={
                        "entity_id": str(entity_id),
                        "domain": domain,
                        "emitted_at": emitted_at.isoformat(),
                        "correlation_id": str(corr_id),
                    },
                )
                return True

            logger.warning(
                "Entity not found for liveness timeout marker update",
                extra={
                    "entity_id": str(entity_id),
                    "domain": domain,
                    "correlation_id": str(corr_id),
                },
            )
            return False

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "update_liveness_timeout_marker", corr_id
                )
            raise InfraConnectionError(
                "Failed to connect to database for marker update",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "update_liveness_timeout_marker", corr_id
                )
            raise InfraTimeoutError(
                "Marker update timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    "update_liveness_timeout_marker", corr_id
                )
            raise RuntimeHostError(
                f"Failed to update liveness timeout marker: {type(e).__name__}",
                context=ctx,
            ) from e

    async def update_heartbeat(
        self,
        entity_id: UUID,
        domain: str,
        last_heartbeat_at: datetime,
        liveness_deadline: datetime,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Update heartbeat tracking fields for a node registration.

        Updates the `last_heartbeat_at` and `liveness_deadline` columns
        when a heartbeat is received from a node. This extends the node's
        liveness window and records when the heartbeat was received.

        This method is called by the heartbeat handler when processing
        NodeHeartbeatReceived events.

        Args:
            entity_id: Node UUID to update.
            domain: Domain namespace.
            last_heartbeat_at: Timestamp when heartbeat was received.
            liveness_deadline: New deadline for next heartbeat.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if heartbeat was updated, False if entity not found.

        Raises:
            InfraConnectionError: If database connection fails.
            InfraTimeoutError: If update times out.
            RuntimeHostError: For other database errors.

        Example:
            >>> await projector.update_heartbeat(
            ...     entity_id=node_id,
            ...     domain="registration",
            ...     last_heartbeat_at=datetime.now(UTC),
            ...     liveness_deadline=datetime.now(UTC) + timedelta(seconds=90),
            ... )

        Related:
            - OMN-1006: Add last_heartbeat_at for liveness expired event reporting
            - handler_node_heartbeat.py: Handler that calls this method
        """
        corr_id = correlation_id or uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="update_heartbeat",
            target_name="projector.registration",
            correlation_id=corr_id,
        )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("update_heartbeat", corr_id)

        update_sql = """
            UPDATE registration_projections
            SET last_heartbeat_at = $3,
                liveness_deadline = $4,
                updated_at = $3
            WHERE entity_id = $1 AND domain = $2
            RETURNING entity_id
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(
                    update_sql,
                    entity_id,
                    domain,
                    last_heartbeat_at,
                    liveness_deadline,
                )

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            if result:
                logger.debug(
                    "Heartbeat updated",
                    extra={
                        "entity_id": str(entity_id),
                        "domain": domain,
                        "last_heartbeat_at": last_heartbeat_at.isoformat(),
                        "liveness_deadline": liveness_deadline.isoformat(),
                        "correlation_id": str(corr_id),
                    },
                )
                return True

            logger.warning(
                "Entity not found for heartbeat update",
                extra={
                    "entity_id": str(entity_id),
                    "domain": domain,
                    "correlation_id": str(corr_id),
                },
            )
            return False

        except asyncpg.PostgresConnectionError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("update_heartbeat", corr_id)
            raise InfraConnectionError(
                "Failed to connect to database for heartbeat update",
                context=ctx,
            ) from e

        except asyncpg.QueryCanceledError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("update_heartbeat", corr_id)
            raise InfraTimeoutError(
                "Heartbeat update timed out",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("update_heartbeat", corr_id)
            raise RuntimeHostError(
                f"Failed to update heartbeat: {type(e).__name__}",
                context=ctx,
            ) from e


__all__: list[str] = ["ProjectorRegistration"]
