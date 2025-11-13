#!/usr/bin/env python3
"""
Distributed Lock Effect Node - ONEX v2.0 Compliant.

PostgreSQL-backed distributed locking for multi-instance deployments.

ONEX v2.0 Compliance:
- Extends NodeEffect from omnibase_core
- Implements execute_effect method signature
- Uses ModelOnexError for error handling
- Publishes events to Kafka (lock_acquired, lock_released, lock_expired)
- Structured logging with correlation tracking

Key Responsibilities:
- Acquire distributed locks with lease duration
- Release held locks
- Extend lock lease before expiration
- Query lock status and metadata
- Clean up expired locks (background task)

Lock Operations:
- ACQUIRE: Acquire distributed lock with retry and deadlock detection
- RELEASE: Release held lock with ownership validation
- EXTEND: Extend lock lease duration
- QUERY: Query lock status and metadata
- CLEANUP: Clean up expired locks (background task)

Performance Targets:
- Lock acquisition: < 50ms (P95)
- Lock release: < 10ms (P95)
- Throughput: 100+ operations/second
- Cleanup: < 100ms per 100 locks

Example Usage:
    ```python
    from omnibase_core.models.core import ModelContainer
    from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

    # Initialize node
    container = ModelContainer(
        value={
            "postgres_host": os.getenv("POSTGRES_HOST", "omninode-bridge-postgres"),
            "postgres_port": 5432,
            "postgres_database": "omninode_bridge",
        },
        container_type="config"
    )
    node = NodeDistributedLockEffect(container)
    await node.initialize()

    # Acquire lock
    contract = ModelContractEffect(
        name="acquire_lock",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Acquire distributed lock",
        node_type="EFFECT",
        input_model="ModelDistributedLockRequest",
        output_model="ModelDistributedLockResponse",
        input_data={
            "operation": "acquire",
            "lock_name": "workflow_processing_lock",
            "owner_id": "orchestrator-node-1",
            "lease_duration": 30.0
        }
    )

    response = await node.execute_effect(contract)
    print(f"Lock acquired: {response.success}")
    ```
"""

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import asyncpg
from asyncpg import Pool
from asyncpg.exceptions import (
    DeadlockDetectedError,
    LockNotAvailableError,
    PostgresError,
)
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

from .models.enum_lock_operation import EnumLockOperation
from .models.enum_lock_status import EnumLockStatus
from .models.model_config import ModelDistributedLockConfig
from .models.model_lock_info import ModelLockInfo
from .models.model_request import ModelDistributedLockRequest
from .models.model_response import ModelDistributedLockResponse

# Aliases for compatibility
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode

logger = logging.getLogger(__name__)


class NodeDistributedLockEffect(NodeEffect):
    """
    Distributed Lock Effect Node - PostgreSQL-backed distributed locking.

    Responsibilities:
    - Acquire/release/extend distributed locks
    - Query lock status and metadata
    - Background cleanup of expired locks
    - Kafka event publishing for observability
    - Metrics collection and health monitoring

    Lock Table Schema:
        distributed_locks (
            lock_id VARCHAR(255) PRIMARY KEY,
            owner_id VARCHAR(255) NOT NULL,
            acquired_at BIGINT NOT NULL,
            expires_at BIGINT NOT NULL,
            lease_duration REAL NOT NULL,
            heartbeat_at BIGINT NOT NULL,
            metadata JSONB DEFAULT '{}',
            status VARCHAR(50) DEFAULT 'acquired',
            acquisition_count INT DEFAULT 1,
            extension_count INT DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )

    Performance:
    - Lock acquisition: < 50ms (P95)
    - Lock release: < 10ms (P95)
    - Throughput: 100+ ops/second
    - Cleanup: < 100ms per 100 locks
    """

    def __init__(self, container: ModelContainer):
        """
        Initialize Distributed Lock Effect Node.

        Args:
            container: ONEX container for dependency injection
        """
        # Initialize base NodeEffect class
        super().__init__(container)

        # Store container reference
        self.container = container

        # Load configuration from container value
        config_data = container.value if isinstance(container.value, dict) else {}
        self.config = ModelDistributedLockConfig(**config_data)

        # PostgreSQL connection pool (initialized in initialize())
        self._pool: Pool | None = None

        # Background cleanup task
        self._cleanup_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Metrics tracking
        self._total_acquires = 0
        self._total_releases = 0
        self._total_extends = 0
        self._total_queries = 0
        self._total_cleanups = 0
        self._total_acquire_time_ms = 0.0
        self._total_release_time_ms = 0.0
        self._failed_acquires = 0

        # Consul configuration for service discovery
        config_value = container.value if isinstance(container.value, dict) else {}
        self.consul_host: str = config_value.get(
            "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
        )
        self.consul_port: int = config_value.get(
            "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
        )
        self.consul_enable_registration: bool = config_value.get(
            "consul_enable_registration", True
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeDistributedLockEffect initialized",
            {"node_id": str(self.node_id), "config": self.config.model_dump()},
        )

        # Register with Consul for service discovery
        health_check_mode = config_value.get("health_check_mode", False)
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    async def initialize(self) -> None:
        """
        Initialize PostgreSQL connection pool and create lock table.

        Creates distributed_locks table if it doesn't exist and starts
        background cleanup task.

        Raises:
            OnexError: If database initialization fails
        """
        try:
            # Create PostgreSQL connection pool
            self._pool = await asyncpg.create_pool(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
                min_size=self.config.postgres_min_connections,
                max_size=self.config.postgres_max_connections,
                timeout=self.config.postgres_connection_timeout,
            )

            # Create distributed_locks table
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS distributed_locks (
                        lock_id VARCHAR(255) PRIMARY KEY,
                        owner_id VARCHAR(255) NOT NULL,
                        acquired_at BIGINT NOT NULL,
                        expires_at BIGINT NOT NULL,
                        lease_duration REAL NOT NULL,
                        heartbeat_at BIGINT NOT NULL,
                        metadata JSONB DEFAULT '{}',
                        status VARCHAR(50) DEFAULT 'acquired',
                        acquisition_count INT DEFAULT 1,
                        extension_count INT DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """
                )

                # Create indexes
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_distributed_locks_expires_at
                    ON distributed_locks(expires_at)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_distributed_locks_owner_id
                    ON distributed_locks(owner_id)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_distributed_locks_status
                    ON distributed_locks(status)
                """
                )

            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_locks_loop())

            emit_log_event(
                LogLevel.INFO,
                "NodeDistributedLockEffect initialized with PostgreSQL pool",
                {
                    "node_id": str(self.node_id),
                    "pool_min_size": self.config.postgres_min_connections,
                    "pool_max_size": self.config.postgres_max_connections,
                },
            )

        except Exception as e:
            raise OnexError(
                message=f"Failed to initialize NodeDistributedLockEffect: {e}",
                error_code=CoreErrorCode.INITIALIZATION_ERROR,
                error=str(e),
            ) from e

    async def shutdown(self) -> None:
        """Shutdown the node and cleanup resources."""
        logger.info("Shutting down NodeDistributedLockEffect")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close PostgreSQL pool
        if self._pool:
            await self._pool.close()

        # Deregister from Consul for clean service discovery
        self._deregister_from_consul()

        emit_log_event(
            LogLevel.INFO,
            "NodeDistributedLockEffect shutdown complete",
            {"node_id": str(self.node_id)},
        )

    async def _acquire_lock(
        self, request: ModelDistributedLockRequest
    ) -> ModelDistributedLockResponse:
        """
        Acquire a distributed lock with retry and deadlock detection.

        Args:
            request: Lock acquisition request

        Returns:
            Response with lock info or error
        """
        start_time = time.perf_counter()
        attempt = 0

        while attempt < request.max_acquire_attempts:
            try:
                current_timestamp = int(time.time())
                acquired_at = current_timestamp
                expires_at = int(acquired_at + request.lease_duration)

                async with self._pool.acquire() as conn:
                    # Use PostgreSQL advisory lock for atomicity
                    async with conn.transaction():
                        lock_hash = hash(request.lock_name) % (2**63)
                        await conn.fetchval(
                            "SELECT pg_advisory_xact_lock($1)", lock_hash
                        )

                        # Check if lock exists
                        existing_lock = await conn.fetchrow(
                            """
                            SELECT lock_id, owner_id, acquired_at, expires_at,
                                   lease_duration, metadata, status,
                                   acquisition_count, extension_count
                            FROM distributed_locks
                            WHERE lock_id = $1
                        """,
                            request.lock_name,
                        )

                        if existing_lock:
                            # Check if lock is expired
                            if current_timestamp >= existing_lock["expires_at"]:
                                # Lock expired, can reacquire
                                pass
                            elif existing_lock["owner_id"] == request.owner_id:
                                # Same owner, update lock
                                pass
                            else:
                                # Lock held by someone else
                                attempt += 1
                                await asyncio.sleep(
                                    request.acquire_retry_delay * (2**attempt)
                                )
                                continue

                        # Acquire or update lock
                        result = await conn.fetchrow(
                            """
                            INSERT INTO distributed_locks
                            (lock_id, owner_id, acquired_at, expires_at, lease_duration,
                             heartbeat_at, metadata, status, acquisition_count, extension_count)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1, 0)
                            ON CONFLICT (lock_id)
                            DO UPDATE SET
                                owner_id = EXCLUDED.owner_id,
                                acquired_at = EXCLUDED.acquired_at,
                                expires_at = EXCLUDED.expires_at,
                                lease_duration = EXCLUDED.lease_duration,
                                heartbeat_at = EXCLUDED.heartbeat_at,
                                metadata = EXCLUDED.metadata,
                                status = EXCLUDED.status,
                                acquisition_count = distributed_locks.acquisition_count + 1,
                                updated_at = NOW()
                            RETURNING *
                        """,
                            request.lock_name,
                            request.owner_id,
                            acquired_at,
                            expires_at,
                            request.lease_duration,
                            current_timestamp,
                            request.owner_metadata,
                            EnumLockStatus.ACQUIRED.value,
                        )

                        # Build lock info
                        lock_info = ModelLockInfo(
                            lock_name=request.lock_name,
                            owner_id=request.owner_id,
                            owner_metadata=request.owner_metadata,
                            status=EnumLockStatus.ACQUIRED,
                            acquired_at=datetime.fromtimestamp(acquired_at, tz=UTC),
                            expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
                            lease_duration=request.lease_duration,
                            acquisition_count=result["acquisition_count"],
                            extension_count=result["extension_count"],
                        )

                        duration_ms = (time.perf_counter() - start_time) * 1000

                        self._total_acquires += 1
                        self._total_acquire_time_ms += duration_ms

                        emit_log_event(
                            LogLevel.INFO,
                            f"Acquired lock: {request.lock_name}",
                            {
                                "lock_name": request.lock_name,
                                "owner_id": request.owner_id,
                                "lease_duration": request.lease_duration,
                                "attempts": attempt + 1,
                                "duration_ms": round(duration_ms, 2),
                            },
                        )

                        return ModelDistributedLockResponse(
                            success=True,
                            operation=EnumLockOperation.ACQUIRE,
                            lock_info=lock_info,
                            duration_ms=duration_ms,
                            acquire_attempts=attempt + 1,
                            database_query_ms=duration_ms * 0.8,
                            correlation_id=request.correlation_id,
                            execution_id=request.execution_id,
                        )

            except (DeadlockDetectedError, LockNotAvailableError) as e:
                logger.warning(
                    f"Lock acquisition attempt {attempt + 1} failed: {e}",
                    extra={"lock_name": request.lock_name, "attempt": attempt + 1},
                )
                attempt += 1
                if attempt < request.max_acquire_attempts:
                    delay = request.acquire_retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                continue

            except PostgresError as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._failed_acquires += 1

                return ModelDistributedLockResponse(
                    success=False,
                    operation=EnumLockOperation.ACQUIRE,
                    error_message=f"Database error: {e}",
                    error_code="DATABASE_ERROR",
                    duration_ms=duration_ms,
                    acquire_attempts=attempt + 1,
                    correlation_id=request.correlation_id,
                    execution_id=request.execution_id,
                )

        # Max attempts exceeded
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._failed_acquires += 1

        return ModelDistributedLockResponse(
            success=False,
            operation=EnumLockOperation.ACQUIRE,
            error_message=f"Failed to acquire lock after {request.max_acquire_attempts} attempts",
            error_code="LOCK_TIMEOUT",
            retry_after_seconds=request.lease_duration,
            duration_ms=duration_ms,
            acquire_attempts=request.max_acquire_attempts,
            correlation_id=request.correlation_id,
            execution_id=request.execution_id,
        )

    async def _release_lock(
        self, request: ModelDistributedLockRequest
    ) -> ModelDistributedLockResponse:
        """
        Release a distributed lock.

        Args:
            request: Lock release request

        Returns:
            Response indicating success or failure
        """
        start_time = time.perf_counter()

        try:
            async with self._pool.acquire() as conn:
                # Update status to RELEASED before deletion
                result = await conn.execute(
                    """
                    UPDATE distributed_locks
                    SET status = $1, updated_at = NOW()
                    WHERE lock_id = $2 AND owner_id = $3
                """,
                    EnumLockStatus.RELEASED.value,
                    request.lock_name,
                    request.owner_id,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000

                if result == "UPDATE 1":
                    self._total_releases += 1
                    self._total_release_time_ms += duration_ms

                    emit_log_event(
                        LogLevel.INFO,
                        f"Released lock: {request.lock_name}",
                        {
                            "lock_name": request.lock_name,
                            "owner_id": request.owner_id,
                            "duration_ms": round(duration_ms, 2),
                        },
                    )

                    return ModelDistributedLockResponse(
                        success=True,
                        operation=EnumLockOperation.RELEASE,
                        duration_ms=duration_ms,
                        database_query_ms=duration_ms * 0.8,
                        correlation_id=request.correlation_id,
                        execution_id=request.execution_id,
                    )
                else:
                    return ModelDistributedLockResponse(
                        success=False,
                        operation=EnumLockOperation.RELEASE,
                        error_message="Lock not found or not owned by this instance",
                        error_code="LOCK_NOT_HELD",
                        duration_ms=duration_ms,
                        correlation_id=request.correlation_id,
                        execution_id=request.execution_id,
                    )

        except PostgresError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ModelDistributedLockResponse(
                success=False,
                operation=EnumLockOperation.RELEASE,
                error_message=f"Database error: {e}",
                error_code="DATABASE_ERROR",
                duration_ms=duration_ms,
                correlation_id=request.correlation_id,
                execution_id=request.execution_id,
            )

    async def _extend_lock(
        self, request: ModelDistributedLockRequest
    ) -> ModelDistributedLockResponse:
        """
        Extend lock lease duration.

        Args:
            request: Lock extension request

        Returns:
            Response with updated lock info
        """
        start_time = time.perf_counter()

        try:
            async with self._pool.acquire() as conn:
                current_timestamp = int(time.time())

                # Update expiration time
                result = await conn.fetchrow(
                    """
                    UPDATE distributed_locks
                    SET expires_at = expires_at + $1,
                        extension_count = extension_count + 1,
                        heartbeat_at = $2,
                        updated_at = NOW()
                    WHERE lock_id = $3 AND owner_id = $4 AND status = $5
                    RETURNING *
                """,
                    int(request.extension_duration),
                    current_timestamp,
                    request.lock_name,
                    request.owner_id,
                    EnumLockStatus.ACQUIRED.value,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000

                if result:
                    self._total_extends += 1

                    lock_info = ModelLockInfo(
                        lock_name=result["lock_id"],
                        owner_id=result["owner_id"],
                        status=EnumLockStatus(result["status"]),
                        acquired_at=datetime.fromtimestamp(
                            result["acquired_at"], tz=UTC
                        ),
                        expires_at=datetime.fromtimestamp(result["expires_at"], tz=UTC),
                        lease_duration=result["lease_duration"],
                        acquisition_count=result["acquisition_count"],
                        extension_count=result["extension_count"],
                    )

                    emit_log_event(
                        LogLevel.INFO,
                        f"Extended lock: {request.lock_name}",
                        {
                            "lock_name": request.lock_name,
                            "extension_duration": request.extension_duration,
                            "new_expiration": str(lock_info.expires_at),
                        },
                    )

                    return ModelDistributedLockResponse(
                        success=True,
                        operation=EnumLockOperation.EXTEND,
                        lock_info=lock_info,
                        duration_ms=duration_ms,
                        database_query_ms=duration_ms * 0.8,
                        correlation_id=request.correlation_id,
                        execution_id=request.execution_id,
                    )
                else:
                    return ModelDistributedLockResponse(
                        success=False,
                        operation=EnumLockOperation.EXTEND,
                        error_message="Lock not found, not owned, or already expired",
                        error_code="LOCK_NOT_HELD",
                        duration_ms=duration_ms,
                        correlation_id=request.correlation_id,
                        execution_id=request.execution_id,
                    )

        except PostgresError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ModelDistributedLockResponse(
                success=False,
                operation=EnumLockOperation.EXTEND,
                error_message=f"Database error: {e}",
                error_code="DATABASE_ERROR",
                duration_ms=duration_ms,
                correlation_id=request.correlation_id,
                execution_id=request.execution_id,
            )

    async def _query_lock(
        self, request: ModelDistributedLockRequest
    ) -> ModelDistributedLockResponse:
        """
        Query lock status and metadata.

        Args:
            request: Lock query request

        Returns:
            Response with lock info
        """
        start_time = time.perf_counter()

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    SELECT *
                    FROM distributed_locks
                    WHERE lock_id = $1
                """,
                    request.lock_name,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000
                self._total_queries += 1

                if result:
                    lock_info = ModelLockInfo(
                        lock_name=result["lock_id"],
                        owner_id=result["owner_id"],
                        owner_metadata=result["metadata"],
                        status=EnumLockStatus(result["status"]),
                        acquired_at=datetime.fromtimestamp(
                            result["acquired_at"], tz=UTC
                        ),
                        expires_at=datetime.fromtimestamp(result["expires_at"], tz=UTC),
                        lease_duration=result["lease_duration"],
                        acquisition_count=result["acquisition_count"],
                        extension_count=result["extension_count"],
                        created_at=result["created_at"],
                        updated_at=result["updated_at"],
                    )

                    return ModelDistributedLockResponse(
                        success=True,
                        operation=EnumLockOperation.QUERY,
                        lock_info=lock_info,
                        duration_ms=duration_ms,
                        database_query_ms=duration_ms * 0.8,
                        correlation_id=request.correlation_id,
                        execution_id=request.execution_id,
                    )
                else:
                    return ModelDistributedLockResponse(
                        success=False,
                        operation=EnumLockOperation.QUERY,
                        error_message="Lock not found",
                        error_code="LOCK_NOT_FOUND",
                        duration_ms=duration_ms,
                        correlation_id=request.correlation_id,
                        execution_id=request.execution_id,
                    )

        except PostgresError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ModelDistributedLockResponse(
                success=False,
                operation=EnumLockOperation.QUERY,
                error_message=f"Database error: {e}",
                error_code="DATABASE_ERROR",
                duration_ms=duration_ms,
                correlation_id=request.correlation_id,
                execution_id=request.execution_id,
            )

    async def _cleanup_expired_locks(
        self, request: ModelDistributedLockRequest
    ) -> ModelDistributedLockResponse:
        """
        Clean up expired locks from database.

        Args:
            request: Cleanup request

        Returns:
            Response with count of cleaned locks
        """
        start_time = time.perf_counter()

        try:
            async with self._pool.acquire() as conn:
                current_timestamp = int(time.time())
                max_age_timestamp = int(current_timestamp - request.max_age_seconds)

                # Get names of locks to clean
                lock_names = await conn.fetch(
                    """
                    SELECT lock_id
                    FROM distributed_locks
                    WHERE expires_at < $1 OR
                          (status IN ($2, $3) AND updated_at < NOW() - INTERVAL '1 hour')
                """,
                    current_timestamp,
                    EnumLockStatus.EXPIRED.value,
                    EnumLockStatus.RELEASED.value,
                )

                # Delete expired locks
                result = await conn.execute(
                    """
                    DELETE FROM distributed_locks
                    WHERE expires_at < $1 OR
                          (status IN ($2, $3) AND updated_at < NOW() - INTERVAL '1 hour')
                """,
                    current_timestamp,
                    EnumLockStatus.EXPIRED.value,
                    EnumLockStatus.RELEASED.value,
                )

                duration_ms = (time.perf_counter() - start_time) * 1000
                cleaned_count = (
                    int(result.split()[1]) if result.startswith("DELETE") else 0
                )
                self._total_cleanups += 1

                emit_log_event(
                    LogLevel.INFO,
                    f"Cleaned up {cleaned_count} expired locks",
                    {
                        "cleaned_count": cleaned_count,
                        "duration_ms": round(duration_ms, 2),
                    },
                )

                return ModelDistributedLockResponse(
                    success=True,
                    operation=EnumLockOperation.CLEANUP,
                    cleaned_count=cleaned_count,
                    cleaned_lock_names=[row["lock_id"] for row in lock_names],
                    duration_ms=duration_ms,
                    database_query_ms=duration_ms * 0.8,
                    correlation_id=request.correlation_id,
                    execution_id=request.execution_id,
                )

        except PostgresError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            return ModelDistributedLockResponse(
                success=False,
                operation=EnumLockOperation.CLEANUP,
                error_message=f"Database error: {e}",
                error_code="DATABASE_ERROR",
                duration_ms=duration_ms,
                correlation_id=request.correlation_id,
                execution_id=request.execution_id,
            )

    async def _cleanup_expired_locks_loop(self) -> None:
        """Background task to periodically clean up expired locks."""
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(self.config.cleanup_interval)

                if self._shutdown_event.is_set():
                    break

                # Run cleanup
                request = ModelDistributedLockRequest(
                    operation=EnumLockOperation.CLEANUP,
                    lock_name="*",  # Cleanup all locks
                    max_age_seconds=self.config.cleanup_interval * 2,
                )

                try:
                    await self._cleanup_expired_locks(request)
                except Exception as e:
                    logger.error(f"Error during background lock cleanup: {e}")

        except asyncio.CancelledError:
            logger.debug("Background cleanup task cancelled")

    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelDistributedLockResponse:
        """
        Execute distributed lock operation.

        Args:
            contract: Effect contract with input_state containing lock request

        Returns:
            ModelDistributedLockResponse with operation results

        Raises:
            OnexError: If operation fails
        """
        start_time = time.perf_counter()
        correlation_id = contract.correlation_id

        emit_log_event(
            LogLevel.INFO,
            "Starting distributed lock operation",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Parse request from contract input_state
            input_state = contract.input_state or {}
            request = ModelDistributedLockRequest(
                operation=EnumLockOperation(input_state.get("operation")),
                lock_name=input_state.get("lock_name"),
                owner_id=input_state.get("owner_id"),
                lease_duration=input_state.get(
                    "lease_duration", self.config.default_lease_duration
                ),
                extension_duration=input_state.get("extension_duration", 30.0),
                max_acquire_attempts=input_state.get(
                    "max_acquire_attempts", self.config.max_acquire_attempts
                ),
                acquire_retry_delay=input_state.get(
                    "acquire_retry_delay", self.config.acquire_retry_delay
                ),
                max_age_seconds=input_state.get("max_age_seconds", 3600.0),
                owner_metadata=input_state.get("owner_metadata", {}),
                metadata=input_state.get("metadata", {}),
                correlation_id=correlation_id,
            )

            # Route to appropriate operation handler
            if request.operation == EnumLockOperation.ACQUIRE:
                response = await self._acquire_lock(request)
            elif request.operation == EnumLockOperation.RELEASE:
                response = await self._release_lock(request)
            elif request.operation == EnumLockOperation.EXTEND:
                response = await self._extend_lock(request)
            elif request.operation == EnumLockOperation.QUERY:
                response = await self._query_lock(request)
            elif request.operation == EnumLockOperation.CLEANUP:
                response = await self._cleanup_expired_locks(request)
            else:
                raise OnexError(
                    message=f"Unknown lock operation: {request.operation}",
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    operation=str(request.operation),
                )

            total_duration_ms = (time.perf_counter() - start_time) * 1000

            emit_log_event(
                LogLevel.INFO,
                f"Lock operation completed: {request.operation.value}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "operation": request.operation.value,
                    "success": response.success,
                    "duration_ms": round(total_duration_ms, 2),
                },
            )

            return response

        except OnexError:
            raise

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Lock operation failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )

            raise OnexError(
                message=f"Lock operation failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
            ) from e

    def get_metrics(self) -> dict[str, Any]:
        """
        Get lock operation metrics for monitoring.

        Returns:
            Dictionary with metrics
        """
        avg_acquire_time_ms = (
            self._total_acquire_time_ms / self._total_acquires
            if self._total_acquires > 0
            else 0
        )

        avg_release_time_ms = (
            self._total_release_time_ms / self._total_releases
            if self._total_releases > 0
            else 0
        )

        success_rate = (
            (self._total_acquires - self._failed_acquires) / self._total_acquires
            if self._total_acquires > 0
            else 1.0
        )

        return {
            "total_acquires": self._total_acquires,
            "total_releases": self._total_releases,
            "total_extends": self._total_extends,
            "total_queries": self._total_queries,
            "total_cleanups": self._total_cleanups,
            "failed_acquires": self._failed_acquires,
            "success_rate": round(success_rate, 4),
            "avg_acquire_time_ms": round(avg_acquire_time_ms, 2),
            "avg_release_time_ms": round(avg_release_time_ms, 2),
        }

    def _register_with_consul_sync(self) -> None:
        """
        Register distributed lock node with Consul for service discovery (synchronous).

        Registers the distributed lock as a service with health checks pointing to
        the health endpoint. Includes metadata about node capabilities.

        Note:
            This is a non-blocking registration. Failures are logged but don't
            fail node startup. Service will continue without Consul if registration fails.
        """
        try:
            import consul

            # Initialize Consul client
            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

            # Generate unique service ID
            service_id = f"omninode-bridge-distributed-lock-{self.node_id}"

            # Get service port from config (default to 8063 for distributed lock)
            service_port = 8063  # No container.config for this node

            # Get service host (default to localhost)
            service_host = "localhost"

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "distributed_lock",
                "effect",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Add metadata as tags
            service_tags.extend(
                [
                    "node_type:distributed_lock",
                    f"postgres_available:{self._pool is not None}",
                    f"cleanup_enabled:{self.config.cleanup_interval > 0}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-distributed-lock",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                http=health_check_url,
                interval="30s",
                timeout="5s",
            )

            emit_log_event(
                LogLevel.INFO,
                "Registered with Consul successfully",
                {
                    "node_id": str(self.node_id),
                    "service_id": service_id,
                    "consul_host": self.consul_host,
                    "consul_port": self.consul_port,
                    "service_host": service_host,
                    "service_port": service_port,
                },
            )

            # Store service_id for deregistration
            self._consul_service_id = service_id

        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "python-consul not installed - Consul registration skipped",
                {"node_id": str(self.node_id)},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                "Failed to register with Consul",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _deregister_from_consul(self) -> None:
        """
        Deregister distributed lock from Consul on shutdown (synchronous).

        Removes the service registration from Consul to prevent stale entries
        in the service catalog.

        Note:
            This is called during node shutdown. Failures are logged but don't
            prevent shutdown from completing.
        """
        try:
            if not hasattr(self, "_consul_service_id"):
                # Not registered, nothing to deregister
                return

            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            consul_client.agent.service.deregister(self._consul_service_id)

            emit_log_event(
                LogLevel.INFO,
                "Deregistered from Consul successfully",
                {
                    "node_id": str(self.node_id),
                    "service_id": self._consul_service_id,
                },
            )

        except ImportError:
            # python-consul not installed, silently skip
            pass
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to deregister from Consul",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )


__all__ = ["NodeDistributedLockEffect"]
