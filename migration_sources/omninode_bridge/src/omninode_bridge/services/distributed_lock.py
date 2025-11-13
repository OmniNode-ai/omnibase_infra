"""Distributed locking service for multi-instance deployments."""

import asyncio
import logging
import time
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from asyncpg import Pool
from asyncpg.exceptions import (
    ConnectionFailureError,
    DeadlockDetectedError,
    InterfaceError,
    LockNotAvailableError,
    PostgresError,
)

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when a lock cannot be acquired."""

    pass


class LockTimeoutError(LockAcquisitionError):
    """Raised when lock acquisition times out."""

    pass


class LockAlreadyHeldError(LockAcquisitionError):
    """Raised when trying to acquire a lock that is already held by this instance."""

    pass


class LockStatus(Enum):
    """Status of distributed lock."""

    AVAILABLE = "available"
    ACQUIRED = "acquired"
    EXPIRED = "expired"
    RELEASED = "released"


@dataclass
class LockInfo:
    """Information about a distributed lock."""

    lock_id: str
    owner_id: str
    acquired_at: float
    expires_at: float
    lease_duration: float
    metadata: dict[str, Any]

    @property
    def is_expired(self) -> bool:
        """Check if the lock has expired."""
        return time.time() > self.expires_at

    @property
    def remaining_time(self) -> float:
        """Get remaining time before lock expires."""
        return max(0, self.expires_at - time.time())


class DistributedLockManager:
    """PostgreSQL-backed distributed locking for multi-instance deployments."""

    def __init__(
        self,
        postgres_pool: Pool,
        instance_id: Optional[str] = None,
        default_lease_duration: float = 30.0,
        cleanup_interval: float = 60.0,
        max_acquire_attempts: int = 3,
    ):
        """
        Initialize distributed lock manager.

        Args:
            postgres_pool: PostgreSQL connection pool
            instance_id: Unique identifier for this instance (auto-generated if None)
            default_lease_duration: Default lock lease duration in seconds
            cleanup_interval: Interval for cleaning up expired locks in seconds
            max_acquire_attempts: Maximum attempts to acquire a lock
        """
        self.pool = postgres_pool
        self.instance_id = instance_id or str(uuid.uuid4())
        self.default_lease_duration = default_lease_duration
        self.cleanup_interval = cleanup_interval
        self.max_acquire_attempts = max_acquire_attempts

        # Track locks held by this instance
        self._held_locks: dict[str, LockInfo] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._heartbeat_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the lock manager and create required database objects."""
        try:
            async with self.pool.acquire() as conn:
                # Create locks table with proper indexes
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
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """
                )

                # Create indexes for efficient queries
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

                # Create trigger to update updated_at
                await conn.execute(
                    """
                    CREATE OR REPLACE FUNCTION update_distributed_locks_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = NOW();
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """
                )

                await conn.execute(
                    """
                    DROP TRIGGER IF EXISTS trigger_update_distributed_locks_updated_at
                    ON distributed_locks;
                """
                )

                await conn.execute(
                    """
                    CREATE TRIGGER trigger_update_distributed_locks_updated_at
                        BEFORE UPDATE ON distributed_locks
                        FOR EACH ROW
                        EXECUTE FUNCTION update_distributed_locks_updated_at();
                """
                )

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_locks())
            logger.info(
                f"Distributed lock manager initialized for instance {self.instance_id}"
            )

        except PostgresError as e:
            logger.error(f"Failed to initialize distributed lock manager: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during lock manager initialization: {e}")
            raise

    async def acquire_lock(
        self,
        lock_id: str,
        lease_duration: Optional[float] = None,
        timeout: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LockInfo:
        """
        Acquire a distributed lock.

        Args:
            lock_id: Unique identifier for the lock
            lease_duration: How long to hold the lock (seconds)
            timeout: Maximum time to wait for lock acquisition (seconds)
            metadata: Additional metadata to store with the lock

        Returns:
            LockInfo object containing lock details

        Raises:
            LockTimeoutError: If lock cannot be acquired within timeout
            LockAlreadyHeldError: If this instance already holds the lock
            LockAcquisitionError: If lock acquisition fails for other reasons
        """
        if lock_id in self._held_locks:
            raise LockAlreadyHeldError(
                f"Lock {lock_id} is already held by this instance"
            )

        lease_duration = lease_duration or self.default_lease_duration
        timeout = timeout or (
            lease_duration / 2
        )  # Default timeout is half lease duration
        metadata = metadata or {}

        start_time = time.time()
        attempt = 0

        while attempt < self.max_acquire_attempts:
            try:
                current_time = time.time()
                if current_time - start_time > timeout:
                    raise LockTimeoutError(
                        f"Failed to acquire lock {lock_id} within {timeout}s"
                    )

                async with self.pool.acquire() as conn:
                    # Try to acquire lock using PostgreSQL's advisory lock for atomicity
                    async with conn.transaction():
                        # Use PostgreSQL advisory lock to ensure atomicity of lock acquisition
                        lock_hash = hash(lock_id) % (2**63)  # Convert to valid bigint
                        await conn.fetchval(
                            "SELECT pg_advisory_xact_lock($1)", lock_hash
                        )

                        # Check if lock exists and is still valid
                        existing_lock = await conn.fetchrow(
                            """
                            SELECT lock_id, owner_id, acquired_at, expires_at, lease_duration, metadata
                            FROM distributed_locks
                            WHERE lock_id = $1
                        """,
                            lock_id,
                        )

                        current_timestamp = int(time.time())

                        if existing_lock:
                            # Check if existing lock has expired
                            if current_timestamp < existing_lock["expires_at"]:
                                # Lock is still valid and held by someone else
                                if existing_lock["owner_id"] != self.instance_id:
                                    attempt += 1
                                    await asyncio.sleep(min(0.1 * (2**attempt), 1.0))
                                    continue
                                else:
                                    # We already own this lock somehow - update it
                                    logger.warning(
                                        f"Updating existing lock {lock_id} owned by this instance"
                                    )

                        # Acquire or update the lock
                        acquired_at = current_timestamp
                        expires_at = int(acquired_at + lease_duration)

                        await conn.execute(
                            """
                            INSERT INTO distributed_locks
                            (lock_id, owner_id, acquired_at, expires_at, lease_duration, heartbeat_at, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            ON CONFLICT (lock_id)
                            DO UPDATE SET
                                owner_id = EXCLUDED.owner_id,
                                acquired_at = EXCLUDED.acquired_at,
                                expires_at = EXCLUDED.expires_at,
                                lease_duration = EXCLUDED.lease_duration,
                                heartbeat_at = EXCLUDED.heartbeat_at,
                                metadata = EXCLUDED.metadata
                        """,
                            lock_id,
                            self.instance_id,
                            acquired_at,
                            expires_at,
                            lease_duration,
                            current_timestamp,
                            metadata,
                        )

                        # Create lock info
                        lock_info = LockInfo(
                            lock_id=lock_id,
                            owner_id=self.instance_id,
                            acquired_at=float(acquired_at),
                            expires_at=float(expires_at),
                            lease_duration=lease_duration,
                            metadata=metadata,
                        )

                        # Track the lock
                        self._held_locks[lock_id] = lock_info

                        # Start heartbeat task if lease duration is significant
                        if lease_duration > 10:
                            self._heartbeat_tasks[lock_id] = asyncio.create_task(
                                self._maintain_lock_heartbeat(lock_id)
                            )

                        logger.info(
                            f"Acquired distributed lock {lock_id} for {lease_duration}s "
                            f"(expires at {expires_at})"
                        )
                        return lock_info

            except (DeadlockDetectedError, LockNotAvailableError) as e:
                logger.warning(
                    f"Lock acquisition attempt {attempt + 1} failed due to contention: {e}"
                )
                attempt += 1
                if attempt < self.max_acquire_attempts:
                    # Exponential backoff with jitter
                    delay = min(0.1 * (2**attempt), 2.0) + (time.time() % 0.1)
                    await asyncio.sleep(delay)
                continue

            except (ConnectionFailureError, InterfaceError) as e:
                logger.error(f"Database connection error during lock acquisition: {e}")
                raise LockAcquisitionError(f"Database connection failed: {e}")

            except PostgresError as e:
                logger.error(f"Database error during lock acquisition: {e}")
                raise LockAcquisitionError(f"Database error: {e}")

            except Exception as e:
                logger.error(f"Unexpected error during lock acquisition: {e}")
                raise LockAcquisitionError(f"Unexpected error: {e}")

        raise LockTimeoutError(
            f"Failed to acquire lock {lock_id} after {self.max_acquire_attempts} attempts"
        )

    async def release_lock(self, lock_id: str) -> bool:
        """
        Release a distributed lock.

        Args:
            lock_id: Unique identifier for the lock

        Returns:
            True if lock was successfully released, False if lock was not held by this instance

        Raises:
            LockAcquisitionError: If lock release fails due to database error
        """
        if lock_id not in self._held_locks:
            logger.warning(
                f"Attempted to release lock {lock_id} not held by this instance"
            )
            return False

        try:
            # Cancel heartbeat task if running
            if lock_id in self._heartbeat_tasks:
                self._heartbeat_tasks[lock_id].cancel()
                del self._heartbeat_tasks[lock_id]

            async with self.pool.acquire() as conn:
                # Only delete if we still own the lock
                result = await conn.execute(
                    """
                    DELETE FROM distributed_locks
                    WHERE lock_id = $1 AND owner_id = $2
                """,
                    lock_id,
                    self.instance_id,
                )

                # Remove from held locks
                del self._held_locks[lock_id]

                if result == "DELETE 1":
                    logger.info(f"Released distributed lock {lock_id}")
                    return True
                else:
                    logger.warning(
                        f"Lock {lock_id} was not owned by this instance during release"
                    )
                    return False

        except PostgresError as e:
            logger.error(f"Database error during lock release: {e}")
            raise LockAcquisitionError(f"Database error during lock release: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during lock release: {e}")
            raise LockAcquisitionError(f"Unexpected error during lock release: {e}")

    @asynccontextmanager
    async def acquire_lock_context(
        self,
        lock_id: str,
        lease_duration: Optional[float] = None,
        timeout: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AbstractAsyncContextManager[LockInfo]:
        """
        Context manager for acquiring and automatically releasing a distributed lock.

        Args:
            lock_id: Unique identifier for the lock
            lease_duration: How long to hold the lock (seconds)
            timeout: Maximum time to wait for lock acquisition (seconds)
            metadata: Additional metadata to store with the lock

        Yields:
            LockInfo object containing lock details
        """
        lock_info = await self.acquire_lock(lock_id, lease_duration, timeout, metadata)
        try:
            yield lock_info
        finally:
            try:
                await self.release_lock(lock_id)
            except Exception as e:
                logger.error(f"Error releasing lock {lock_id} in context manager: {e}")

    async def extend_lock(self, lock_id: str, additional_time: float) -> bool:
        """
        Extend the lease duration of a held lock.

        Args:
            lock_id: Unique identifier for the lock
            additional_time: Additional time to add to the lease (seconds)

        Returns:
            True if lock was successfully extended, False otherwise
        """
        if lock_id not in self._held_locks:
            logger.warning(f"Cannot extend lock {lock_id} - not held by this instance")
            return False

        try:
            async with self.pool.acquire() as conn:
                current_time = int(time.time())
                new_expires_at = int(
                    self._held_locks[lock_id].expires_at + additional_time
                )

                result = await conn.execute(
                    """
                    UPDATE distributed_locks
                    SET expires_at = $1, heartbeat_at = $2
                    WHERE lock_id = $3 AND owner_id = $4 AND expires_at > $2
                """,
                    new_expires_at,
                    current_time,
                    lock_id,
                    self.instance_id,
                )

                if result == "UPDATE 1":
                    # Update local tracking
                    self._held_locks[lock_id].expires_at = float(new_expires_at)
                    logger.info(f"Extended lock {lock_id} by {additional_time}s")
                    return True
                else:
                    logger.warning(
                        f"Failed to extend lock {lock_id} - may have expired or been released"
                    )
                    # Remove from local tracking since we no longer hold it
                    if lock_id in self._held_locks:
                        del self._held_locks[lock_id]
                    return False

        except PostgresError as e:
            logger.error(f"Database error during lock extension: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during lock extension: {e}")
            return False

    async def get_lock_status(self, lock_id: str) -> Optional[LockInfo]:
        """
        Get the current status of a lock.

        Args:
            lock_id: Unique identifier for the lock

        Returns:
            LockInfo if lock exists, None if lock doesn't exist
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT lock_id, owner_id, acquired_at, expires_at, lease_duration, metadata
                    FROM distributed_locks
                    WHERE lock_id = $1
                """,
                    lock_id,
                )

                if row:
                    return LockInfo(
                        lock_id=row["lock_id"],
                        owner_id=row["owner_id"],
                        acquired_at=float(row["acquired_at"]),
                        expires_at=float(row["expires_at"]),
                        lease_duration=float(row["lease_duration"]),
                        metadata=dict(row["metadata"]),
                    )
                return None

        except PostgresError as e:
            logger.error(f"Database error during lock status check: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during lock status check: {e}")
            return None

    async def list_held_locks(self) -> dict[str, LockInfo]:
        """
        Get all locks currently held by this instance.

        Returns:
            Dictionary of lock_id -> LockInfo for all held locks
        """
        return self._held_locks.copy()

    async def _maintain_lock_heartbeat(self, lock_id: str) -> None:
        """Maintain heartbeat for a long-lived lock."""
        try:
            while not self._shutdown_event.is_set() and lock_id in self._held_locks:
                # Update heartbeat every 1/3 of lease duration or every 5 seconds, whichever is shorter
                lock_info = self._held_locks[lock_id]
                heartbeat_interval = min(lock_info.lease_duration / 3, 5.0)

                await asyncio.sleep(heartbeat_interval)

                if lock_id not in self._held_locks:
                    break

                try:
                    async with self.pool.acquire() as conn:
                        current_time = int(time.time())
                        await conn.execute(
                            """
                            UPDATE distributed_locks
                            SET heartbeat_at = $1
                            WHERE lock_id = $2 AND owner_id = $3
                        """,
                            current_time,
                            lock_id,
                            self.instance_id,
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to update heartbeat for lock {lock_id}: {e}"
                    )

        except asyncio.CancelledError:
            logger.debug(f"Heartbeat task for lock {lock_id} cancelled")
        except Exception as e:
            logger.error(f"Error in heartbeat task for lock {lock_id}: {e}")

    async def _cleanup_expired_locks(self) -> None:
        """Periodically clean up expired locks from the database."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.cleanup_interval)

                    if self._shutdown_event.is_set():
                        break

                    async with self.pool.acquire() as conn:
                        current_time = int(time.time())

                        # Delete expired locks
                        result = await conn.execute(
                            """
                            DELETE FROM distributed_locks
                            WHERE expires_at < $1
                        """,
                            current_time,
                        )

                        if result.startswith("DELETE") and int(result.split()[1]) > 0:
                            count = int(result.split()[1])
                            logger.info(f"Cleaned up {count} expired distributed locks")

                        # Also clean up any locks we think we hold but have actually expired
                        expired_local_locks = [
                            lock_id
                            for lock_id, lock_info in self._held_locks.items()
                            if lock_info.is_expired
                        ]

                        for lock_id in expired_local_locks:
                            logger.warning(
                                f"Local lock {lock_id} has expired, removing from tracking"
                            )
                            if lock_id in self._heartbeat_tasks:
                                self._heartbeat_tasks[lock_id].cancel()
                                del self._heartbeat_tasks[lock_id]
                            del self._held_locks[lock_id]

                except Exception as e:
                    logger.error(f"Error during lock cleanup: {e}")

        except asyncio.CancelledError:
            logger.debug("Lock cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in lock cleanup task: {e}")

    async def shutdown(self) -> None:
        """Shutdown the lock manager and release all held locks."""
        logger.info("Shutting down distributed lock manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all heartbeat tasks
        for task in self._heartbeat_tasks.values():
            task.cancel()

        # Wait for heartbeat tasks to complete
        if self._heartbeat_tasks:
            await asyncio.gather(
                *self._heartbeat_tasks.values(), return_exceptions=True
            )
        self._heartbeat_tasks.clear()

        # Release all held locks
        lock_ids = list(self._held_locks.keys())
        for lock_id in lock_ids:
            try:
                await self.release_lock(lock_id)
            except Exception as e:
                logger.error(f"Error releasing lock {lock_id} during shutdown: {e}")

        logger.info(
            f"Distributed lock manager shutdown complete for instance {self.instance_id}"
        )
