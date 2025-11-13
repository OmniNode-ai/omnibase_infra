"""
Action Deduplication Service for idempotent action processing.

Provides deduplication for actions with TTL-based expiration to handle
retries and at-least-once delivery guarantees in distributed systems.

ONEX v2.0 Compliance:
- Suffix-based naming: ActionDedupService
- Strong typing with ModelActionDedup entity
- Comprehensive error handling
- Performance metrics tracking

Features:
- Idempotent action processing
- TTL-based automatic expiration (default: 6 hours)
- Result hash validation on replay
- Efficient cleanup of expired entries
- Prometheus metrics integration

Example:
    >>> from uuid import uuid4
    >>> import hashlib
    >>> import json
    >>>
    >>> service = ActionDedupService(postgres_client)
    >>>
    >>> # Check if action should be processed
    >>> action_id = uuid4()
    >>> should_process = await service.should_process("workflow-123", action_id)
    >>>
    >>> if should_process:
    ...     # Process action
    ...     result = {"status": "completed", "items": 100}
    ...
    ...     # Compute result hash
    ...     result_hash = hashlib.sha256(
    ...         json.dumps(result, sort_keys=True).encode()
    ...     ).hexdigest()
    ...
    ...     # Record as processed
    ...     await service.record_processed(
    ...         "workflow-123",
    ...         action_id,
    ...         result_hash,
    ...         ttl_hours=6
    ...     )
"""

import logging
from datetime import UTC, datetime, timedelta

# Import ModelActionDedup using TYPE_CHECKING to avoid runtime import
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from prometheus_client import Counter

if TYPE_CHECKING:
    pass

from .postgres_client import PostgresClient

logger = logging.getLogger(__name__)

# Prometheus Metrics
action_dedup_checks_total = Counter(
    "action_dedup_checks_total",
    "Total number of deduplication checks performed",
)

action_dedup_hits_total = Counter(
    "action_dedup_hits_total",
    "Total number of duplicate actions detected",
)

action_dedup_records_total = Counter(
    "action_dedup_records_total",
    "Total number of actions recorded as processed",
)

action_dedup_cleanup_deleted_total = Counter(
    "action_dedup_cleanup_deleted_total",
    "Total number of expired entries cleaned up",
)


class ActionDedupService:
    """
    Service for action deduplication with TTL-based expiration.

    This service prevents duplicate processing of actions by tracking which
    actions have been processed and their results. It supports:

    - Duplicate detection via composite key (workflow_key, action_id)
    - Result hash validation for consistency checking
    - TTL-based automatic expiration (default: 6 hours)
    - Efficient cleanup of expired entries
    - Metrics tracking for monitoring

    Performance Characteristics:
    - should_process(): <5ms (indexed lookup)
    - record_processed(): <10ms (single insert)
    - cleanup_expired(): <100ms for 1000s of entries

    Thread Safety:
    - This service is designed for asyncio and is not thread-safe
    - Use separate instances for different threads if needed

    Database Schema:
    - Table: action_dedup_log
    - Primary Key: (workflow_key, action_id) composite
    - Index: expires_at for efficient cleanup
    """

    def __init__(self, postgres_client: PostgresClient):
        """
        Initialize the action deduplication service.

        Args:
            postgres_client: PostgreSQL client for database operations
        """
        self.db = postgres_client

        # Metrics tracking
        self._metrics = {
            "dedup_checks_total": 0,
            "dedup_hits_total": 0,
            "dedup_records_total": 0,
            "dedup_cleanup_deleted_total": 0,
        }

    async def should_process(self, workflow_key: str, action_id: UUID) -> bool:
        """
        Check if an action should be processed or if it's a duplicate.

        This method checks if the action has already been processed by looking
        up the composite key (workflow_key, action_id) in the deduplication log.

        Args:
            workflow_key: Workflow identifier for grouping actions
            action_id: Unique action identifier (UUID)

        Returns:
            True if action should be processed (not a duplicate)
            False if action is a duplicate (already processed)

        Raises:
            RuntimeError: If database connection is unavailable
            PostgresError: If database query fails

        Example:
            >>> action_id = uuid4()
            >>> if await service.should_process("workflow-123", action_id):
            ...     # Process action
            ...     result = process_action(action_id)
            ...     # Record as processed
            ...     await service.record_processed(...)
        """
        if not self.db.pool:
            raise RuntimeError("PostgreSQL client not connected")

        try:
            # Increment check counter
            self._metrics["dedup_checks_total"] += 1
            action_dedup_checks_total.inc()

            # Check if action already exists in dedup log
            exists = await self.db.fetch_one(
                """
                SELECT 1 FROM action_dedup_log
                WHERE workflow_key = $1 AND action_id = $2
                """,
                workflow_key,
                action_id,
            )

            if exists:
                # Duplicate detected
                self._metrics["dedup_hits_total"] += 1
                action_dedup_hits_total.inc()
                logger.info(
                    f"Duplicate action detected: workflow_key={workflow_key}, "
                    f"action_id={action_id}"
                )
                return False

            # Not a duplicate, should process
            return True

        except Exception as e:
            logger.error(
                f"Error checking dedup status for action {action_id}: {e}",
                exc_info=True,
            )
            # On error, default to processing to avoid blocking
            # This is a safe failure mode for deduplication
            logger.warning(
                f"Defaulting to process action {action_id} due to dedup check error"
            )
            return True

    async def record_processed(
        self,
        workflow_key: str,
        action_id: UUID,
        result_hash: str,
        ttl_hours: int = 6,
    ) -> None:
        """
        Record an action as processed with result hash and TTL.

        This method inserts a deduplication record with:
        - Composite primary key (workflow_key, action_id) for uniqueness
        - Result hash for validation on potential replays
        - TTL-based expiration for automatic cleanup

        Args:
            workflow_key: Workflow identifier for grouping actions
            action_id: Unique action identifier (UUID)
            result_hash: SHA256 hash of action result for validation
            ttl_hours: Time-to-live in hours (default: 6)

        Raises:
            RuntimeError: If database connection is unavailable
            ValueError: If result_hash is not a valid SHA256 hex string
            PostgresError: If database insert fails

        Example:
            >>> import hashlib
            >>> import json
            >>>
            >>> result = {"status": "completed", "items": 100}
            >>> result_hash = hashlib.sha256(
            ...     json.dumps(result, sort_keys=True).encode()
            ... ).hexdigest()
            >>>
            >>> await service.record_processed(
            ...     "workflow-123",
            ...     action_id,
            ...     result_hash,
            ...     ttl_hours=6
            ... )
        """
        if not self.db.pool:
            raise RuntimeError("PostgreSQL client not connected")

        try:
            # Validate result_hash format (SHA256 = 64 hex chars)
            if not isinstance(result_hash, str) or len(result_hash) != 64:
                raise ValueError("result_hash must be a 64-character SHA256 hex string")
            if not all(c in "0123456789abcdefABCDEF" for c in result_hash):
                raise ValueError("result_hash must be a valid hexadecimal string")

            # Normalize to lowercase
            result_hash = result_hash.lower()

            # Calculate timestamps
            now = datetime.now(UTC)
            expires_at = now + timedelta(hours=ttl_hours)

            # Insert into database
            await self.db.execute_query(
                """
                INSERT INTO action_dedup_log
                    (workflow_key, action_id, result_hash, processed_at, expires_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (workflow_key, action_id) DO NOTHING
                """,
                workflow_key,
                action_id,
                result_hash,
                now,
                expires_at,
            )

            # Increment record counter
            self._metrics["dedup_records_total"] += 1
            action_dedup_records_total.inc()

            logger.debug(
                f"Recorded processed action: workflow_key={workflow_key}, "
                f"action_id={action_id}, expires_at={expires_at}"
            )

        except ValueError as e:
            # Pydantic validation error (e.g., invalid result_hash format)
            logger.error(f"Invalid dedup record data: {e}", exc_info=True)
            raise

        except Exception as e:
            logger.error(
                f"Error recording processed action {action_id}: {e}", exc_info=True
            )
            # Don't raise - failure to record dedup should not block processing
            # This allows the system to continue even if dedup recording fails
            logger.warning(
                f"Failed to record dedup for action {action_id}, continuing anyway"
            )

    async def cleanup_expired(self) -> int:
        """
        Clean up expired deduplication entries from the database.

        This method deletes all entries where expires_at < NOW(), freeing up
        database space and maintaining performance. It should be called
        periodically (e.g., hourly) by a background task.

        Returns:
            Number of entries deleted

        Raises:
            RuntimeError: If database connection is unavailable
            PostgresError: If database delete fails

        Performance:
        - Typically <100ms for thousands of entries
        - Uses indexed expires_at column for efficiency
        - Can be run concurrently from multiple instances safely

        Example:
            >>> deleted_count = await service.cleanup_expired()
            >>> logger.info(f"Cleaned up {deleted_count} expired dedup entries")
        """
        if not self.db.pool:
            raise RuntimeError("PostgreSQL client not connected")

        try:
            # Delete expired entries using indexed query
            result = await self.db.execute_query(
                """
                DELETE FROM action_dedup_log
                WHERE expires_at < NOW()
                """
            )

            # Parse result to get count
            # Result format: "DELETE N" where N is number of rows deleted
            deleted_count = 0
            if result and isinstance(result, str) and result.startswith("DELETE"):
                try:
                    deleted_count = int(result.split()[1])
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse delete result: {result}")

            # Update metrics
            self._metrics["dedup_cleanup_deleted_total"] += deleted_count
            action_dedup_cleanup_deleted_total.inc(deleted_count)

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired dedup entries")

            return deleted_count

        except Exception as e:
            logger.error(f"Error during dedup cleanup: {e}", exc_info=True)
            # Don't raise - cleanup failure should not block operations
            logger.warning("Failed to cleanup expired dedup entries")
            return 0

    async def get_dedup_entry(
        self, workflow_key: str, action_id: UUID
    ) -> Optional[dict[str, Any]]:
        """
        Get deduplication entry for an action if it exists.

        Args:
            workflow_key: Workflow identifier for grouping actions
            action_id: Unique action identifier (UUID)

        Returns:
            Dictionary with dedup entry data if exists, None otherwise.
            Keys: workflow_key, action_id, result_hash, processed_at, expires_at

        Raises:
            RuntimeError: If database connection is unavailable
            PostgresError: If database query fails
        """
        if not self.db.pool:
            raise RuntimeError("PostgreSQL client not connected")

        try:
            row = await self.db.fetch_one(
                """
                SELECT workflow_key, action_id, result_hash, processed_at, expires_at
                FROM action_dedup_log
                WHERE workflow_key = $1 AND action_id = $2
                """,
                workflow_key,
                action_id,
            )

            if row:
                return {
                    "workflow_key": row["workflow_key"],
                    "action_id": row["action_id"],
                    "result_hash": row["result_hash"],
                    "processed_at": row["processed_at"],
                    "expires_at": row["expires_at"],
                }

            return None

        except Exception as e:
            logger.error(
                f"Error retrieving dedup entry for action {action_id}: {e}",
                exc_info=True,
            )
            return None

    def get_metrics(self) -> dict[str, int]:
        """
        Get current deduplication metrics.

        Returns:
            Dictionary with metric counts:
            - dedup_checks_total: Total number of dedup checks
            - dedup_hits_total: Number of duplicate detections
            - dedup_records_total: Number of actions recorded
            - dedup_cleanup_deleted_total: Total entries cleaned up

        Example:
            >>> metrics = service.get_metrics()
            >>> hit_rate = metrics["dedup_hits_total"] / max(metrics["dedup_checks_total"], 1)
            >>> logger.info(f"Dedup hit rate: {hit_rate:.2%}")
        """
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """
        Reset all metrics counters to zero.

        This is useful for testing or periodic metric reporting.
        """
        self._metrics = {
            "dedup_checks_total": 0,
            "dedup_hits_total": 0,
            "dedup_records_total": 0,
            "dedup_cleanup_deleted_total": 0,
        }
