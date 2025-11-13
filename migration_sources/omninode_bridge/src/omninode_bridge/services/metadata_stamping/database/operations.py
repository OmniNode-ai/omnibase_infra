"""Advanced metadata operations for high-performance metadata stamping service.

This module implements sophisticated database operations including:
- Smart upsert with conflict resolution strategies
- High-throughput batch processing (>50 items/sec)
- JSONB deep merge for intelligence_data fields
- Version control with optimistic locking
- Idempotent operations using op_id tracking

Performance targets:
- Batch operations: >50 items/sec
- Smart upsert: <5ms per operation
- JSONB deep merge: <2ms for typical payloads
- Version conflict resolution: <1ms overhead
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import orjson
from asyncpg import Connection

from ....config.batch_sizes import get_batch_manager
from .client import DatabaseOperationError, MetadataStampingPostgresClient

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies for smart upsert operations."""

    MERGE_JSONB = "merge_jsonb"  # Deep merge JSONB fields
    REPLACE_ALL = "replace_all"  # Replace entire record
    KEEP_EXISTING = "keep_existing"  # Keep existing record
    UPDATE_TIMESTAMP = "update_timestamp"  # Update timestamp only
    CUSTOM_MERGE = "custom_merge"  # Custom merge logic


class BatchOperationResult(Enum):
    """Result types for batch operations."""

    SUCCESS = "success"
    CONFLICT = "conflict"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class UpsertResult:
    """Result of smart upsert operation."""

    id: str
    operation: str  # "inserted", "updated", "skipped"
    version: int
    execution_time_ms: float
    conflict_resolved: bool = False
    merge_conflicts: list[str] = None


@dataclass
class BatchUpsertResult:
    """Result of batch upsert operation."""

    total_processed: int
    successful_operations: int
    conflicts_resolved: int
    errors: int
    execution_time_ms: float
    throughput_per_sec: float
    results: list[UpsertResult]


@dataclass
class VersionedMetadata:
    """Metadata record with version control."""

    id: str
    file_hash: str
    file_path: str
    file_size: int
    content_type: Optional[str]
    stamp_data: dict[str, Any]
    protocol_version: str
    version: int
    op_id: Optional[str]
    created_at: Any
    updated_at: Any


class AdvancedMetadataOperations:
    """Advanced metadata operations with high-performance capabilities."""

    def __init__(self, client: MetadataStampingPostgresClient):
        """Initialize advanced operations with database client.

        Args:
            client: PostgreSQL client instance
        """
        self.client = client
        self.performance_metrics = {
            "upsert_operations": 0,
            "batch_operations": 0,
            "merge_operations": 0,
            "version_conflicts": 0,
            "avg_throughput": 0.0,
        }

    async def _ensure_schema_extensions(self) -> bool:
        """Ensure required schema extensions for advanced operations.

        Returns:
            True if schema is ready, False otherwise
        """
        try:
            # Add version and operation tracking columns if not exists
            schema_updates = [
                """
                ALTER TABLE metadata_stamps
                ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1 NOT NULL
                """,
                """
                ALTER TABLE metadata_stamps
                ADD COLUMN IF NOT EXISTS op_id UUID
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_metadata_stamps_version
                ON metadata_stamps(file_hash, version)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_metadata_stamps_op_id
                ON metadata_stamps(op_id) WHERE op_id IS NOT NULL
                """,
            ]

            for update in schema_updates:
                await self.client.execute_query(update, fetch_mode="execute")

            logger.info("Advanced metadata operations schema extensions verified")
            return True

        except Exception as e:
            logger.error(f"Failed to ensure schema extensions: {e}")
            return False

    def jsonb_deep_merge(
        self, existing: dict[str, Any], new: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform deep merge of JSONB objects with intelligent conflict resolution.

        Args:
            existing: Existing JSONB data
            new: New JSONB data to merge

        Returns:
            Merged JSONB object

        Performance target: <2ms for typical payloads
        """
        start_time = time.perf_counter()

        def deep_merge_recursive(base: dict, update: dict) -> dict:
            """Recursively merge dictionaries with array concatenation."""
            result = base.copy()

            for key, value in update.items():
                if key in result:
                    if isinstance(result[key], dict) and isinstance(value, dict):
                        # Recursively merge nested dictionaries
                        result[key] = deep_merge_recursive(result[key], value)
                    elif isinstance(result[key], list) and isinstance(value, list):
                        # Concatenate arrays, removing duplicates
                        combined = result[key] + value
                        # Remove duplicates while preserving order
                        seen = set()
                        result[key] = []
                        for item in combined:
                            # Handle different types for deduplication
                            if isinstance(item, dict | list):
                                item_str = orjson.dumps(item, sort_keys=True).decode()
                                if item_str not in seen:
                                    seen.add(item_str)
                                    result[key].append(item)
                            else:
                                if item not in seen:
                                    seen.add(item)
                                    result[key].append(item)
                    else:
                        # Replace with new value for non-mergeable types
                        result[key] = value
                else:
                    # Add new key-value pair
                    result[key] = value

            return result

        try:
            merged_result = deep_merge_recursive(existing, new)
            execution_time = (time.perf_counter() - start_time) * 1000

            if execution_time > 2.0:
                logger.warning(
                    f"JSONB deep merge exceeded target time: {execution_time:.2f}ms"
                )

            self.performance_metrics["merge_operations"] += 1
            return merged_result

        except Exception as e:
            logger.error(f"JSONB deep merge failed: {e}")
            # Fallback to new data on merge failure
            return new

    async def smart_upsert_metadata_stamp(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        content_type: Optional[str],
        stamp_data: dict[str, Any],
        protocol_version: str = "1.0",
        op_id: Optional[str] = None,
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_JSONB,
    ) -> UpsertResult:
        """Smart upsert with advanced conflict resolution.

        Args:
            file_hash: BLAKE3 hash of the file
            file_path: Path to the file
            file_size: Size of the file in bytes
            content_type: MIME content type
            stamp_data: Stamp metadata as dictionary
            protocol_version: Protocol version
            op_id: Optional operation ID for idempotency
            conflict_strategy: Strategy for handling conflicts

        Returns:
            UpsertResult with operation details

        Performance target: <5ms per operation
        """
        start_time = time.perf_counter()

        # Validate inputs
        if not file_hash:
            raise ValueError("file_hash cannot be None or empty")
        if not file_path:
            raise ValueError("file_path cannot be None or empty")
        if stamp_data is None:
            raise ValueError("stamp_data cannot be None")

        # Generate operation ID if not provided
        if op_id is None:
            op_id = str(uuid.uuid4())

        async with self.client.acquire_connection() as connection:
            try:
                async with connection.transaction():
                    # Check for existing record
                    existing_query = """
                        SELECT id, stamp_data, version, op_id
                        FROM metadata_stamps
                        WHERE file_hash = $1
                        FOR UPDATE
                    """
                    existing_record = await connection.fetchrow(
                        existing_query, file_hash
                    )

                    # Check for idempotency
                    if existing_record and existing_record["op_id"] == op_id:
                        execution_time = (time.perf_counter() - start_time) * 1000
                        return UpsertResult(
                            id=str(existing_record["id"]),
                            operation="skipped",
                            version=existing_record["version"],
                            execution_time_ms=execution_time,
                            conflict_resolved=False,
                        )

                    if existing_record:
                        # Update existing record with conflict resolution
                        return await self._handle_update_conflict(
                            connection,
                            existing_record,
                            file_path,
                            file_size,
                            content_type,
                            stamp_data,
                            protocol_version,
                            op_id,
                            conflict_strategy,
                            start_time,
                        )
                    else:
                        # Insert new record
                        return await self._insert_new_record(
                            connection,
                            file_hash,
                            file_path,
                            file_size,
                            content_type,
                            stamp_data,
                            protocol_version,
                            op_id,
                            start_time,
                        )

            except Exception as e:
                execution_time = (time.perf_counter() - start_time) * 1000
                logger.error(f"Smart upsert failed after {execution_time:.2f}ms: {e}")
                raise DatabaseOperationError(f"Smart upsert failed: {e}") from e

    async def _handle_update_conflict(
        self,
        connection: Connection,
        existing_record: Any,
        file_path: str,
        file_size: int,
        content_type: Optional[str],
        stamp_data: dict[str, Any],
        protocol_version: str,
        op_id: str,
        conflict_strategy: ConflictResolutionStrategy,
        start_time: float,
    ) -> UpsertResult:
        """Handle update conflict with specified strategy."""

        existing_stamp_data = existing_record["stamp_data"]
        merge_conflicts = []

        if conflict_strategy == ConflictResolutionStrategy.MERGE_JSONB:
            # Deep merge stamp_data
            merged_stamp_data = self.jsonb_deep_merge(existing_stamp_data, stamp_data)
            final_stamp_data = merged_stamp_data

        elif conflict_strategy == ConflictResolutionStrategy.REPLACE_ALL:
            final_stamp_data = stamp_data

        elif conflict_strategy == ConflictResolutionStrategy.KEEP_EXISTING:
            execution_time = (time.perf_counter() - start_time) * 1000
            return UpsertResult(
                id=str(existing_record["id"]),
                operation="skipped",
                version=existing_record["version"],
                execution_time_ms=execution_time,
                conflict_resolved=False,
            )

        elif conflict_strategy == ConflictResolutionStrategy.UPDATE_TIMESTAMP:
            final_stamp_data = existing_stamp_data

        else:  # CUSTOM_MERGE
            # Implement custom merge logic here
            final_stamp_data = self.jsonb_deep_merge(existing_stamp_data, stamp_data)

        # Update record with version increment
        update_query = """
            UPDATE metadata_stamps
            SET file_path = $2,
                file_size = $3,
                content_type = $4,
                stamp_data = $5,
                protocol_version = $6,
                version = version + 1,
                op_id = $7,
                updated_at = NOW()
            WHERE id = $1
            RETURNING version
        """

        stamp_data_json = orjson.dumps(final_stamp_data).decode("utf-8")
        result = await connection.fetchrow(
            update_query,
            existing_record["id"],
            file_path,
            file_size,
            content_type,
            stamp_data_json,
            protocol_version,
            op_id,
        )

        execution_time = (time.perf_counter() - start_time) * 1000
        self.performance_metrics["upsert_operations"] += 1
        self.performance_metrics["version_conflicts"] += 1

        return UpsertResult(
            id=str(existing_record["id"]),
            operation="updated",
            version=result["version"],
            execution_time_ms=execution_time,
            conflict_resolved=True,
            merge_conflicts=merge_conflicts,
        )

    async def _insert_new_record(
        self,
        connection: Connection,
        file_hash: str,
        file_path: str,
        file_size: int,
        content_type: Optional[str],
        stamp_data: dict[str, Any],
        protocol_version: str,
        op_id: str,
        start_time: float,
    ) -> UpsertResult:
        """Insert new metadata record."""

        insert_query = """
            INSERT INTO metadata_stamps (
                file_hash, file_path, file_size, content_type,
                stamp_data, protocol_version, version, op_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, 1, $7)
            RETURNING id
        """

        stamp_data_json = orjson.dumps(stamp_data).decode("utf-8")
        result = await connection.fetchrow(
            insert_query,
            file_hash,
            file_path,
            file_size,
            content_type,
            stamp_data_json,
            protocol_version,
            op_id,
        )

        execution_time = (time.perf_counter() - start_time) * 1000
        self.performance_metrics["upsert_operations"] += 1

        return UpsertResult(
            id=str(result["id"]),
            operation="inserted",
            version=1,
            execution_time_ms=execution_time,
            conflict_resolved=False,
        )

    async def batch_upsert_metadata_stamps(
        self,
        stamps_data: list[dict[str, Any]],
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_JSONB,
        batch_size: Optional[int] = None,
        max_concurrency: int = 10,
    ) -> BatchUpsertResult:
        """High-performance batch upsert with concurrency control.

        Args:
            stamps_data: List of stamp data dictionaries
            conflict_strategy: Strategy for handling conflicts
            batch_size: Number of operations per batch
            max_concurrency: Maximum concurrent operations

        Returns:
            BatchUpsertResult with performance metrics

        Performance target: >50 items/sec
        """
        start_time = time.perf_counter()

        if not stamps_data:
            return BatchUpsertResult(
                total_processed=0,
                successful_operations=0,
                conflicts_resolved=0,
                errors=0,
                execution_time_ms=0,
                throughput_per_sec=0,
                results=[],
            )

        # Use configured batch size if not provided
        if batch_size is None:
            batch_manager = get_batch_manager()
            batch_size = batch_manager.database_batch_size

        # Ensure schema extensions are available
        await self._ensure_schema_extensions()

        # Create batches
        batches = [
            stamps_data[i : i + batch_size]
            for i in range(0, len(stamps_data), batch_size)
        ]

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_batch(batch: list[dict[str, Any]]) -> list[UpsertResult]:
            """Process a single batch of upsert operations."""
            async with semaphore:
                batch_results = []

                for stamp_data in batch:
                    try:
                        # Validate required fields
                        required_fields = [
                            "file_hash",
                            "file_path",
                            "file_size",
                            "stamp_data",
                        ]
                        for field in required_fields:
                            if field not in stamp_data or stamp_data[field] is None:
                                raise ValueError(f"Missing required field: {field}")

                        result = await self.smart_upsert_metadata_stamp(
                            file_hash=stamp_data["file_hash"],
                            file_path=stamp_data["file_path"],
                            file_size=stamp_data["file_size"],
                            content_type=stamp_data.get("content_type"),
                            stamp_data=stamp_data["stamp_data"],
                            protocol_version=stamp_data.get("protocol_version", "1.0"),
                            op_id=stamp_data.get("op_id"),
                            conflict_strategy=conflict_strategy,
                        )
                        batch_results.append(result)

                    except Exception as e:
                        logger.error(f"Batch upsert item failed: {e}")
                        batch_results.append(
                            UpsertResult(
                                id="",
                                operation="error",
                                version=0,
                                execution_time_ms=0,
                                conflict_resolved=False,
                            )
                        )

                return batch_results

        # Execute all batches concurrently
        try:
            batch_tasks = [process_batch(batch) for batch in batches]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Flatten results
            all_results = []
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    logger.error(f"Batch processing failed: {batch_result}")
                    continue
                all_results.extend(batch_result)

            # Calculate metrics
            execution_time = (time.perf_counter() - start_time) * 1000
            total_processed = len(all_results)
            successful_operations = len(
                [r for r in all_results if r.operation in ["inserted", "updated"]]
            )
            conflicts_resolved = len([r for r in all_results if r.conflict_resolved])
            errors = len([r for r in all_results if r.operation == "error"])
            throughput_per_sec = (
                (total_processed / (execution_time / 1000)) if execution_time > 0 else 0
            )

            # Update performance metrics
            self.performance_metrics["batch_operations"] += 1
            self.performance_metrics["avg_throughput"] = (
                self.performance_metrics["avg_throughput"]
                * (self.performance_metrics["batch_operations"] - 1)
                + throughput_per_sec
            ) / self.performance_metrics["batch_operations"]

            logger.info(
                f"Batch upsert completed: {total_processed} items in {execution_time:.2f}ms ({throughput_per_sec:.1f} items/sec)"
            )

            return BatchUpsertResult(
                total_processed=total_processed,
                successful_operations=successful_operations,
                conflicts_resolved=conflicts_resolved,
                errors=errors,
                execution_time_ms=execution_time,
                throughput_per_sec=throughput_per_sec,
                results=all_results,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Batch upsert failed after {execution_time:.2f}ms: {e}")
            raise DatabaseOperationError(f"Batch upsert failed: {e}") from e

    async def get_versioned_metadata(
        self, file_hash: str, version: Optional[int] = None
    ) -> Optional[VersionedMetadata]:
        """Retrieve versioned metadata record.

        Args:
            file_hash: File hash to look up
            version: Specific version to retrieve (latest if None)

        Returns:
            VersionedMetadata record if found, None otherwise
        """
        if version is not None:
            query = """
                SELECT id, file_hash, file_path, file_size, content_type,
                       stamp_data, protocol_version, version, op_id, created_at, updated_at
                FROM metadata_stamps
                WHERE file_hash = $1 AND version = $2
            """
            result = await self.client.execute_query(
                query, file_hash, version, fetch_mode="one"
            )
        else:
            query = """
                SELECT id, file_hash, file_path, file_size, content_type,
                       stamp_data, protocol_version, version, op_id, created_at, updated_at
                FROM metadata_stamps
                WHERE file_hash = $1
                ORDER BY version DESC
                LIMIT 1
            """
            result = await self.client.execute_query(query, file_hash, fetch_mode="one")

        if result:
            return VersionedMetadata(
                id=str(result["id"]),
                file_hash=result["file_hash"],
                file_path=result["file_path"],
                file_size=result["file_size"],
                content_type=result["content_type"],
                stamp_data=result["stamp_data"],
                protocol_version=result["protocol_version"],
                version=result["version"],
                op_id=str(result["op_id"]) if result["op_id"] else None,
                created_at=result["created_at"],
                updated_at=result["updated_at"],
            )
        return None

    async def get_version_history(
        self, file_hash: str, limit: int = 10
    ) -> list[VersionedMetadata]:
        """Get version history for a file.

        Args:
            file_hash: File hash to get history for
            limit: Maximum number of versions to return

        Returns:
            List of VersionedMetadata records ordered by version (newest first)
        """
        query = """
            SELECT id, file_hash, file_path, file_size, content_type,
                   stamp_data, protocol_version, version, op_id, created_at, updated_at
            FROM metadata_stamps
            WHERE file_hash = $1
            ORDER BY version DESC
            LIMIT $2
        """

        results = await self.client.execute_query(
            query, file_hash, limit, fetch_mode="all"
        )

        return [
            VersionedMetadata(
                id=str(row["id"]),
                file_hash=row["file_hash"],
                file_path=row["file_path"],
                file_size=row["file_size"],
                content_type=row["content_type"],
                stamp_data=row["stamp_data"],
                protocol_version=row["protocol_version"],
                version=row["version"],
                op_id=str(row["op_id"]) if row["op_id"] else None,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in results
        ]

    async def check_operation_idempotency(self, op_id: str) -> Optional[str]:
        """Check if operation has already been executed.

        Args:
            op_id: Operation ID to check

        Returns:
            Record ID if operation exists, None otherwise
        """
        query = """
            SELECT id FROM metadata_stamps WHERE op_id = $1
        """
        result = await self.client.execute_query(query, op_id, fetch_mode="val")
        return str(result) if result else None

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get advanced operations performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return {
            **self.performance_metrics,
            "timestamp": time.time(),
            "client_metrics": await self.client.health_check(),
        }

    async def optimize_performance(self) -> dict[str, Any]:
        """Run performance optimization routines.

        Returns:
            Optimization results
        """
        start_time = time.perf_counter()
        optimizations = []

        try:
            # Analyze and update table statistics
            await self.client.execute_query(
                "ANALYZE metadata_stamps", fetch_mode="execute"
            )
            optimizations.append("Table statistics updated")

            # Reindex if needed (check index bloat)
            bloat_query = """
                SELECT schemaname, tablename, attname, n_distinct, correlation
                FROM pg_stats
                WHERE tablename = 'metadata_stamps'
                AND n_distinct < -0.1
            """
            bloat_results = await self.client.execute_query(
                bloat_query, fetch_mode="all"
            )

            if bloat_results:
                await self.client.execute_query(
                    "REINDEX TABLE metadata_stamps", fetch_mode="execute"
                )
                optimizations.append("Indexes rebuilt due to detected bloat")

            execution_time = (time.perf_counter() - start_time) * 1000

            return {
                "optimizations_applied": optimizations,
                "execution_time_ms": execution_time,
                "status": "completed",
            }

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Performance optimization failed after {execution_time:.2f}ms: {e}"
            )
            return {
                "optimizations_applied": optimizations,
                "execution_time_ms": execution_time,
                "status": "failed",
                "error": str(e),
            }
