# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mock Registration Storage Handler.

This module provides an in-memory mock implementation of the registration
storage handler protocol for testing purposes.

Thread Safety:
    This handler uses asyncio.Lock for coroutine-safe access to the
    in-memory record store.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.handlers.registration_storage.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)

logger = logging.getLogger(__name__)


class MockRegistrationStorageHandler:
    """In-memory mock for testing registration storage.

    Provides a simple in-memory implementation of the registration storage
    protocol for unit and integration testing.

    Thread Safety:
        This handler is coroutine-safe. All operations on the internal
        record store are protected by asyncio.Lock.

    Attributes:
        handler_type: Returns "mock" identifier.

    Example:
        >>> handler = MockRegistrationStorageHandler()
        >>> result = await handler.store_registration(record)
        >>> assert result.success
    """

    def __init__(self) -> None:
        """Initialize MockRegistrationStorageHandler with empty record store."""
        self._records: dict[UUID, ModelRegistrationRecord] = {}
        self._lock = asyncio.Lock()

        logger.debug("MockRegistrationStorageHandler initialized")

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            "mock" identifier string.
        """
        return "mock"

    async def store_registration(
        self,
        record: ModelRegistrationRecord,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Store a registration record in the mock store.

        Args:
            record: Registration record to store.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelUpsertResult with upsert outcome.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        async with self._lock:
            was_insert = record.node_id not in self._records
            now = datetime.now(UTC)

            # Update timestamps
            updated_record = ModelRegistrationRecord(
                node_id=record.node_id,
                node_type=record.node_type,
                node_version=record.node_version,
                endpoints=record.endpoints,
                metadata=record.metadata,
                created_at=record.created_at or now,
                updated_at=now,
                correlation_id=correlation_id,
            )

            self._records[record.node_id] = updated_record

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Mock registration stored",
            extra={
                "node_id": str(record.node_id),
                "was_insert": was_insert,
                "correlation_id": str(correlation_id),
            },
        )

        return ModelUpsertResult(
            success=True,
            node_id=record.node_id,
            operation="insert" if was_insert else "update",
            duration_ms=duration_ms,
            backend_type=self.handler_type,
            correlation_id=correlation_id,
        )

    async def query_registrations(
        self,
        node_type: EnumNodeKind | None = None,
        node_version: str | None = None,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> ModelStorageResult:
        """Query registration records from the mock store.

        Args:
            node_type: Optional node type to filter by.
            node_version: Optional version pattern to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelStorageResult with list of matching records.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        async with self._lock:
            matching_records: list[ModelRegistrationRecord] = []

            for record in self._records.values():
                # Filter by node type
                if node_type is not None and record.node_type != node_type:
                    continue

                # Filter by version prefix
                if node_version is not None and not record.node_version.startswith(
                    node_version
                ):
                    continue

                matching_records.append(record)

            # Sort by updated_at descending
            matching_records.sort(
                key=lambda r: r.updated_at or datetime.min.replace(tzinfo=UTC),
                reverse=True,
            )

            total_count = len(matching_records)

            # Apply pagination
            paginated = matching_records[offset : offset + limit]

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Mock registration query completed",
            extra={
                "record_count": len(paginated),
                "total_count": total_count,
                "correlation_id": str(correlation_id),
            },
        )

        return ModelStorageResult(
            success=True,
            records=tuple(paginated),
            total_count=total_count,
            duration_ms=duration_ms,
            backend_type=self.handler_type,
            correlation_id=correlation_id,
        )

    async def update_registration(
        self,
        node_id: UUID,
        endpoints: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Update an existing registration record in the mock store.

        Args:
            node_id: ID of the node to update.
            endpoints: Optional new endpoints dict.
            metadata: Optional new metadata dict.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelUpsertResult with update outcome.
        """
        correlation_id = correlation_id or uuid4()
        start_time = time.monotonic()

        async with self._lock:
            if node_id not in self._records:
                duration_ms = (time.monotonic() - start_time) * 1000
                return ModelUpsertResult(
                    success=False,
                    node_id=node_id,
                    operation="update",
                    error="Record not found",
                    duration_ms=duration_ms,
                    backend_type=self.handler_type,
                    correlation_id=correlation_id,
                )

            existing = self._records[node_id]
            now = datetime.now(UTC)

            # Update with new values
            updated_record = ModelRegistrationRecord(
                node_id=existing.node_id,
                node_type=existing.node_type,
                node_version=existing.node_version,
                endpoints=endpoints if endpoints is not None else existing.endpoints,
                metadata=metadata if metadata is not None else existing.metadata,
                created_at=existing.created_at,
                updated_at=now,
                correlation_id=correlation_id,
            )

            self._records[node_id] = updated_record

        duration_ms = (time.monotonic() - start_time) * 1000

        logger.debug(
            "Mock registration updated",
            extra={
                "node_id": str(node_id),
                "correlation_id": str(correlation_id),
            },
        )

        return ModelUpsertResult(
            success=True,
            node_id=node_id,
            operation="update",
            duration_ms=duration_ms,
            backend_type=self.handler_type,
            correlation_id=correlation_id,
        )

    async def delete_registration(
        self,
        node_id: UUID,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Delete a registration record from the mock store.

        Args:
            node_id: ID of the node to delete.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            True if record was deleted, False if not found.
        """
        correlation_id = correlation_id or uuid4()

        async with self._lock:
            if node_id in self._records:
                del self._records[node_id]
                deleted = True
            else:
                deleted = False

        logger.debug(
            "Mock registration deletion completed",
            extra={
                "node_id": str(node_id),
                "deleted": deleted,
                "correlation_id": str(correlation_id),
            },
        )

        return deleted

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> dict[str, object]:
        """Perform a health check on the mock handler.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            Dict with health status information.
        """
        correlation_id = correlation_id or uuid4()

        async with self._lock:
            record_count = len(self._records)

        return {
            "healthy": True,
            "backend_type": self.handler_type,
            "record_count": record_count,
            "correlation_id": str(correlation_id),
        }

    async def clear(self) -> None:
        """Clear all records from the mock store.

        Utility method for test cleanup.
        """
        async with self._lock:
            self._records.clear()

        logger.debug("Mock registration store cleared")

    async def get_record_count(self) -> int:
        """Get the number of stored records.

        Returns:
            Number of records in the store.
        """
        async with self._lock:
            return len(self._records)

    async def get_record(self, node_id: UUID) -> ModelRegistrationRecord | None:
        """Get a specific record by node ID.

        Args:
            node_id: ID of the node to retrieve.

        Returns:
            The record if found, None otherwise.
        """
        async with self._lock:
            return self._records.get(node_id)


__all__ = ["MockRegistrationStorageHandler"]
