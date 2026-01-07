# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Registration Storage Handler.

This module defines the protocol that registration storage handlers must implement
to be used with capability-oriented nodes.

Concurrency Safety:
    Implementations MUST be safe for concurrent async calls.
    Multiple coroutines may invoke methods simultaneously.
    Implementations should use asyncio.Lock for coroutine-safety
    when protecting shared state.

Related:
    - NodeRegistrationStorageEffect: Effect node that uses this protocol
    - PostgresRegistrationStorageHandler: PostgreSQL implementation
    - MockRegistrationStorageHandler: In-memory mock for testing
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_core.enums.enum_node_kind import EnumNodeKind

from omnibase_infra.handlers.registration_storage.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelDeleteResult,
    ModelStorageHealthCheckResult,
)


@runtime_checkable
class ProtocolRegistrationStorageHandler(Protocol):
    """Protocol for registration storage handler implementations.

    Defines the interface that all registration storage handlers must implement.
    Handlers are responsible for storing, querying, updating, and deleting
    registration records.

    Concurrency Safety:
        Implementations MUST be safe for concurrent async coroutine calls.

        **Guarantees implementers MUST provide:**
            - Concurrent method calls are coroutine-safe
            - Connection pooling (if used) is async-safe
            - Database transactions are properly isolated
            - Internal state (if any) is protected by asyncio.Lock

        **What callers can assume:**
            - Multiple coroutines can call methods concurrently
            - Each operation is independent
            - Failures in one operation do not affect others

        Note: asyncio.Lock provides coroutine-safety, not thread-safety.
    """

    @property
    def handler_type(self) -> str:
        """Return the handler type identifier.

        Returns:
            Handler type string (e.g., "postgresql", "mock").
        """
        ...

    async def store_registration(
        self,
        record: ModelRegistrationRecord,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Store a registration record in the storage backend.

        Args:
            record: Registration record to store.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelUpsertResult with success status and operation metadata.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def query_registrations(
        self,
        node_type: EnumNodeKind | None = None,
        node_version: str | None = None,
        limit: int = 100,
        offset: int = 0,
        correlation_id: UUID | None = None,
    ) -> ModelStorageResult:
        """Query registration records from storage.

        Args:
            node_type: Optional node type to filter by.
            node_version: Optional version pattern to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelStorageResult with list of matching records
            and operation metadata.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def update_registration(
        self,
        node_id: UUID,
        endpoints: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Update an existing registration record.

        Args:
            node_id: ID of the node to update.
            endpoints: Optional new endpoints dict.
            metadata: Optional new metadata dict.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelUpsertResult with success status and operation metadata.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def delete_registration(
        self,
        node_id: UUID,
        correlation_id: UUID | None = None,
    ) -> ModelDeleteResult:
        """Delete a registration record from storage.

        Args:
            node_id: ID of the node to delete.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelDeleteResult with deletion outcome.

        Raises:
            InfraConnectionError: If connection to backend fails.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelStorageHealthCheckResult:
        """Perform a health check on the handler.

        Args:
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelStorageHealthCheckResult with health status information.
        """
        ...


__all__ = ["ProtocolRegistrationStorageHandler"]
