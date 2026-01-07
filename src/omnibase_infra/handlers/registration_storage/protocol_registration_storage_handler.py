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

from omnibase_infra.handlers.registration_storage.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelDeleteResult,
    ModelRegistrationUpdate,
    ModelStorageHealthCheckResult,
    ModelStorageQuery,
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
        query: ModelStorageQuery,
        correlation_id: UUID | None = None,
    ) -> ModelStorageResult:
        """Query registration records from storage.

        Args:
            query: ModelStorageQuery containing filter and pagination parameters.
                Supports filtering by node_id, node_type, capability_filter,
                and pagination via limit/offset.
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
        updates: ModelRegistrationUpdate,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Update an existing registration record.

        Args:
            node_id: ID of the node to update.
            updates: ModelRegistrationUpdate containing fields to update.
                Only non-None fields will be applied.
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
