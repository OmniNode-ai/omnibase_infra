# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Registration Persistence Operations.

This module defines the protocol that registration storage handlers must implement
to be used with capability-oriented nodes.

Protocol Duplication Note:
    This protocol is intentionally separate from the node-level protocol
    at ``nodes/node_registration_storage_effect/protocols/protocol_registration_persistence.py``:

    - **This protocol (handlers/*/protocol_*.py)**: Handler implementation
      contract used by tests for compliance verification.

    - **Node protocol (nodes/*/protocols/)**: Node-level contract used for
      container-based dependency injection and registry binding.

    The separation allows different import paths and independent evolution
    of handler contracts vs node-level DI contracts.

Concurrency Safety:
    Implementations MUST be safe for concurrent async calls.
    Multiple coroutines may invoke methods simultaneously.
    Implementations should use asyncio.Lock for coroutine-safety
    when protecting shared state.

Related:
    - NodeRegistrationStorageEffect: Effect node that uses this protocol
    - HandlerRegistrationStoragePostgres: PostgreSQL implementation
    - HandlerRegistrationStorageMock: In-memory mock for testing
    - nodes/node_registration_storage_effect/protocols/: Node-level protocol
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.handlers.registration_storage.models import (
    ModelDeleteRegistrationRequest,
    ModelUpdateRegistrationRequest,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelDeleteResult,
    ModelRegistrationRecord,
    ModelStorageHealthCheckResult,
    ModelStorageQuery,
    ModelStorageResult,
    ModelUpsertResult,
)


@runtime_checkable
class ProtocolRegistrationPersistence(Protocol):
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
        request: ModelUpdateRegistrationRequest,
    ) -> ModelUpsertResult:
        """Update an existing registration record.

        Args:
            request: ModelUpdateRegistrationRequest containing:
                - node_id: ID of the node to update
                - updates: ModelRegistrationUpdate with fields to update
                  (only non-None fields will be applied)
                - correlation_id: Optional correlation ID for tracing

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
        request: ModelDeleteRegistrationRequest,
    ) -> ModelDeleteResult:
        """Delete a registration record from storage.

        Args:
            request: ModelDeleteRegistrationRequest containing:
                - node_id: ID of the node to delete
                - correlation_id: Optional correlation ID for tracing

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


__all__ = ["ProtocolRegistrationPersistence"]
