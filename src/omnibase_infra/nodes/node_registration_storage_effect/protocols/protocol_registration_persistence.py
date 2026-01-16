# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Registration Persistence Operations.

This module defines ProtocolRegistrationPersistence, the interface for
pluggable storage backends in the registration storage effect node.

Architecture:
    The protocol defines a capability-oriented interface for storage operations:
    - store_registration: Idempotent upsert operation
    - query_registrations: Flexible query with filtering and pagination
    - update_registration: Partial update of specific fields
    - delete_registration: Delete by node ID
    - health_check: Backend health and connectivity check

    Implementations include:
    - PostgresRegistrationStorageHandler: PostgreSQL backend

    The handler_type property identifies the backend for routing.

Protocol Duplication Note:
    This protocol is intentionally separate from the handler-level protocol
    at ``handlers/registration_storage/protocol_registration_persistence.py``:

    - **This protocol (nodes/*/protocols/)**: Node-level contract used for
      container-based dependency injection and registry binding.

    - **Handler protocol (handlers/*/protocol_*.py)**: Handler implementation
      contract used by tests for compliance verification.

    The separation allows:
    1. Node-level protocols to evolve independently of handler contracts
    2. Clear ownership boundaries (nodes own their protocols)
    3. Different import paths for different use cases (DI vs testing)

    Both protocols share the same name for discoverability, but serve
    different architectural layers.

Protocol Verification:
    Per ONEX conventions, protocol compliance is verified via duck typing
    rather than isinstance checks. The @runtime_checkable decorator enables
    structural subtyping checks.

    However, registries use isinstance() for fail-fast validation at
    registration time. See registry_infra_registration_storage.py for rationale.

Thread Safety:
    Handler implementations may be invoked concurrently. Implementations
    should be stateless or use appropriate synchronization.

Related:
    - NodeRegistrationStorageEffect: Effect node that uses this protocol
    - ModelRegistrationRecord: Record type for storage operations
    - ModelRegistrationUpdate: Update parameters for partial updates
    - ModelStorageQuery: Query parameters for retrieval
    - ModelStorageResult: Query result container
    - ModelDeleteResult: Delete operation result
    - ModelStorageHealthCheckResult: Health check result for handlers
    - ModelStorageHealthCheckDetails: Backend-specific health check diagnostics
    - ModelUpsertResult: Insert/update result
    - handlers/registration_storage/protocol_registration_persistence.py: Handler protocol
"""

from __future__ import annotations

__all__ = ["ProtocolRegistrationPersistence"]

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
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
    """Protocol for registration storage backend handlers.

    Defines the interface for pluggable storage backends. Each implementation
    provides the same operations but uses different storage technology.

    Core Principle:
        "I'm interested in what you do, not what you are"

        The protocol is named by capability (registration storage), not by
        implementation. Consumers interact with the protocol interface
        without knowing the underlying storage technology.

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing
        rather than isinstance checks. Verify required methods exist:

        .. code-block:: python

            # Duck typing check
            if (hasattr(handler, 'handler_type') and
                hasattr(handler, 'store_registration') and
                callable(handler.store_registration)):
                registry.register_handler(handler)

    Implementations:
        - PostgresRegistrationStorageHandler: PostgreSQL backend

    Example:
        .. code-block:: python

            class PostgresRegistrationStorageHandler:
                '''PostgreSQL implementation of registration storage.'''

                @property
                def handler_type(self) -> str:
                    return "postgresql"

                async def store_registration(
                    self,
                    record: ModelRegistrationRecord,
                    correlation_id: UUID | None = None,
                ) -> ModelUpsertResult:
                    # PostgreSQL-specific implementation
                    # Generate correlation_id if not provided
                    cid = correlation_id or uuid4()
                    ...

            # Usage
            handler = PostgresRegistrationStorageHandler(pool, config)
            result = await handler.store_registration(record)  # correlation_id optional
            result = await handler.store_registration(record, correlation_id)  # or explicit

    Attributes:
        handler_type: Identifier for the storage backend (e.g., "postgresql").

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.
    """

    @property
    def handler_type(self) -> str:
        """Return the storage backend type identifier.

        Used for handler routing and observability. Common values:
        - "postgresql": PostgreSQL relational database

        Returns:
            str: Handler type identifier (e.g., "postgresql")
        """
        ...

    async def store_registration(
        self,
        record: ModelRegistrationRecord,
        correlation_id: UUID | None = None,
    ) -> ModelUpsertResult:
        """Store a registration record (idempotent upsert).

        If a record with the same node_id exists, it is updated.
        Otherwise, a new record is inserted. This operation is idempotent.

        Args:
            record: The registration record to store.
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, implementations should generate one.

        Returns:
            ModelUpsertResult: Result indicating success/failure and operation type.

        Raises:
            InfraConnectionError: If unable to connect to storage backend.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def query_registrations(
        self,
        query: ModelStorageQuery,
        correlation_id: UUID | None = None,
    ) -> ModelStorageResult:
        """Query registration records with optional filters.

        Supports filtering by node_id, node_type, and capabilities.
        Supports pagination via limit and offset.

        Args:
            query: Query parameters including filters and pagination.
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, implementations should generate one.

        Returns:
            ModelStorageResult: Result containing matching records and total count.

        Raises:
            InfraConnectionError: If unable to connect to storage backend.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.
        """
        ...

    async def update_registration(
        self,
        request: ModelUpdateRegistrationRequest,
    ) -> ModelUpsertResult:
        """Update specific fields of a registration record.

        Performs a partial update, modifying only the specified fields.
        The node_id identifies the record to update.

        Args:
            request: ModelUpdateRegistrationRequest containing:
                - node_id: UUID of the registration record to update
                - updates: ModelRegistrationUpdate containing fields to update.
                  Only non-None fields will be applied. The model provides
                  type-safe partial updates for endpoints, metadata,
                  capabilities, and node_version fields.
                - correlation_id: Optional correlation ID for distributed tracing.
                  If not provided, implementations should generate one.

        Returns:
            ModelUpsertResult: Result indicating success/failure.

        Raises:
            InfraConnectionError: If unable to connect to storage backend.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.

        Example:
            >>> from omnibase_infra.handlers.registration_storage.models import (
            ...     ModelUpdateRegistrationRequest,
            ... )
            >>> from omnibase_infra.nodes.node_registration_storage_effect.models import (
            ...     ModelRegistrationUpdate,
            ... )
            >>> request = ModelUpdateRegistrationRequest(
            ...     node_id=node_id,
            ...     updates=ModelRegistrationUpdate(
            ...         endpoints={"health": "http://new-host:8080/health"},
            ...         metadata={"env": "production"},
            ...     ),
            ... )
            >>> result = await handler.update_registration(request)
        """
        ...

    async def delete_registration(
        self,
        request: ModelDeleteRegistrationRequest,
    ) -> ModelDeleteResult:
        """Delete a registration record by node ID.

        Removes the registration record if it exists. The result indicates
        whether the operation succeeded and whether a record was deleted.

        Args:
            request: ModelDeleteRegistrationRequest containing:
                - node_id: UUID of the registration record to delete
                - correlation_id: Optional correlation ID for distributed tracing.
                  If not provided, implementations should generate one.

        Returns:
            ModelDeleteResult: Result indicating success/failure and whether
                a record was actually deleted (deleted=False if not found).

        Raises:
            InfraConnectionError: If unable to connect to storage backend.
            InfraTimeoutError: If operation times out.
            InfraUnavailableError: If circuit breaker is open.

        Example:
            >>> from omnibase_infra.handlers.registration_storage.models import (
            ...     ModelDeleteRegistrationRequest,
            ... )
            >>> request = ModelDeleteRegistrationRequest(node_id=node_id)
            >>> result = await handler.delete_registration(request)
            >>> if result.was_deleted():
            ...     print("Record deleted successfully")
            >>> elif result.success and not result.deleted:
            ...     print("Record not found")
        """
        ...

    async def health_check(
        self,
        correlation_id: UUID | None = None,
    ) -> ModelStorageHealthCheckResult:
        """Check storage backend health and connectivity.

        Performs a lightweight connectivity check to verify the storage
        backend is reachable and responsive.

        Args:
            correlation_id: Optional correlation ID for distributed tracing.
                If not provided, implementations should generate one.

        Returns:
            ModelStorageHealthCheckResult: Health status including:
                - healthy: bool indicating overall health
                - backend_type: str identifying the backend
                - latency_ms: float connection latency
                - reason: str explaining the health status
                - error_type: str | None exception type if failed
                - details: ModelStorageHealthCheckDetails with typed diagnostics
                - correlation_id: UUID | None for tracing

        Example:
            >>> from omnibase_infra.nodes.node_registration_storage_effect.models import (
            ...     ModelStorageHealthCheckDetails,
            ... )
            >>> health = await handler.health_check()
            >>> health
            ModelStorageHealthCheckResult(
                healthy=True,
                backend_type="postgresql",
                latency_ms=2.5,
                reason="ok",
                details=ModelStorageHealthCheckDetails(
                    pool_size=10,
                    active_connections=3,
                ),
            )
        """
        ...
