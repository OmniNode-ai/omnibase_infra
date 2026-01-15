# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Idempotency System.

This module provides idempotency checking and deduplication capabilities
for message processing in distributed systems.

Components:
    - Models: Pydantic models for idempotency records, check results, configuration, and metrics
    - Store: Persistent storage backends for idempotency records (PostgreSQL, in-memory)
    - Guard: Decorator and middleware for automatic idempotency checking

Models:
    - ModelIdempotencyRecord: Record of a processed message for deduplication
    - ModelIdempotencyCheckResult: Result of an idempotency check operation
    - ModelIdempotencyStoreMetrics: Metrics for store observability
    - ModelPostgresIdempotencyStoreConfig: Configuration for PostgreSQL store
    - ModelIdempotencyGuardConfig: Configuration for the idempotency guard

Stores:
    - StoreIdempotencyInmemory: In-memory store for testing (OMN-945)
    - StoreIdempotencyPostgres: Production PostgreSQL store (OMN-945)

Example - InMemory (Testing):
    >>> from omnibase_infra.idempotency import StoreIdempotencyInmemory
    >>> from uuid import uuid4
    >>>
    >>> store = StoreIdempotencyInmemory()
    >>> message_id = uuid4()
    >>>
    >>> # First call returns True (message is new)
    >>> result = await store.check_and_record(message_id, domain="test")
    >>> assert result is True
    >>>
    >>> # Second call returns False (duplicate)
    >>> result = await store.check_and_record(message_id, domain="test")
    >>> assert result is False

Example - Models:
    >>> from omnibase_infra.idempotency import (
    ...     ModelIdempotencyRecord,
    ...     ModelIdempotencyCheckResult,
    ...     ModelIdempotencyStoreMetrics,
    ...     ModelPostgresIdempotencyStoreConfig,
    ...     ModelIdempotencyGuardConfig,
    ... )
    >>> from uuid import uuid4
    >>> from datetime import datetime, timezone
    >>>
    >>> # Create an idempotency record
    >>> record = ModelIdempotencyRecord(
    ...     message_id=uuid4(),
    ...     domain="orders",
    ...     processed_at=datetime.now(timezone.utc),
    ... )
    >>>
    >>> # Configure the guard
    >>> guard_config = ModelIdempotencyGuardConfig(
    ...     enabled=True,
    ...     store_type="postgres",
    ...     domain_from_operation=True,
    ... )
    >>>
    >>> # Get store metrics
    >>> metrics = store.get_metrics()
    >>> print(f"Duplicate rate: {metrics.duplicate_rate:.2%}")
"""

from omnibase_infra.idempotency.models import (
    ModelIdempotencyCheckResult,
    ModelIdempotencyGuardConfig,
    ModelIdempotencyRecord,
    ModelIdempotencyStoreMetrics,
    ModelPostgresIdempotencyStoreConfig,
)
from omnibase_infra.idempotency.protocol_idempotency_store import (
    ProtocolIdempotencyStore,
)
from omnibase_infra.idempotency.store_inmemory import StoreIdempotencyInmemory
from omnibase_infra.idempotency.store_postgres import StoreIdempotencyPostgres

__all__: list[str] = [
    # Stores
    "StoreIdempotencyInmemory",
    # Models
    "ModelIdempotencyCheckResult",
    "ModelIdempotencyGuardConfig",
    "ModelIdempotencyRecord",
    "ModelIdempotencyStoreMetrics",
    "ModelPostgresIdempotencyStoreConfig",
    "StoreIdempotencyPostgres",
    # Protocol
    "ProtocolIdempotencyStore",
]
