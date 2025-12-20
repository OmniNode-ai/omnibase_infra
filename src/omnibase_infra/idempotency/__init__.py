# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Idempotency System.

This module provides idempotency checking and deduplication capabilities
for message processing in distributed systems.

Components:
    - Models: Pydantic models for idempotency records, check results, and configuration
    - Store: Persistent storage backends for idempotency records (PostgreSQL, in-memory)
    - Guard: Decorator and middleware for automatic idempotency checking

Models:
    - ModelIdempotencyRecord: Record of a processed message for deduplication
    - ModelIdempotencyCheckResult: Result of an idempotency check operation
    - ModelPostgresIdempotencyStoreConfig: Configuration for PostgreSQL store
    - ModelIdempotencyGuardConfig: Configuration for the idempotency guard

Stores:
    - InMemoryIdempotencyStore: In-memory store for testing (OMN-945)
    - PostgresIdempotencyStore: Production PostgreSQL store (OMN-945)

Example - InMemory (Testing):
    >>> from omnibase_infra.idempotency import InMemoryIdempotencyStore
    >>> from uuid import uuid4
    >>>
    >>> store = InMemoryIdempotencyStore()
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
"""

from omnibase_infra.idempotency.models import (
    ModelIdempotencyCheckResult,
    ModelIdempotencyGuardConfig,
    ModelIdempotencyRecord,
    ModelPostgresIdempotencyStoreConfig,
)
from omnibase_infra.idempotency.store_inmemory import InMemoryIdempotencyStore
from omnibase_infra.idempotency.store_postgres import PostgresIdempotencyStore

__all__ = [
    # Models
    "ModelIdempotencyCheckResult",
    "ModelIdempotencyGuardConfig",
    "ModelIdempotencyRecord",
    "ModelPostgresIdempotencyStoreConfig",
    # Stores
    "InMemoryIdempotencyStore",
    "PostgresIdempotencyStore",
]
