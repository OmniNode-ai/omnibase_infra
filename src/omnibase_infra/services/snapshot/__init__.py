# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot Service Module.

Provides the ServiceSnapshot class for managing point-in-time state
snapshots with injectable persistence backends, along with storage
backend implementations.

Exports:
    SnapshotNotFoundError: Error raised when a requested snapshot does not exist.
    ServiceSnapshot: Generic snapshot service with CRUD, diff, and fork operations.
    StoreSnapshotInMemory: In-memory store implementation for testing.
    StoreSnapshotPostgres: PostgreSQL store implementation for production.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from omnibase_infra.services.snapshot.service_snapshot import (
    ServiceSnapshot,
    SnapshotNotFoundError,
)
from omnibase_infra.services.snapshot.store_inmemory import StoreSnapshotInMemory
from omnibase_infra.services.snapshot.store_postgres import StoreSnapshotPostgres

__all__: list[str] = [
    "ServiceSnapshot",
    "SnapshotNotFoundError",
    "StoreSnapshotInMemory",
    "StoreSnapshotPostgres",
]
