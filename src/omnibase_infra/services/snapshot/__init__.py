# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot Repository Module.

Provides the SnapshotRepository class for managing point-in-time state
snapshots with injectable persistence backends, along with storage
backend implementations.

Exports:
    SnapshotRepository: Generic snapshot repository with CRUD, diff, and fork operations.
    StoreSnapshotInMemory: In-memory store implementation for testing.
    StoreSnapshotPostgres: PostgreSQL store implementation for production.

Related Tickets:
    - OMN-1246: SnapshotRepository Infrastructure Primitive
"""

from omnibase_infra.services.snapshot.snapshot_repository import SnapshotRepository
from omnibase_infra.services.snapshot.store_inmemory import StoreSnapshotInMemory
from omnibase_infra.services.snapshot.store_postgres import StoreSnapshotPostgres

__all__: list[str] = [
    "SnapshotRepository",
    "StoreSnapshotInMemory",
    "StoreSnapshotPostgres",
]
