# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registration Storage Handlers Module.

Pluggable handler implementations for registration
storage operations, supporting the capability-oriented node architecture.

Handlers:
    - HandlerRegistrationStoragePostgres: PostgreSQL-backed storage
    - HandlerRegistrationStorageMock: In-memory mock for testing

Models (from omnibase_infra.nodes.node_registration_storage_effect.models):
    - ModelRegistrationRecord: Registration record model
    - ModelUpsertResult: Upsert operation result
    - ModelStorageResult: Storage query result

Protocols:
    - ProtocolRegistrationPersistence: Registration persistence protocol definition
"""

from omnibase_infra.handlers.registration_storage.handler_registration_storage_mock import (
    HandlerRegistrationStorageMock,
)
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
from omnibase_infra.handlers.registration_storage.protocol_registration_persistence import (
    ProtocolRegistrationPersistence,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)

__all__: list[str] = [
    "HandlerRegistrationStorageMock",
    "HandlerRegistrationStoragePostgres",
    "ModelRegistrationRecord",
    "ModelStorageResult",
    "ModelUpsertResult",
    "ProtocolRegistrationPersistence",
]
