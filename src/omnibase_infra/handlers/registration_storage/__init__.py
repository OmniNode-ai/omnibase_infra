# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Storage Handlers Module.

This module provides pluggable handler implementations for registration
storage operations, supporting the capability-oriented node architecture.

Handlers:
    - PostgresRegistrationStorageHandler: PostgreSQL-backed storage
    - MockRegistrationStorageHandler: In-memory mock for testing

Models (from omnibase_infra.nodes.node_registration_storage_effect.models):
    - ModelRegistrationRecord: Registration record model
    - ModelUpsertResult: Upsert operation result
    - ModelStorageResult: Storage query result

Protocols:
    - ProtocolRegistrationStorageHandler: Handler protocol definition
"""

from omnibase_infra.handlers.registration_storage.handler_mock_registration_storage import (
    MockRegistrationStorageHandler,
)
from omnibase_infra.handlers.registration_storage.handler_postgres_registration_storage import (
    PostgresRegistrationStorageHandler,
)
from omnibase_infra.handlers.registration_storage.protocol_registration_storage_handler import (
    ProtocolRegistrationStorageHandler,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationRecord,
    ModelStorageResult,
    ModelUpsertResult,
)

__all__: list[str] = [
    "MockRegistrationStorageHandler",
    "ModelRegistrationRecord",
    "ModelStorageResult",
    "ModelUpsertResult",
    "PostgresRegistrationStorageHandler",
    "ProtocolRegistrationStorageHandler",
]
