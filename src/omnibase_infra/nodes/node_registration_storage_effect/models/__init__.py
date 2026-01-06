# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Registration Storage Effect Node.

This module exports models used by the NodeRegistrationStorageEffect for
capability-oriented storage operations.

Available Models:
    - ModelRegistrationRecord: Complete registration record for storage
    - ModelStorageQuery: Query parameters with filtering and pagination
    - ModelStorageResult: Query results with records and metadata
    - ModelUpsertResult: Insert/update operation result

All models are:
    - Frozen (immutable after creation)
    - Extra="forbid" (no extra fields allowed)
    - Strongly typed (no Any types)
"""

from .model_registration_record import ModelRegistrationRecord
from .model_storage_query import ModelStorageQuery
from .model_storage_result import ModelStorageResult
from .model_upsert_result import ModelUpsertResult

__all__ = [
    "ModelRegistrationRecord",
    "ModelStorageQuery",
    "ModelStorageResult",
    "ModelUpsertResult",
]
