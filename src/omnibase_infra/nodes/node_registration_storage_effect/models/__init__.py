# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Registration Storage Effect Node.

This module exports models used by the NodeRegistrationStorageEffect for
capability-oriented storage operations.

Available Models:
    - ModelDeleteResult: Delete operation result
    - ModelRegistrationRecord: Complete registration record for storage
    - ModelRegistrationUpdate: Partial update parameters
    - ModelStorageHealthCheckDetails: Backend-specific health check diagnostics
    - ModelStorageHealthCheckResult: Health check result for storage backends
    - ModelStorageQuery: Query parameters with filtering and pagination
    - ModelStorageResult: Query results with records and metadata
    - ModelUpsertResult: Insert/update operation result

All models are:
    - Frozen (immutable after creation)
    - Extra="forbid" (no extra fields allowed)
    - Strongly typed (no Any types, no dict primitives in signatures)
"""

from .model_delete_result import ModelDeleteResult
from .model_registration_record import ModelRegistrationRecord
from .model_registration_update import ModelRegistrationUpdate
from .model_storage_health_check_details import ModelStorageHealthCheckDetails
from .model_storage_health_check_result import (
    ModelStorageHealthCheckResult,
)
from .model_storage_query import ModelStorageQuery
from .model_storage_result import ModelStorageResult
from .model_upsert_result import ModelUpsertResult

__all__ = [
    "ModelDeleteResult",
    "ModelRegistrationRecord",
    "ModelRegistrationUpdate",
    "ModelStorageHealthCheckDetails",
    "ModelStorageHealthCheckResult",
    "ModelStorageQuery",
    "ModelStorageResult",
    "ModelUpsertResult",
]
