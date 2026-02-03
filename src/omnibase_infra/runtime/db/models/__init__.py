# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Runtime Models Module.

This module exports Pydantic models for database runtime configuration
and repository contracts.

Exports:
    ModelRepositoryRuntimeConfig: Configuration for PostgresRepositoryRuntime
        - Safety constraints (max_row_limit, timeout_ms)
        - Operation allowlisting (select, insert, update, upsert)
        - Feature flags (allow_raw_operations, allow_delete_operations)
        - Determinism controls (primary_key_column, default_order_by)
        - Metrics emission configuration

    ModelDbRepositoryContract: Repository contract definition
    ModelDbOperation: Individual database operation definition
    ModelDbParam: Parameter definition for operations
    ModelDbReturn: Return type specification
    ModelDbSafetyPolicy: Safety constraints for operations
"""

from __future__ import annotations

# Contract models (local to omnibase_infra since 0.3.2)
from omnibase_infra.runtime.db.models.model_db_operation import ModelDbOperation
from omnibase_infra.runtime.db.models.model_db_param import ModelDbParam
from omnibase_infra.runtime.db.models.model_db_repository_contract import (
    ModelDbRepositoryContract,
)
from omnibase_infra.runtime.db.models.model_db_return import ModelDbReturn
from omnibase_infra.runtime.db.models.model_db_safety_policy import ModelDbSafetyPolicy

# Runtime config is local to omnibase_infra
from omnibase_infra.runtime.db.models.model_repository_runtime_config import (
    ModelRepositoryRuntimeConfig,
)

__all__: list[str] = [
    "ModelDbOperation",
    "ModelDbParam",
    "ModelDbRepositoryContract",
    "ModelDbReturn",
    "ModelDbSafetyPolicy",
    "ModelRepositoryRuntimeConfig",
]
