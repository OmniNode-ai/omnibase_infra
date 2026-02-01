# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database Runtime Models Module.

This module exports Pydantic models for database runtime configuration.
Contract models (ModelDbRepositoryContract, ModelDbOperation, ModelDbReturn)
are imported from omnibase_core.models.contracts.

Exports:
    ModelRepositoryRuntimeConfig: Configuration for PostgresRepositoryRuntime
        - Safety constraints (max_row_limit, timeout_ms)
        - Operation allowlisting (select, insert, update, upsert)
        - Feature flags (allow_raw_operations, allow_delete_operations)
        - Determinism controls (primary_key_column, default_order_by)
        - Metrics emission configuration

    ModelDbRepositoryContract: (re-export from omnibase_core)
    ModelDbOperation: (re-export from omnibase_core)
    ModelDbReturn: (re-export from omnibase_core)
"""

from __future__ import annotations

# Contract models from omnibase_core (canonical source)
from omnibase_core.models.contracts import (
    ModelDbOperation,
    ModelDbRepositoryContract,
    ModelDbReturn,
)

# Runtime config is local to omnibase_infra
from omnibase_infra.runtime.db.models.model_repository_runtime_config import (
    ModelRepositoryRuntimeConfig,
)

__all__: list[str] = [
    "ModelDbOperation",
    "ModelDbRepositoryContract",
    "ModelDbReturn",
    "ModelRepositoryRuntimeConfig",
]
