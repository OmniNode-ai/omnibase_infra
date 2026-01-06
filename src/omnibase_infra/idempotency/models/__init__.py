# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Idempotency Models.

This module provides Pydantic models for the idempotency system,
including records, check results, configuration, and metrics models.

Exports:
    ModelIdempotencyStoreHealthCheckResult: Result of an idempotency store health check
    ModelIdempotencyRecord: Represents a stored idempotency record
    ModelIdempotencyCheckResult: Result of an idempotency check operation
    ModelIdempotencyGuardConfig: Configuration for idempotency guard middleware
    ModelIdempotencyStoreMetrics: Metrics for store observability
    ModelPostgresIdempotencyStoreConfig: Configuration for PostgreSQL store
"""

from omnibase_infra.idempotency.models.model_idempotency_check_result import (
    ModelIdempotencyCheckResult,
)
from omnibase_infra.idempotency.models.model_idempotency_guard_config import (
    ModelIdempotencyGuardConfig,
)
from omnibase_infra.idempotency.models.model_idempotency_record import (
    ModelIdempotencyRecord,
)
from omnibase_infra.idempotency.models.model_idempotency_store_health_check_result import (
    ModelIdempotencyStoreHealthCheckResult,
)
from omnibase_infra.idempotency.models.model_idempotency_store_metrics import (
    ModelIdempotencyStoreMetrics,
)
from omnibase_infra.idempotency.models.model_postgres_idempotency_store_config import (
    ModelPostgresIdempotencyStoreConfig,
)

__all__ = [
    "ModelIdempotencyCheckResult",
    "ModelIdempotencyGuardConfig",
    "ModelIdempotencyRecord",
    "ModelIdempotencyStoreHealthCheckResult",
    "ModelIdempotencyStoreMetrics",
    "ModelPostgresIdempotencyStoreConfig",
]
