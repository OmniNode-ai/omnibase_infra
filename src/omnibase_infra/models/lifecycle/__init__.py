# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Lifecycle Models.

Provides models for lifecycle operation results, replacing tuple-based return
patterns with strongly-typed Pydantic models for better type safety and
expressiveness.

Note:
    This module re-exports lifecycle models from ``omnibase_infra.runtime.models``
    to maintain a single source of truth. The runtime models are the canonical
    implementations with full functionality required by ProtocolLifecycleExecutor.

Exports:
    ModelBatchLifecycleResult: Result of batch lifecycle operations
    ModelHealthCheckResult: Result of a handler health check operation
    ModelLifecycleResult: Result of a single lifecycle operation

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 3 (OMN-1003).

.. versionchanged:: 0.7.0
    Changed to re-export from runtime/models to eliminate duplicate models
    and consolidate to single source (OMN-1003 consolidation).
"""

from omnibase_infra.runtime.models.model_batch_lifecycle_result import (
    ModelBatchLifecycleResult,
)
from omnibase_infra.runtime.models.model_health_check_result import (
    ModelHealthCheckResult,
)
from omnibase_infra.runtime.models.model_lifecycle_result import ModelLifecycleResult

__all__ = [
    "ModelBatchLifecycleResult",
    "ModelHealthCheckResult",
    "ModelLifecycleResult",
]
