# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Deprecated: Health Check Result Model.

.. deprecated:: 0.7.0
    This module is deprecated. Use
    ``omnibase_infra.idempotency.models.model_idempotency_store_health_check_result``
    instead. This module is preserved for backward compatibility only.

The model has been renamed to ``ModelIdempotencyStoreHealthCheckResult`` to avoid
naming conflicts with ``omnibase_infra.runtime.models.ModelHealthCheckResult``.
"""

from __future__ import annotations

import warnings

from omnibase_infra.idempotency.models.model_idempotency_store_health_check_result import (
    ModelIdempotencyStoreHealthCheckResult,
)

# Issue deprecation warning on module import
warnings.warn(
    "omnibase_infra.idempotency.models.model_health_check_result is deprecated. "
    "Use omnibase_infra.idempotency.models.model_idempotency_store_health_check_result "
    "or import ModelIdempotencyStoreHealthCheckResult from omnibase_infra.idempotency.models.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward compatibility alias
ModelHealthCheckResult = ModelIdempotencyStoreHealthCheckResult

__all__: list[str] = ["ModelHealthCheckResult"]
