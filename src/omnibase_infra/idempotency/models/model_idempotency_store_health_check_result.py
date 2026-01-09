# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Idempotency Store Health Check Result Model.

This module provides the Pydantic model for health check results from
idempotency stores, replacing untyped dict[str, object] returns with
strongly-typed model instances.

Note:
    This model is domain-specific to idempotency stores and uses a Literal
    type for the `reason` field to constrain valid status values. For
    general-purpose handler health checks, see
    ``omnibase_infra.runtime.models.ModelHealthCheckResult``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelIdempotencyStoreHealthCheckResult(BaseModel):
    """Result of an idempotency store health check.

    This model represents the outcome of a health check operation on an
    idempotency store, providing structured diagnostics about the store's
    operational status.

    Attributes:
        healthy: Whether the store is healthy and operational.
            When True, the store can accept operations normally.
            When False, consult the reason field for diagnostics.
        reason: Descriptive reason for the health status.
            - "ok": Store is healthy and operational
            - "not_initialized": Store has not been initialized
            - "table_not_found": Required database table does not exist
            - "check_failed": Health check encountered an exception
        error_type: Exception type name if health check failed.
            Only populated when reason is "check_failed".
            Useful for debugging and categorizing failures.

    Example:
        >>> # Healthy store
        >>> result = ModelIdempotencyStoreHealthCheckResult(healthy=True, reason="ok")
        >>> if result.healthy:
        ...     print("Store is operational")
        Store is operational

        >>> # Failed health check
        >>> result = ModelIdempotencyStoreHealthCheckResult(
        ...     healthy=False,
        ...     reason="check_failed",
        ...     error_type="ConnectionRefusedError",
        ... )
        >>> if not result.healthy:
        ...     print(f"Store unhealthy: {result.reason} ({result.error_type})")
        Store unhealthy: check_failed (ConnectionRefusedError)
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    healthy: bool = Field(
        description="Whether the store is healthy and operational",
    )
    reason: Literal["ok", "not_initialized", "table_not_found", "check_failed"] = Field(
        description="Reason for the health status",
    )
    error_type: str | None = Field(
        default=None,
        description="Exception type name if health check failed",
    )


__all__: list[str] = [
    "ModelIdempotencyStoreHealthCheckResult",
]
