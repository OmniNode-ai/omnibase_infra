# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Response Model for Dual-Backend Registration Operations.

This module provides ModelRegistryResponse, representing the complete response
from the NodeRegistryEffect node after executing dual-backend registration.

Architecture:
    ModelRegistryResponse captures the outcome of registering a node in both
    Consul and PostgreSQL backends, with support for partial failure scenarios:

    - status=EnumRegistryResponseStatus.SUCCESS: Both backends succeeded
    - status=EnumRegistryResponseStatus.PARTIAL: One backend succeeded, one failed
    - status=EnumRegistryResponseStatus.FAILED: Both backends failed

    Each backend's individual result is captured in consul_result and
    postgres_result fields, enabling targeted retry strategies.

Partial Failure Handling:
    When one backend fails but the other succeeds:
    1. Status is set to EnumRegistryResponseStatus.PARTIAL
    2. The successful backend's result shows success=True
    3. The failed backend's result shows success=False with error details
    4. Retry logic can target only the failed backend

    This supports idempotent retry: if Consul already succeeded, only
    PostgreSQL needs to be retried on subsequent attempts.

Related:
    - ModelBackendResult: Individual backend operation result
    - NodeRegistryEffect: Effect node that produces this response
    - ModelRegistryRequest: Input request model
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumBackendType, EnumRegistryResponseStatus
from omnibase_infra.nodes.effects.models.model_backend_result import (
    ModelBackendResult,
)


class ModelRegistryResponse(BaseModel):
    """Response model for dual-backend registration operations.

    Captures the complete outcome of registering a node in both Consul
    and PostgreSQL backends, with individual results for each backend.

    Status Semantics:
        - SUCCESS: Both consul_result.success AND postgres_result.success are True
        - PARTIAL: Exactly one of consul_result.success or postgres_result.success is True
        - FAILED: Both consul_result.success AND postgres_result.success are False

    Immutability:
        This model uses frozen=True to ensure responses are immutable
        once created, supporting safe concurrent access and comparison.

    Attributes:
        status: Overall status of the dual-registration operation.
        node_id: UUID of the node that was registered.
        correlation_id: Correlation ID for distributed tracing.
        consul_result: Result of the Consul registration operation.
        postgres_result: Result of the PostgreSQL upsert operation.
        processing_time_ms: Total time for the dual-registration operation.
        timestamp: When this response was created.
        error_summary: Aggregated error message for failed operations.

    Example (success):
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumRegistryResponseStatus
        >>> response = ModelRegistryResponse(
        ...     status=EnumRegistryResponseStatus.SUCCESS,
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     consul_result=ModelBackendResult(
        ...         success=True, duration_ms=45.0, backend_id="consul"
        ...     ),
        ...     postgres_result=ModelBackendResult(
        ...         success=True, duration_ms=30.0, backend_id="postgres"
        ...     ),
        ...     processing_time_ms=75.0,
        ... )
        >>> response.status == EnumRegistryResponseStatus.SUCCESS
        True

    Example (partial failure):
        >>> response = ModelRegistryResponse(
        ...     status=EnumRegistryResponseStatus.PARTIAL,
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     consul_result=ModelBackendResult(
        ...         success=True, duration_ms=45.0, backend_id="consul"
        ...     ),
        ...     postgres_result=ModelBackendResult(
        ...         success=False,
        ...         error="Connection refused",
        ...         duration_ms=5000.0,
        ...         backend_id="postgres",
        ...     ),
        ...     processing_time_ms=5045.0,
        ...     error_summary="PostgreSQL: Connection refused",
        ... )
        >>> response.status == EnumRegistryResponseStatus.PARTIAL
        True
        >>> response.consul_result.success
        True
        >>> response.postgres_result.success
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumRegistryResponseStatus = Field(
        ...,
        description="Overall status: success, partial, or failed",
    )
    node_id: UUID = Field(
        ...,
        description="UUID of the node that was registered",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    consul_result: ModelBackendResult = Field(
        ...,
        description="Result of the Consul registration operation",
    )
    postgres_result: ModelBackendResult = Field(
        ...,
        description="Result of the PostgreSQL upsert operation",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total time for the dual-registration operation in milliseconds",
        ge=0.0,
    )
    # Timestamps - MUST be explicitly injected (no default_factory for testability)
    timestamp: datetime = Field(
        ...,
        description="When this response was created (must be explicitly provided)",
    )
    error_summary: str | None = Field(
        default=None,
        description="Aggregated error message for failed operations",
    )

    @classmethod
    def from_backend_results(
        cls,
        node_id: UUID,
        correlation_id: UUID,
        consul_result: ModelBackendResult,
        postgres_result: ModelBackendResult,
        timestamp: datetime,
    ) -> ModelRegistryResponse:
        """Create a response from individual backend results.

        Automatically determines the status based on backend success flags:
        - Both success -> SUCCESS
        - One success, one failure -> PARTIAL
        - Both failure -> FAILED

        Processing time is calculated from the sum of backend durations.

        Args:
            node_id: UUID of the registered node.
            correlation_id: Correlation ID for tracing.
            consul_result: Result from Consul registration.
            postgres_result: Result from PostgreSQL upsert.
            timestamp: When this response was created (must be explicitly provided).

        Returns:
            ModelRegistryResponse with computed status, processing_time, and error_summary.
        """
        # Determine status based on backend results
        if consul_result.success and postgres_result.success:
            status = EnumRegistryResponseStatus.SUCCESS
        elif consul_result.success or postgres_result.success:
            status = EnumRegistryResponseStatus.PARTIAL
        else:
            status = EnumRegistryResponseStatus.FAILED

        # Calculate processing time from backend durations
        processing_time_ms = consul_result.duration_ms + postgres_result.duration_ms

        # Build error summary from failed backends
        errors: list[str] = []
        if not consul_result.success and consul_result.error:
            errors.append(f"Consul: {consul_result.error}")
        if not postgres_result.success and postgres_result.error:
            errors.append(f"PostgreSQL: {postgres_result.error}")
        error_summary = "; ".join(errors) if errors else None

        return cls(
            status=status,
            node_id=node_id,
            correlation_id=correlation_id,
            consul_result=consul_result,
            postgres_result=postgres_result,
            processing_time_ms=processing_time_ms,
            timestamp=timestamp,
            error_summary=error_summary,
        )

    def is_complete_success(self) -> bool:
        """Check if both backends succeeded.

        Returns:
            True if status is SUCCESS, False otherwise.
        """
        return self.status == EnumRegistryResponseStatus.SUCCESS

    def is_partial_failure(self) -> bool:
        """Check if exactly one backend failed.

        Returns:
            True if status is PARTIAL, False otherwise.
        """
        return self.status == EnumRegistryResponseStatus.PARTIAL

    def is_complete_failure(self) -> bool:
        """Check if both backends failed.

        Returns:
            True if status is FAILED, False otherwise.
        """
        return self.status == EnumRegistryResponseStatus.FAILED

    def get_failed_backends(self) -> list[str]:
        """Get list of backends that failed.

        Returns:
            List of backend names that failed ("consul", "postgres").
        """
        failed: list[str] = []
        if not self.consul_result.success:
            failed.append(EnumBackendType.CONSUL.value)
        if not self.postgres_result.success:
            failed.append(EnumBackendType.POSTGRES.value)
        return failed

    def get_successful_backends(self) -> list[str]:
        """Get list of backends that succeeded.

        Returns:
            List of backend names that succeeded ("consul", "postgres").
        """
        succeeded: list[str] = []
        if self.consul_result.success:
            succeeded.append(EnumBackendType.CONSUL.value)
        if self.postgres_result.success:
            succeeded.append(EnumBackendType.POSTGRES.value)
        return succeeded


__all__ = ["ModelRegistryResponse"]
