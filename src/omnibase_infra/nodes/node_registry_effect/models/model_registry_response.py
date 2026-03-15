# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry Response Model for PostgreSQL Registration Operations.  # ai-slop-ok: pre-existing docstring opener

This module provides ModelRegistryResponse, representing the complete response
from the NodeRegistryEffect node after executing registration.

Architecture:
    ModelRegistryResponse captures the outcome of registering a node in
    PostgreSQL, with support for failure scenarios:

    - status=EnumRegistryResponseStatus.SUCCESS: Backend succeeded
    - status=EnumRegistryResponseStatus.FAILED: Backend failed

Related:
    - ModelBackendResult: Individual backend operation result
    - NodeRegistryEffect: Effect node that produces this response
    - ModelRegistryRequest: Input request model
    - OMN-954: Partial failure scenario testing
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumBackendType, EnumRegistryResponseStatus
from omnibase_infra.models.model_backend_result import (
    ModelBackendResult,
)


class ModelRegistryResponse(BaseModel):
    """Response model for PostgreSQL registration operations.

    Captures the complete outcome of registering a node in PostgreSQL,
    with individual results for the backend.

    Status Semantics:
        - SUCCESS: postgres_result.success is True
        - FAILED: postgres_result.success is False

    Immutability:
        This model uses frozen=True to ensure responses are immutable
        once created, supporting safe concurrent access and comparison.

    Attributes:
        status: Overall status of the registration operation.
        node_id: UUID of the node that was registered.
        correlation_id: Correlation ID for distributed tracing.
        postgres_result: Result of the PostgreSQL upsert operation.
        processing_time_ms: Total time for the registration operation.
        timestamp: When this response was created.
        error_summary: Aggregated error message for failed operations.

    Example (success):
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumRegistryResponseStatus
        >>> response = ModelRegistryResponse(
        ...     status=EnumRegistryResponseStatus.SUCCESS,
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     postgres_result=ModelBackendResult(
        ...         success=True, duration_ms=30.0, backend_id="postgres"
        ...     ),
        ...     processing_time_ms=30.0,
        ... )
        >>> response.status == EnumRegistryResponseStatus.SUCCESS
        True

    Example (failure):
        >>> response = ModelRegistryResponse(
        ...     status=EnumRegistryResponseStatus.FAILED,
        ...     node_id=uuid4(),
        ...     correlation_id=uuid4(),
        ...     postgres_result=ModelBackendResult(
        ...         success=False,
        ...         error="Connection refused",
        ...         duration_ms=5000.0,
        ...         backend_id="postgres",
        ...     ),
        ...     processing_time_ms=5000.0,
        ...     error_summary="PostgreSQL: Connection refused",
        ... )
        >>> response.status == EnumRegistryResponseStatus.FAILED
        True
        >>> response.postgres_result.success
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumRegistryResponseStatus = Field(
        ...,
        description="Overall status: success or failed",
    )
    node_id: UUID = Field(
        ...,
        description="UUID of the node that was registered",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    postgres_result: ModelBackendResult = Field(
        ...,
        description="Result of the PostgreSQL upsert operation",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Total time for the registration operation in milliseconds",
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
        postgres_result: ModelBackendResult,
        timestamp: datetime,
    ) -> ModelRegistryResponse:
        """Create a response from PostgreSQL backend result.

        Automatically determines the status based on backend success flag:
        - success -> SUCCESS
        - failure -> FAILED

        Processing time is taken from the backend duration.

        Args:
            node_id: UUID of the registered node.
            correlation_id: Correlation ID for tracing.
            postgres_result: Result from PostgreSQL upsert.
            timestamp: When this response was created (must be explicitly provided).

        Returns:
            ModelRegistryResponse with computed status, processing_time, and error_summary.
        """
        # Determine status based on backend result
        if postgres_result.success:
            status = EnumRegistryResponseStatus.SUCCESS
        else:
            status = EnumRegistryResponseStatus.FAILED

        # Calculate processing time from backend duration
        processing_time_ms = postgres_result.duration_ms

        # Build error summary from failed backend
        errors: list[str] = []
        if not postgres_result.success and postgres_result.error:
            errors.append(f"PostgreSQL: {postgres_result.error}")
        error_summary = "; ".join(errors) if errors else None

        return cls(
            status=status,
            node_id=node_id,
            correlation_id=correlation_id,
            postgres_result=postgres_result,
            processing_time_ms=processing_time_ms,
            timestamp=timestamp,
            error_summary=error_summary,
        )

    def is_complete_success(self) -> bool:
        """Check if the backend succeeded.

        Returns:
            True if status is SUCCESS, False otherwise.
        """
        return self.status == EnumRegistryResponseStatus.SUCCESS

    def is_complete_failure(self) -> bool:
        """Check if the backend failed.

        Returns:
            True if status is FAILED, False otherwise.
        """
        return self.status == EnumRegistryResponseStatus.FAILED

    def get_failed_backends(self) -> list[str]:
        """Get list of backends that failed.

        Returns:
            List of backend names that failed ("postgres").
        """
        failed: list[str] = []
        if not self.postgres_result.success:
            failed.append(EnumBackendType.POSTGRES.value)
        return failed

    def get_successful_backends(self) -> list[str]:
        """Get list of backends that succeeded.

        Returns:
            List of backend names that succeeded ("postgres").
        """
        succeeded: list[str] = []
        if self.postgres_result.success:
            succeeded.append(EnumBackendType.POSTGRES.value)
        return succeeded


__all__ = ["ModelRegistryResponse"]
