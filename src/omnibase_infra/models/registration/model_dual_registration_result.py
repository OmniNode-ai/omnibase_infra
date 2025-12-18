# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dual Registration Result Model.

This module provides ModelDualRegistrationResult for capturing the outcome of
parallel Consul + PostgreSQL registration attempts in the ONEX 2-way registration
pattern.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDualRegistrationResult(BaseModel):
    """Result model for dual registration operations.

    Captures the outcome of parallel Consul + PostgreSQL registration attempts
    performed by the dual registration reducer. The status field reflects the
    combined outcome of both registration targets.

    Status semantics:
        - "success": Both Consul and PostgreSQL registrations succeeded.
        - "partial": One registration succeeded, the other failed.
        - "failed": Both registrations failed.

    Attributes:
        node_id: Unique identifier of the registered node (UUID).
        consul_registered: Whether Consul registration succeeded.
        postgres_registered: Whether PostgreSQL registration succeeded.
        status: Combined registration outcome (success, partial, failed).
        consul_error: Error message if Consul registration failed (None if succeeded).
        postgres_error: Error message if PostgreSQL registration failed (None if succeeded).
        registration_time_ms: Total time taken for registration in milliseconds (>= 0.0).
        correlation_id: Request correlation ID for distributed tracing (UUID).

    Example:
        >>> from uuid import uuid4
        >>> result = ModelDualRegistrationResult(
        ...     node_id=uuid4(),
        ...     consul_registered=True,
        ...     postgres_registered=True,
        ...     status="success",
        ...     registration_time_ms=42.5,
        ...     correlation_id=uuid4(),
        ... )
        >>> result.status
        'success'

    Example (partial failure):
        >>> from uuid import uuid4
        >>> result = ModelDualRegistrationResult(
        ...     node_id=uuid4(),
        ...     consul_registered=True,
        ...     postgres_registered=False,
        ...     status="partial",
        ...     postgres_error="Connection refused",
        ...     registration_time_ms=150.0,
        ...     correlation_id=uuid4(),
        ... )
        >>> result.postgres_error
        'Connection refused'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    # Required fields
    node_id: UUID = Field(..., description="Unique identifier of the registered node")
    consul_registered: bool = Field(
        ..., description="Whether Consul registration succeeded"
    )
    postgres_registered: bool = Field(
        ..., description="Whether PostgreSQL registration succeeded"
    )
    status: Literal["success", "partial", "failed"] = Field(
        ..., description="Combined registration outcome"
    )
    registration_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total time taken for registration in milliseconds",
    )
    correlation_id: UUID = Field(
        ..., description="Request correlation ID for distributed tracing"
    )

    # Optional error fields
    consul_error: str | None = Field(
        default=None, description="Error message if Consul registration failed"
    )
    postgres_error: str | None = Field(
        default=None, description="Error message if PostgreSQL registration failed"
    )

    @model_validator(mode="after")
    def validate_status_consistency(self) -> ModelDualRegistrationResult:
        """Validate that status matches registration outcomes.

        Rules:
            - status="success" requires both consul_registered=True AND postgres_registered=True
            - status="partial" requires exactly one of consul_registered or postgres_registered=True
            - status="failed" requires both consul_registered=False AND postgres_registered=False

        Additionally validates error field consistency:
            - consul_error should only be set if consul_registered=False
            - postgres_error should only be set if postgres_registered=False

        Returns:
            The validated model instance.

        Raises:
            ValueError: If status does not match registration outcomes or error fields
                are inconsistent with registration states.
        """
        both_succeeded = self.consul_registered and self.postgres_registered
        both_failed = not self.consul_registered and not self.postgres_registered
        one_succeeded = self.consul_registered != self.postgres_registered

        # Validate status matches outcomes
        if self.status == "success" and not both_succeeded:
            raise ValueError(
                "status='success' requires both consul_registered=True and "
                "postgres_registered=True"
            )
        if self.status == "partial" and not one_succeeded:
            raise ValueError(
                "status='partial' requires exactly one of consul_registered or "
                "postgres_registered to be True"
            )
        if self.status == "failed" and not both_failed:
            raise ValueError(
                "status='failed' requires both consul_registered=False and "
                "postgres_registered=False"
            )

        # Validate error field consistency
        if self.consul_registered and self.consul_error is not None:
            raise ValueError(
                "consul_error should not be set when consul_registered=True"
            )
        if self.postgres_registered and self.postgres_error is not None:
            raise ValueError(
                "postgres_error should not be set when postgres_registered=True"
            )

        return self


__all__ = ["ModelDualRegistrationResult"]
