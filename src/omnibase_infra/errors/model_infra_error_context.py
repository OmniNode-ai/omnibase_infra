# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Error Context Configuration Model.

This module defines the configuration model for infrastructure error context,
encapsulating common structured fields to reduce __init__ parameter count
while maintaining strong typing per ONEX standards.
"""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType


class ModelInfraErrorContext(BaseModel):
    """Configuration model for infrastructure error context.

    Encapsulates common structured fields for infrastructure errors
    to reduce __init__ parameter count while maintaining strong typing.
    This follows the ONEX pattern of using configuration models to
    bundle related parameters.

    Attributes:
        transport_type: Type of infrastructure transport (HTTP, DATABASE, KAFKA, etc.)
        operation: Operation being performed (connect, query, authenticate, etc.)
        target_name: Target resource or endpoint name
        correlation_id: Request correlation ID for distributed tracing

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="process_request",
        ...     target_name="api-gateway",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise RuntimeHostError("Operation failed", context=context)
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields
    )

    transport_type: Optional[EnumInfraTransportType] = Field(
        default=None,
        description="Type of infrastructure transport (HTTP, DATABASE, KAFKA, etc.)",
    )
    operation: Optional[str] = Field(
        default=None,
        description="Operation being performed (connect, query, authenticate, etc.)",
    )
    target_name: Optional[str] = Field(
        default=None,
        description="Target resource or endpoint name",
    )
    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Request correlation ID for distributed tracing",
    )


__all__ = ["ModelInfraErrorContext"]
