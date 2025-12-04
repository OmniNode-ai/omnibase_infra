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


class ModelInfraErrorContext(BaseModel):
    """Configuration model for infrastructure error context.

    Encapsulates common structured fields for infrastructure errors
    to reduce __init__ parameter count while maintaining strong typing.
    This follows the ONEX pattern of using configuration models to
    bundle related parameters.

    Attributes:
        handler_type: Type of handler (http, db, kafka, consul, vault, etc.)
        operation: Operation being performed (connect, query, authenticate, etc.)
        service_name: Service or resource name
        correlation_id: Request correlation ID for distributed tracing

    Example:
        >>> context = ModelInfraErrorContext(
        ...     handler_type="http",
        ...     operation="process_request",
        ...     service_name="api-gateway",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise RuntimeHostError("Operation failed", context=context)
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields
    )

    handler_type: Optional[str] = Field(
        default=None,
        description="Type of handler (http, db, kafka, consul, vault, etc.)",
    )
    operation: Optional[str] = Field(
        default=None,
        description="Operation being performed (connect, query, authenticate, etc.)",
    )
    service_name: Optional[str] = Field(
        default=None,
        description="Service or resource name",
    )
    correlation_id: Optional[UUID] = Field(
        default=None,
        description="Request correlation ID for distributed tracing",
    )


__all__ = ["ModelInfraErrorContext"]
