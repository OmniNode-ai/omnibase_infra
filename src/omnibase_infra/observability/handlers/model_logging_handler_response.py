# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Response model for the structured logging handler.

This module defines the response model for HandlerLoggingStructured operations,
containing operation status, buffer metrics, and correlation tracking.

Usage:
    >>> from omnibase_infra.observability.handlers import ModelLoggingHandlerResponse
    >>> from omnibase_infra.enums import EnumResponseStatus
    >>> from uuid import uuid4
    >>>
    >>> response = ModelLoggingHandlerResponse(
    ...     status=EnumResponseStatus.SUCCESS,
    ...     operation="logging.emit",
    ...     message="Log entry buffered",
    ...     correlation_id=uuid4(),
    ...     buffer_size=42,
    ...     drop_count=0,
    ... )
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from omnibase_infra.enums import EnumResponseStatus


class ModelLoggingHandlerResponse(BaseModel):
    """Response model for logging handler operations.

    Attributes:
        status: Operation status (SUCCESS or ERROR).
        operation: The operation that was executed.
        message: Human-readable status message.
        correlation_id: Correlation ID for request tracing.
        buffer_size: Current number of entries in buffer (after operation).
        drop_count: Total number of entries dropped since handler init.
    """

    status: EnumResponseStatus
    operation: str
    message: str
    correlation_id: UUID
    buffer_size: int = Field(default=0, description="Current buffer size")
    drop_count: int = Field(default=0, description="Total dropped entries")


__all__: list[str] = [
    "ModelLoggingHandlerResponse",
]
