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

from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
        error_message: Error description if status is ERROR.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EnumResponseStatus
    operation: str
    message: str
    correlation_id: UUID
    buffer_size: int = Field(default=0, ge=0, description="Current buffer size")
    drop_count: int = Field(default=0, ge=0, description="Total dropped entries")
    error_message: str | None = Field(
        default=None,
        description="Error description if status is ERROR",
    )

    @model_validator(mode="after")
    def _validate_error_message_required_on_error(self) -> Self:
        """Validate that error_message is provided when status is ERROR.

        This validator ensures API consumers always receive a meaningful error
        description when an operation fails.

        Raises:
            ValueError: If status is ERROR but error_message is None or empty.
        """
        if self.status == EnumResponseStatus.ERROR:
            if not self.error_message:
                raise ValueError("error_message is required when status is ERROR")
        return self


__all__: list[str] = [
    "ModelLoggingHandlerResponse",
]
