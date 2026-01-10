# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP tool result model."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelMcpToolResult(BaseModel):
    """Result model for MCP tool invocation.

    Represents the response from a tool invocation to be returned via MCP.

    Attributes:
        success: Whether the tool execution succeeded.
        content: The result content (string or structured data).
        is_error: Whether this result represents an error.
        error_message: Error details if is_error is True.
        correlation_id: Correlation ID from the original request.
        execution_time_ms: Execution duration in milliseconds.
    """

    success: bool = Field(..., description="Whether the tool execution succeeded")
    content: str | dict[str, object] | list[object] = Field(
        ..., description="The result content (string or structured data)"
    )
    is_error: bool = Field(
        default=False, description="Whether this result represents an error"
    )
    error_message: str | None = Field(
        default=None, description="Error details if is_error is True"
    )
    correlation_id: UUID | None = Field(
        default=None, description="Correlation ID from the original request"
    )
    execution_time_ms: float | None = Field(
        default=None,
        description="Execution duration in milliseconds (sub-ms precision)",
    )

    model_config = {"frozen": True}


__all__ = ["ModelMcpToolResult"]
