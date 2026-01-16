# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP tool call request model."""

from uuid import UUID

from pydantic import BaseModel, Field


class ModelMcpToolCall(BaseModel):
    """Request model for MCP tool invocation.

    Represents an incoming tool call from an AI agent via MCP protocol.

    Attributes:
        tool_name: Name of the tool to invoke.
        arguments: Tool parameters as key-value pairs.
        correlation_id: Optional correlation ID for request tracing.
    """

    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, object] = Field(
        default_factory=dict, description="Tool parameters as key-value pairs"
    )
    correlation_id: UUID | None = Field(
        default=None, description="Optional correlation ID for request tracing"
    )

    model_config = {"frozen": True}


__all__ = ["ModelMcpToolCall"]
