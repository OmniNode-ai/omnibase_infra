# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S104
# S104 disabled: Binding to 0.0.0.0 is intentional for container networking
"""MCP handler configuration model."""

from pydantic import BaseModel, Field


class ModelMcpHandlerConfig(BaseModel):
    """Configuration for MCP handler initialization.

    Attributes:
        host: Host to bind the MCP server to.
        port: Port for the MCP streamable HTTP endpoint.
        path: URL path for the MCP endpoint (default: "/mcp").
        stateless: Enable stateless mode for horizontal scaling.
        json_response: Return JSON responses instead of SSE streaming.
        timeout_seconds: Default timeout for tool execution.
        max_tools: Maximum number of tools to expose.
    """

    host: str = Field(default="0.0.0.0", description="Host to bind MCP server to")
    port: int = Field(default=8090, description="Port for MCP streamable HTTP endpoint")
    path: str = Field(default="/mcp", description="URL path for MCP endpoint")
    stateless: bool = Field(
        default=True, description="Enable stateless mode for horizontal scaling"
    )
    json_response: bool = Field(
        default=True, description="Return JSON responses instead of SSE streaming"
    )
    timeout_seconds: float = Field(
        default=30.0, description="Default timeout for tool execution"
    )
    max_tools: int = Field(default=100, description="Maximum number of tools to expose")

    model_config = {"frozen": True}


__all__ = ["ModelMcpHandlerConfig"]
