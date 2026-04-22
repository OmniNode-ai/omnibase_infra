# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: S104
# S104 disabled: Binding to 0.0.0.0 is intentional for container networking
"""MCP handler configuration model."""

from pydantic import BaseModel, Field, model_validator


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
        auth_enabled: Whether bearer token / API-key auth middleware is active.
            When False, a WARNING is logged at startup. Default True.
        api_keys: Tuple of accepted API keys / bearer tokens. Every key listed is
            equally valid — this supports issuing separate credentials per client
            or service (OMN-1419). Loaded from Infisical or env (typically
            comma-separated and split by the caller). At least one non-empty
            key is required when ``auth_enabled`` is True.
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
    auth_enabled: bool = Field(
        default=True,
        description=(
            "Whether bearer token / API-key auth middleware is active. "
            "When False, a WARNING is logged at startup."
        ),
    )
    api_keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Accepted bearer tokens / API keys. Every listed key grants access. "
            "Supports multiple clients/services holding distinct credentials. "
            "At least one non-empty key is required when auth_enabled=True."
        ),
    )

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _validate_keys_when_auth_enabled(self) -> "ModelMcpHandlerConfig":
        """Reject empty/whitespace keys and enforce the auth_enabled contract."""
        if any((not k) or (not k.strip()) for k in self.api_keys):
            raise ValueError(
                "api_keys contains empty or whitespace-only entries; "
                "remove them before constructing the config"
            )
        if self.auth_enabled and not self.api_keys:
            raise ValueError(
                "auth_enabled=True but api_keys is empty. "
                "Provide at least one API key or set auth_enabled=False."
            )
        return self


__all__ = ["ModelMcpHandlerConfig"]
