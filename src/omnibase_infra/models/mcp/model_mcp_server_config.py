# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP server configuration model.

This model defines the configuration for the MCP server lifecycle,
including event bus registry discovery, Kafka hot reload, and HTTP server settings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMCPServerConfig(BaseModel):
    """Configuration for the MCP server lifecycle.

    This model captures all configuration needed for the MCP server:
    - Event bus registry settings for cold-start service discovery
    - Kafka settings for hot reload
    - HTTP server binding
    - Execution defaults
    - Authentication settings (OMN-2701)

    Attributes:
        registry_query_limit: Maximum nodes fetched per cold-start discovery query.
        kafka_enabled: Whether to enable Kafka for hot reload.
        http_host: Host to bind the MCP HTTP server.
        http_port: Port for the MCP HTTP server.
        default_timeout: Default execution timeout for tools.
        dev_mode: Whether to run in development mode (local contracts).
        contracts_dir: Directory for contract scanning in dev mode.
        auth_enabled: Whether bearer token / API-key auth middleware is active.
        api_key: API key / bearer token value for authenticated requests.
    """

    registry_query_limit: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum nodes to fetch per cold-start discovery query from the registry",
    )
    kafka_enabled: bool = Field(
        default=True, description="Whether to enable Kafka for hot reload"
    )
    http_host: str = Field(
        default="0.0.0.0",  # noqa: S104 - Intentional bind-all for server
        description="Host to bind the MCP HTTP server",
    )
    http_port: int = Field(
        default=8090, ge=1, le=65535, description="Port for the MCP HTTP server"
    )
    default_timeout: float = Field(
        default=30.0, gt=0, le=300, description="Default execution timeout for tools"
    )
    dev_mode: bool = Field(
        default=False, description="Whether to run in development mode"
    )
    contracts_dir: str | None = Field(
        default=None, description="Directory for contract scanning in dev mode"
    )
    auth_enabled: bool = Field(
        default=True,
        description=(
            "Whether bearer token / API-key auth middleware is active. "
            "When False, a WARNING is logged at startup. Default True."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Bearer token / API key required for authenticated MCP requests. "
            "Loaded from Infisical or env. Required when auth_enabled=True."
        ),
    )


__all__ = ["ModelMCPServerConfig"]
