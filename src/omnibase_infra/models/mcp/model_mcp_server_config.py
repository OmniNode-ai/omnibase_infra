# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP server configuration model.

This model defines the configuration for the MCP server lifecycle,
including Consul discovery, Kafka hot reload, and HTTP server settings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMCPServerConfig(BaseModel):
    """Configuration for the MCP server lifecycle.

    This model captures all configuration needed for the MCP server:
    - Consul connection settings for service discovery
    - Kafka settings for hot reload
    - HTTP server binding
    - Execution defaults

    Attributes:
        consul_host: Consul server hostname for service discovery.
        consul_port: Consul server port.
        consul_scheme: HTTP scheme for Consul (http/https).
        consul_token: Optional ACL token for Consul authentication.
        kafka_enabled: Whether to enable Kafka for hot reload.
        http_host: Host to bind the MCP HTTP server.
        http_port: Port for the MCP HTTP server.
        default_timeout: Default execution timeout for tools.
        dev_mode: Whether to run in development mode (local contracts).
        contracts_dir: Directory for contract scanning in dev mode.
    """

    consul_host: str = Field(default="localhost", description="Consul server hostname")
    consul_port: int = Field(
        default=8500, ge=1, le=65535, description="Consul server port"
    )
    consul_scheme: str = Field(
        default="http", pattern="^https?$", description="HTTP scheme for Consul"
    )
    consul_token: str | None = Field(
        default=None, description="Optional ACL token for Consul authentication"
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


__all__ = ["ModelMCPServerConfig"]
