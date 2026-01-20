# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP services for Model Context Protocol integration.

This package provides services for exposing ONEX orchestrator nodes
as MCP tools for AI agent integration.

Services:
    ServiceMCPToolRegistry: Event-loop safe in-memory cache of tool definitions
    ServiceMCPToolDiscovery: Consul scanner for MCP-enabled orchestrators
    ServiceMCPToolSync: Kafka listener for hot reload with idempotency
    MCPServerLifecycle: Server lifecycle management
"""

from omnibase_infra.services.mcp.mcp_server_lifecycle import (
    MCPServerLifecycle,
    ModelMCPServerConfig,
)
from omnibase_infra.services.mcp.service_mcp_tool_discovery import (
    ServiceMCPToolDiscovery,
)
from omnibase_infra.services.mcp.service_mcp_tool_registry import ServiceMCPToolRegistry
from omnibase_infra.services.mcp.service_mcp_tool_sync import ServiceMCPToolSync

__all__ = [
    "MCPServerLifecycle",
    "ModelMCPServerConfig",
    "ServiceMCPToolDiscovery",
    "ServiceMCPToolRegistry",
    "ServiceMCPToolSync",
]
