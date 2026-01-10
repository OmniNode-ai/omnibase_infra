# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP integration module for ONEX.

Provides streamable HTTP transport and ONEX-to-MCP adapter for exposing
ONEX nodes as MCP tools.
"""

from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import ONEXToMCPAdapter
from omnibase_infra.handlers.mcp.protocols import ProtocolToolExecutor
from omnibase_infra.handlers.mcp.transport_streamable_http import (
    TransportMCPStreamableHttp,
)

__all__ = [
    "ONEXToMCPAdapter",
    "ProtocolToolExecutor",
    "TransportMCPStreamableHttp",
]
