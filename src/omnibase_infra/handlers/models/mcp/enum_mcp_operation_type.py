# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP operation type enumeration."""

from enum import Enum


class EnumMcpOperationType(str, Enum):
    """Operation types for MCP handler.

    The MCP handler supports three primary operations:
        - LIST_TOOLS: Discover available tools from registered ONEX nodes
        - CALL_TOOL: Invoke a specific tool with parameters
        - DESCRIBE: Return handler metadata and capabilities

    These map to the MCP protocol's tools/list and tools/call methods.
    """

    LIST_TOOLS = "mcp.list_tools"
    CALL_TOOL = "mcp.call_tool"
    DESCRIBE = "mcp.describe"


__all__ = ["EnumMcpOperationType"]
