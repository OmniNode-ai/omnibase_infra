# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ASGI mount factory for the MCP SDK protocol route (OMN-21894).

A contract's ``http_mount`` block names this factory and the dependencies it
needs; the runtime passes them as keyword arguments (``ModelHttpMount``). The
route is off until a contract declares the mount: nothing here activates it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from omnibase_infra.handlers.mcp.transport_streamable_http import (
    TransportMCPStreamableHttp,
)
from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from omnibase_spi.protocols.types.protocol_mcp_tool_types import (
        ProtocolMCPToolDefinition,
    )


def create_mcp_app(
    *,
    config: ModelMcpHandlerConfig,
    tools: Sequence[ProtocolMCPToolDefinition],
    tool_executor: Callable[[str, dict[str, object]], object],
) -> Starlette:
    """Build the SDK protocol app for the runtime to serve at ``config.path``."""
    return TransportMCPStreamableHttp(config).create_app(tools, tool_executor)


__all__ = ["create_mcp_app"]
