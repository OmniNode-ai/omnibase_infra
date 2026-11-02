# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""MCP server lifecycle timeouts (OMN-22130)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelMcpServerTimeouts(BaseModel):
    """Waits HandlerMCP applies to its uvicorn server, in seconds.

    Declared in the MCP handler contract under ``metadata.transport.server_timeouts``
    and passed to ``HandlerMCP.initialize(config)`` by ``extract_handler_config``.

    Attributes:
        shutdown_seconds: Wait for the server to stop gracefully.
        cancel_seconds: Wait for the server task after a forced cancel.
        startup_seconds: Wait for the server to accept connections.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    shutdown_seconds: float = Field(default=5.0, gt=0)
    cancel_seconds: float = Field(default=1.0, gt=0)
    startup_seconds: float = Field(default=2.0, gt=0)


__all__ = ["ModelMcpServerTimeouts"]
