# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""MCP server lifecycle timeouts (OMN-22130)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    shutdown_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    cancel_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    startup_seconds: float = Field(default=2.0, gt=0, allow_inf_nan=False)

    @field_validator(
        "shutdown_seconds", "cancel_seconds", "startup_seconds", mode="before"
    )
    @classmethod
    def _refuse_bool(cls, value: Any) -> Any:
        """Refuse ``true``/``false``: pydantic would read them as 1.0/0.0 seconds."""
        if isinstance(value, bool):
            raise ValueError("a timeout is a number of seconds, not a boolean")
        return value


__all__ = ["ModelMcpServerTimeouts"]
