# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""MCP server lifecycle timeouts (OMN-22130)."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


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
    def _refuse_bool(cls, value: object) -> object:
        """Refuse ``true``/``false``: pydantic would read them as 1.0/0.0 seconds."""
        if isinstance(value, bool):
            raise ValueError("a timeout is a number of seconds, not a boolean")
        return value

    @classmethod
    def validate_contract_block(cls, value: object, handler_type: str) -> None:
        """Refuse a malformed ``server_timeouts`` block, naming the handler.

        A non-mapping, an unknown key or a non-positive number raises here
        instead of surfacing later as a pydantic error in ``initialize``.
        """
        try:
            cls.model_validate(value)
        except ValidationError as e:
            raise ProtocolConfigurationError(
                f"Invalid metadata.transport.server_timeouts in handler contract "
                f"for '{handler_type}': {e.error_count()} validation error(s)",
                context=ModelInfraErrorContext.with_correlation(
                    operation="extract_handler_config",
                    target_name=handler_type,
                ),
            ) from e


__all__ = ["ModelMcpServerTimeouts"]
