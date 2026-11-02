# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler contract ``metadata.transport`` keys copied into the handler config.

``extract_handler_config`` copies each key a handler contract declares under
``metadata.transport`` into the flat config the runtime passes to
``handler.initialize(config)``. A ``default_`` prefix is dropped on the way:
``default_port`` in the contract is ``port`` in the config. The
``server_timeouts`` block is OMN-22130.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.types import JsonType


class EnumContractTransportKey(StrEnum):
    """A ``metadata.transport`` key of a handler contract."""

    DEFAULT_HOST = "default_host"
    DEFAULT_PORT = "default_port"
    DEFAULT_PATH = "default_path"
    STATELESS = "stateless"
    JSON_RESPONSE = "json_response"
    SERVER_TIMEOUTS = "server_timeouts"

    @property
    def config_key(self) -> str:
        """The key the value takes in the flat handler config."""
        return self.value.removeprefix("default_")

    def checked(self, transport: dict[str, JsonType], handler_type: str) -> JsonType:
        """Return this key's value in ``transport``, refusing a malformed ``server_timeouts`` block.

        A non-mapping, an unknown key or a non-positive number raises here,
        naming the handler, instead of surfacing later as a pydantic error in
        ``initialize``. The other keys are copied as declared.
        """
        value = transport[self.value]
        if self is not EnumContractTransportKey.SERVER_TIMEOUTS:
            return value
        # Imported here: the model layer imports enums, not the reverse.
        from pydantic import ValidationError

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.handlers.models.mcp.model_mcp_server_timeouts import (
            ModelMcpServerTimeouts,
        )
        from omnibase_infra.models.errors.model_infra_error_context import (
            ModelInfraErrorContext,
        )

        try:
            ModelMcpServerTimeouts.model_validate(value)
        except ValidationError as e:
            raise ProtocolConfigurationError(
                f"Invalid metadata.transport.server_timeouts in handler contract "
                f"for '{handler_type}': {e.error_count()} validation error(s)",
                context=ModelInfraErrorContext.with_correlation(
                    operation="extract_handler_config",
                    target_name=handler_type,
                ),
            ) from e
        return value


__all__ = ["EnumContractTransportKey"]
