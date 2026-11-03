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


__all__ = ["EnumContractTransportKey"]
