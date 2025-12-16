# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enabled Protocols Configuration Model.

This module provides the Pydantic model for enabled protocol configuration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Literal type for valid protocol names
# These correspond to handler_registry constants:
# HANDLER_TYPE_HTTP, HANDLER_TYPE_DATABASE, HANDLER_TYPE_KAFKA, etc.
# NOTE: Values must match the operation prefixes used in handler implementations
# (e.g., "db" matches "db.query", "valkey" matches "valkey.get")
ProtocolName = Literal[
    "http",
    "db",
    "kafka",
    "vault",
    "consul",
    "valkey",
    "grpc",
]


def _default_enabled_protocols() -> list[ProtocolName]:
    """Create default list of enabled protocols.

    Returns a properly typed list of default protocol names.
    Using a factory function avoids cast() in default_factory for type safety.
    """
    return ["http", "db"]


class ModelEnabledProtocolsConfig(BaseModel):
    """Enabled protocols configuration model.

    Defines which protocol types are enabled for the runtime.

    Attributes:
        enabled: List of enabled protocol type names (e.g., ['http', 'db'])
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    enabled: list[ProtocolName] = Field(
        default_factory=_default_enabled_protocols,
        description="List of enabled protocol type names",
    )


__all__: list[str] = ["ModelEnabledProtocolsConfig", "ProtocolName"]
