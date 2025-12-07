# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enabled Protocols Configuration Model.

This module provides the Pydantic model for enabled protocol configuration.
"""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

# Literal type for valid protocol names
# These correspond to handler_registry constants:
# HANDLER_TYPE_HTTP, HANDLER_TYPE_DATABASE, HANDLER_TYPE_KAFKA, etc.
ProtocolName = Literal[
    "http",
    "database",
    "kafka",
    "vault",
    "consul",
    "redis",
    "grpc",
]


class ModelEnabledProtocolsConfig(BaseModel):
    """Enabled protocols configuration model.

    Defines which protocol types are enabled for the runtime.

    Attributes:
        enabled: List of enabled protocol type names (e.g., ['http', 'database'])
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    enabled: list[ProtocolName] = Field(
        default_factory=lambda: cast(list[ProtocolName], ["http", "database"]),
        description="List of enabled protocol type names",
    )


__all__: list[str] = ["ModelEnabledProtocolsConfig", "ProtocolName"]
