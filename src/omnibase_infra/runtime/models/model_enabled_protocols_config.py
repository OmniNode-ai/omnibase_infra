# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enabled Protocols Configuration Model.

This module provides the Pydantic model for enabled protocol configuration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEnabledProtocolsConfig(BaseModel):
    """Enabled protocols configuration model.

    Defines which protocol types are enabled for the runtime.

    Attributes:
        enabled: List of enabled protocol type names (e.g., ['http', 'database'])
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        extra="forbid",
    )

    enabled: list[str] = Field(
        default_factory=lambda: ["http", "database"],
        description="List of enabled protocol type names",
    )


__all__: list[str] = ["ModelEnabledProtocolsConfig"]
