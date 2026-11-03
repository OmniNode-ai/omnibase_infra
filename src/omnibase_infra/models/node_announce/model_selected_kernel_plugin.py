# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A kernel plugin the overlay enabled and the catalog admitted (OMN-21591)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSelectedKernelPlugin(BaseModel):
    """One enabled plugin, with where its class is imported from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="Plugin name, as the overlay names it")
    distribution: str = Field(description="Distribution that announced the plugin")
    target: str = Field(
        min_length=1, description="Importable ``module:attribute`` of the plugin class"
    )
