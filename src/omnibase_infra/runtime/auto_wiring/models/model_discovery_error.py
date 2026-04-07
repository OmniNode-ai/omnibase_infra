# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Discovery error model for contract scanning failures (OMN-7653)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDiscoveryError(BaseModel):
    """An error encountered during contract discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    entry_point_name: str = Field(..., description="Entry point that failed")
    package_name: str = Field(default="unknown", description="Package name")
    error: str = Field(..., description="Error message")
