# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Discovery error model for contract scanning failures (OMN-7653)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDiscoveryError(BaseModel):
    """An error encountered during contract discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    entry_point_name: str = Field(..., description="Entry point that failed")
    package_name: str = Field(default="unknown", description="Package name")
    error: str = Field(..., description="Error message")
    contract_path: Path | None = Field(
        default=None, description="Contract path, when resolved before the failure"
    )
    reason: Literal["discovery_error", "parse_error", "duplicate_contract_name"] = (
        Field(
            default="discovery_error",
            description="Machine-readable discovery failure reason",
        )
    )
