# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime health check dimension model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeHealthDimension(BaseModel):
    """A single health dimension in a runtime health check event."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: str = Field(
        ..., description="Dimension identifier (e.g. 'consumer_coverage')"
    )
    status: Literal["HEALTHY", "DEGRADED", "CRITICAL"] = Field(
        ..., description="Health status for this dimension"
    )
    detail: str = Field(default="", description="Human-readable detail or empty string")


__all__: list[str] = ["ModelRuntimeHealthDimension"]
