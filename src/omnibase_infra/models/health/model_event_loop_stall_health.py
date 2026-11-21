# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``/health`` body served while the runtime event loop is stalled (OMN-26316)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.health.model_event_loop_stall_details import (
    ModelEventLoopStallDetails,
)


class ModelEventLoopStallHealth(BaseModel):
    """A ``degraded`` ``/health`` answer naming a stalled event loop."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: Literal["degraded"] = "degraded"
    version: str
    details: ModelEventLoopStallDetails


__all__ = ["ModelEventLoopStallHealth"]
