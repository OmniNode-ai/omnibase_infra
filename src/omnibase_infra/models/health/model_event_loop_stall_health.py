# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``/health`` body served while the runtime event loop is stalled (OMN-26316)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_health_degraded_reason import (
    EnumHealthDegradedReason,
)


class ModelEventLoopStallDetails(BaseModel):
    """Details of a stalled-loop ``/health`` answer.

    The keys ``healthy``, ``degraded`` and ``runtime_health`` match the
    loop-hosted handler's body, so every existing reader parses this answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    healthy: Literal[False] = False
    degraded: Literal[True] = True
    degraded_reason: EnumHealthDegradedReason
    event_loop_stalled_seconds: float = Field(ge=0.0)
    wait_budget_seconds: float = Field(gt=0.0)
    runtime_health: None = None


class ModelEventLoopStallHealth(BaseModel):
    """A ``degraded`` ``/health`` answer naming a stalled event loop."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: Literal["degraded"] = "degraded"
    version: str
    details: ModelEventLoopStallDetails


__all__ = ["ModelEventLoopStallDetails", "ModelEventLoopStallHealth"]
