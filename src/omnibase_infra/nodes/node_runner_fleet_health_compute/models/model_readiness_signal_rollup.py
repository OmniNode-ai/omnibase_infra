# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fleet-level rollup of one readiness signal (OMN-15255)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_readiness_signal import (
    EnumRunnerReadinessSignal,
)


class ModelReadinessSignalRollup(BaseModel):
    """How many runners FAILed / could not determine one readiness signal.

    This is the fleet view's answer to "which surface disagrees, and about how
    many runners" -- the thing an operator previously produced by hand by
    diffing ``gh api .../actions/runners`` against ``docker ps`` against
    per-container ``_diag`` mtimes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    signal: EnumRunnerReadinessSignal = Field(..., description="Signal rolled up.")
    fail_count: int = Field(
        ..., ge=0, description="Runners whose probe returned FAIL for this signal."
    )
    unknown_count: int = Field(
        ..., ge=0, description="Runners whose probe could not determine this signal."
    )


__all__ = ["ModelReadinessSignalRollup"]
