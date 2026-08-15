# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One evaluated readiness signal for one runner (OMN-15255)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_readiness_signal_outcome import (
    EnumReadinessSignalOutcome,
)
from omnibase_infra.nodes.node_runner_fleet_health_compute.models.enum_runner_readiness_signal import (
    EnumRunnerReadinessSignal,
)


class ModelRunnerReadinessSignal(BaseModel):
    """The evaluated outcome of a single readiness signal.

    Every signal is emitted for every runner on every tick, including the
    PASSing ones. A fleet view that only reports failures cannot answer "was
    this checked?" -- which is the question the manual three-surface
    comparison existed to answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    signal: EnumRunnerReadinessSignal = Field(
        ..., description="Which readiness question this outcome answers."
    )
    outcome: EnumReadinessSignalOutcome = Field(
        ..., description="PASS / FAIL / UNKNOWN for this runner."
    )
    detail: str = Field(
        default="",
        description=(
            "Observed value and threshold in human-readable form. Populated "
            "for FAIL and UNKNOWN; empty for a plain PASS."
        ),
    )


__all__ = ["ModelRunnerReadinessSignal"]
