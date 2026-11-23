# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request to write the one boot summary line (OMN-26561)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBootSummaryRequest(BaseModel):
    """The counts and the longest event-loop gap per boot phase, read at the first healthy answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parse_runs: int = Field(
        ge=0, description="Contract files actually parsed in this process"
    )
    topic_scans: int = Field(
        ge=0, description="Contract topic scans actually run in this process"
    )
    seconds_since_kernel_start: float = Field(
        ge=0.0,
        description="Seconds from the kernel's start to the first healthy answer",
    )
    phase_gap_seconds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Longest event-loop gap seen in each boot phase that ran, measured from "
            "before boot step 0; a phase that never ran has no entry"
        ),
    )
