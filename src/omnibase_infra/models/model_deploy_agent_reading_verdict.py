# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict on the dev deploy agent ticket's three readings (OMN-26724)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDeployAgentReadingVerdict(BaseModel):
    """One boolean per acceptance criterion and a reason for each that is not met."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ac1_met: bool
    ac2_met: bool
    ac3_met: bool
    reasons: tuple[str, ...] = ()
