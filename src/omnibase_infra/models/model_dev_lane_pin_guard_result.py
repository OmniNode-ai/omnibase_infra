# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the pin service guard found when it let a build proceed (OMN-23368)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_lane_pin_staleness import (
    ModelDevLanePinStaleness,
)


class ModelDevLanePinGuardResult(BaseModel):
    """The staleness reading; ``report`` is empty when the agent's code is not behind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: str
    staleness: ModelDevLanePinStaleness | None = None
