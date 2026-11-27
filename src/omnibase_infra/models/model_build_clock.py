# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Container start to first healthy /health for one build (OMN-26316)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelBuildClock(BaseModel):
    """The build clock and its verdict.

    ``first_healthy_at`` is the time of the first passing probe, an upper bound on
    the true first healthy moment, since probes are not continuous.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    #: None when the start stamp was not a Docker time; the verdict is then "unread".
    container_started_at: datetime | None
    first_healthy_at: datetime
    minutes_to_healthy: float
    limit_minutes: float
    exceeded: bool
    verdict: Literal["pass", "fail", "unread"]
    detail: str
