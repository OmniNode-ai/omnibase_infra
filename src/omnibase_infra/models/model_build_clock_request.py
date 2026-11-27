# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Container start and first healthy reading of one build (OMN-26316)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelBuildClockRequest(BaseModel):
    """What the deploy agent read: Docker's start stamp and the first passing /health."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    #: The runtime container's State.StartedAt exactly as Docker prints it.
    container_started_at: str
    #: Wall time of the first /health probe that passed.
    first_healthy_at: datetime
    #: A build past this many minutes fails.
    limit_minutes: float = 5.0
