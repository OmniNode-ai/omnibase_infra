# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev_build_health event: one dev build's time to health and whether dev went back (OMN-26703)."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDevBuildHealth(BaseModel):
    """Published once per start of the dev runtime container, when its health has settled.

    ``health_seconds`` is the time from the container's ``StartedAt`` to its first passing
    health check; it is ``None`` (and ``under_limit`` is ``None``) when no passing check was
    read, which is a build that never answered or one whose first pass left docker's short
    health log. ``rolled_back`` is true only when git proves the running image revision is
    an ancestor of the previous build's. ``no_rollback_since`` is where the 24 hour
    no-rollback clock starts: the start of the build that went back, else the start of the
    first build read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    runtime_lane: Literal["dev"]
    container: str = Field(min_length=1)
    started_at: datetime
    first_healthy_at: datetime | None = None
    health_seconds: float | None = None
    limit_seconds: int = Field(gt=0)
    under_limit: bool | None = None
    health_status: str = ""
    image_revision: str = ""
    previous_image_revision: str = ""
    revision_order: Literal["first", "same", "forward", "back", "unrelated", "unread"]
    rolled_back: bool
    no_rollback_since: datetime
