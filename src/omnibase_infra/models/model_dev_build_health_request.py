# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the dev build health handler reads for one dev runtime container start (OMN-26703)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_dev_container_reading import ModelDevContainerReading

#: How the running image's revision stands against the previous build's, read from git.
RevisionOrder = Literal["first", "same", "forward", "back", "unrelated", "unread"]


class ModelDevBuildHealthRequest(BaseModel):
    """The container reading, the order of its image revision against the previous build's
    and the start of the no-rollback clock the previous build left.

    ``limit_seconds`` is the node contract's ``build_health_limit_seconds``, never a default.
    ``revision_order`` is a git read the caller makes (this handler reads no repository):
    ``first`` when no previous build is known, ``forward`` when the previous revision is an
    ancestor of this one, ``back`` when this one is an ancestor of the previous one,
    ``unrelated`` when neither is, ``unread`` when git could not say.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reading: ModelDevContainerReading
    previous_image_revision: str = ""
    revision_order: RevisionOrder
    previous_no_rollback_since: datetime | None = None
    limit_seconds: int = Field(gt=0)
