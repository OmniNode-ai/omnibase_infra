# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One docker inspect reading of the dev runtime container (OMN-25912)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelDevContainerReading(BaseModel):
    """The container as ``docker inspect`` reports it at one tick.

    ``container_id`` changes when compose recreates the container; ``started_at`` and
    ``restart_count`` change when the same container starts again. ``image_id`` is the
    image the container came up on and ``image_ref`` the reference its config names.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    container: str = Field(min_length=1)
    container_id: str = Field(min_length=1)
    started_at: datetime
    restart_count: int = Field(ge=0)
    exit_code: int | None = None
    image_id: str = ""
    image_ref: str = ""

    @field_validator("started_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("started_at must carry a UTC offset")
        return value
