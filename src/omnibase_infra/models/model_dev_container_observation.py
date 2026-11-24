# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One reading of the dev runtime container's own start (OMN-25912)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelDevContainerObservation(BaseModel):
    """The container's StartedAt and RestartCount as docker inspect reports them.

    A start the restart policy makes (exit 0, no deploy job) shows up here and nowhere in
    the job records, so the reader matches each observation to a start event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    container: str = Field(min_length=1)
    started_at: datetime
    restart_count: int = Field(ge=0)
    exit_code: int | None = None

    @field_validator("started_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("started_at must carry a UTC offset")
        return value
