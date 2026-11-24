# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One reading of the dev runtime container's own start (OMN-25912)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_dev_recreate_cause import EnumDevRecreateCause


class ModelDevContainerObservation(BaseModel):
    """The container's StartedAt and RestartCount as docker inspect reports them.

    A start the restart policy makes (exit 0, no deploy job) shows up here and nowhere in
    the job records, so the reader matches each observation to a start event. ``cause`` and
    ``same_build`` come from the ``dev_container_recreated`` event the observation was built
    from (the cause the observer could prove, and whether the container came up on the image
    the previous reading named); they are empty for a reading taken straight from docker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    container: str = Field(min_length=1)
    started_at: datetime
    restart_count: int = Field(ge=0)
    exit_code: int | None = None
    cause: EnumDevRecreateCause | None = None
    same_build: bool = False

    @field_validator("started_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("started_at must carry a UTC offset")
        return value
