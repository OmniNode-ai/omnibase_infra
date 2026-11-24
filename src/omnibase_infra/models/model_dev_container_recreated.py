# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev_container_recreated event: one start of the dev runtime container (OMN-25912)."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_dev_recreate_cause import EnumDevRecreateCause


class ModelDevContainerRecreated(BaseModel):
    """Published by the deploy agent's tick for each start it sees on the container.

    Topic ``dev_container_recreated`` of node_dev_roll_events_effect. ``job_id`` and
    ``requested_by`` are set only for ``deploy_agent_job`` (and ``requested_by`` only when a
    manifest names the job's issuer); every other cause leaves them empty.
    ``same_build`` is true when the container came up on the image the previous reading
    named, so a recreate that changed nothing reads as one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    runtime_lane: Literal["dev"]
    container: str = Field(min_length=1)
    container_digest: str = Field(min_length=1)
    started_at: datetime
    restart_count: int = Field(ge=0)
    exit_code: int | None = None
    image_digest: str = ""
    image_ref: str = ""
    previous_image_digest: str = ""
    same_build: bool
    cause: EnumDevRecreateCause
    job_id: str = ""
    requested_by: str = ""

    @field_validator("started_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("started_at must carry a UTC offset")
        return value
