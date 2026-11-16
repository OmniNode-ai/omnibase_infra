# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev_runtime_started event: one dev runtime start on one roll job."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_dev_start_cause import EnumDevStartCause

JobId = Annotated[str, Field(min_length=1)]


class ModelDevRuntimeStarted(BaseModel):
    """Published by the deploy agent at each dev runtime start (five whys R1).

    Topic ``onex.evt.omnimarket.dev-runtime-started.v1``. The five fields are
    pinned by omnimarket node_dev_roll_verify_compute, which reads them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    job_id: JobId
    runtime_lane: Literal["dev"]
    started_at: datetime
    cause: EnumDevStartCause

    @field_validator("started_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("started_at must carry a UTC offset")
        return value
