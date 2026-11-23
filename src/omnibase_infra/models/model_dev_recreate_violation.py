# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The event published for each dev recreate rule a start broke (OMN-25912)."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_dev_recreate_violation_kind import (
    EnumDevRecreateViolationKind,
)


class ModelDevRecreateViolationRow(BaseModel):
    """One start that shows the violation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    at: datetime
    requested_by: str
    announced: bool


class ModelDevRecreateViolation(BaseModel):
    """Published to ``dev_recreate_violation`` of node_dev_roll_events_effect.

    ``restart_count`` is set only for ``start_without_job_record``: the RestartCount the
    container reported at that start. ``event_id`` is derived from the case, so reading the
    same window twice names each violation once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    runtime_lane: Literal["dev"]
    kind: EnumDevRecreateViolationKind
    at: datetime
    window_start: datetime
    rows: tuple[ModelDevRecreateViolationRow, ...] = ()
    restart_count: int | None = Field(default=None, ge=0)
    detail: str
