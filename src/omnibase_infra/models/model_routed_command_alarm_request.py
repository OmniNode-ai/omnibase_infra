# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routed commands and the job records each instance holds (OMN-26485)."""

from __future__ import annotations

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from omnibase_infra.models.model_job_record import ModelJobRecord
from omnibase_infra.models.model_routed_command import ModelRoutedCommand


class ModelRoutedCommandAlarmRequest(BaseModel):
    """One poll: the commands routed so far, the job records read, and the time of the poll.

    ``job_records`` maps an instance name to the records that instance's agent answers
    with. A command is answered by a record with its ``correlation_id`` on the instance
    it was routed to; a record on the other instance does not answer it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    commands: tuple[ModelRoutedCommand, ...]
    job_records: dict[str, tuple[ModelJobRecord, ...]] = Field(default_factory=dict)
    now: AwareDatetime
