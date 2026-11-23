# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One deploy agent job record, as the overdue dev merge build decision reads it (OMN-26316)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_job_status import EnumJobStatus


class ModelOverdueDevMergeJob(BaseModel):
    """The fields of one job record under the agent's state directory that the decision reads.

    ``accepted_at`` is when the agent accepted the job: it staged its source then, so a job
    accepted before a merge never carries that merge. ``status`` is the record's own status
    (``accepted``, ``in_progress``, ``success``, ``failed`` or ``superseded``).
    ``git_ref`` is the command's ref: ``origin/dev`` builds the dev head at acceptance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
    runtime_lane: str = Field(min_length=1)
    accepted_at: datetime
    status: EnumJobStatus
    git_ref: str = Field(min_length=1)
