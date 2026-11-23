# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One deploy agent job record, as the overdue dev merge build decision reads it (OMN-26316)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelOverdueDevMergeJob(BaseModel):
    """The fields of one job record under the agent's state directory that the decision reads.

    ``accepted_at`` is when the agent accepted the job: it staged its source then, so a job
    accepted before a merge never carries that merge. ``status`` is the record's own status
    (``success``, ``failed``, ``superseded`` or a non-terminal one such as ``in_progress``).
    ``git_ref`` is the command's ref: ``origin/dev`` builds the dev head at acceptance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    runtime_lane: str = Field(min_length=1)
    accepted_at: datetime
    status: str = Field(min_length=1)
    git_ref: str = Field(min_length=1)
