# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The manifest of one dev roll command: which command, and who issued it."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDevRollManifest(BaseModel):
    """Published once when the deploy agent accepts a dev runtime or full roll.

    ``job_id`` is the command's correlation id; ``issuer`` is the command's
    ``requested_by``, the typed issuer cell until provenance fields exist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    job_id: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    git_ref: str
    scope: str
    requested_at: datetime | None
    accepted_at: datetime
    open_pr: str
    rollback: bool
