# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_job_status import EnumJobStatus
from omnibase_infra.models.model_job_command import ModelJobCommand
from omnibase_infra.models.model_job_lineage import ModelJobLineage


class ModelJobRecord(BaseModel):
    """The fields of an agent job record (``JobState``) this rule reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    correlation_id: UUID
    command: ModelJobCommand
    status: EnumJobStatus
    completed_at: datetime | None = None
    errors: tuple[str, ...] = ()
    lineage: ModelJobLineage | None = None
