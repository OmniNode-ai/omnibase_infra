# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.adapters.project_tracker.enum_project_tracker_issue_status_type import (
    EnumProjectTrackerIssueStatusType,
)


class ModelProjectTrackerIssueStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str
    name: str
    type: EnumProjectTrackerIssueStatusType
    team_id: str | None = Field(default=None)


__all__: list[str] = [
    "ModelProjectTrackerIssueStatus",
]
