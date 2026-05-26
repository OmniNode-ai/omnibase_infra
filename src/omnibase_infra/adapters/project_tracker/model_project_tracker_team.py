# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectTrackerTeam(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str
    name: str
    key: str


class ModelProjectTrackerLabel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str
    name: str
    color: str | None = Field(default=None)
    team_id: str | None = Field(default=None)


class ModelProjectTrackerIssueStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str
    name: str
    type: str
    team_id: str | None = Field(default=None)


__all__: list[str] = [
    "ModelProjectTrackerIssueStatus",
    "ModelProjectTrackerLabel",
    "ModelProjectTrackerTeam",
]
