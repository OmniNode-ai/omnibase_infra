# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelProjectTrackerTeam(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    id: str
    name: str
    key: str


__all__: list[str] = [
    "ModelProjectTrackerTeam",
]
