# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Wire-format issue model for the local project tracker adapter.

Mirrors omnibase_spi.contracts.services.contract_project_tracker_types.ModelIssue
(omnibase_spi >= 0.21.0). Kept here because installed omnibase_spi 0.20.x
predates the contracts/services module.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ModelStubIssue(BaseModel):
    model_config = {"frozen": True, "extra": "allow"}

    id: str
    identifier: str
    title: str
    description: str | None = None
    state: str
    priority: str | None = None
    assignee: str | None = None
    labels: list[str] = []
    team: str | None = None
    project_id: str | None = None
    url: str | None = None
    created_at: datetime
    updated_at: datetime
