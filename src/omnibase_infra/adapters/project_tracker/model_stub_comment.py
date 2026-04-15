# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Wire-format comment model for the local project tracker adapter.

Mirrors omnibase_spi.contracts.services.contract_project_tracker_types.ModelComment
(omnibase_spi >= 0.21.0). Kept here because installed omnibase_spi 0.20.x
predates the contracts/services module.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

_SCHEMA_VERSION = "1.0"


class ModelStubComment(BaseModel):
    model_config = {"frozen": True, "extra": "allow"}

    schema_version: str = Field(default=_SCHEMA_VERSION)
    id: str
    body: str
    author: str
    created_at: datetime
