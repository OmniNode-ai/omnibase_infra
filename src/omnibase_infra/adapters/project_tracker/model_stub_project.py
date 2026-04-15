# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Wire-format project model for the local project tracker adapter.

Mirrors omnibase_spi.contracts.services.contract_project_tracker_types.ModelProject
(omnibase_spi >= 0.21.0). Kept here because installed omnibase_spi 0.20.x
predates the contracts/services module.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelStubProject(BaseModel):
    model_config = {"frozen": True, "extra": "allow"}

    id: str
    name: str
    description: str | None = None
    state: str | None = None
    progress: float = 0.0
    url: str | None = None
