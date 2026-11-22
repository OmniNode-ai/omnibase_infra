# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelJobCommand(BaseModel):
    """The two fields of a job's command this rule reads."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    runtime_lane: str = ""
    requested_by: str = ""
