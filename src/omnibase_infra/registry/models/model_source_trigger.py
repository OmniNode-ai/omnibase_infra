# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSourceTrigger(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    pattern: str
    change_scope: Literal["structural", "semantic", "any"] = "any"
    match_fields: list[str] = Field(default_factory=list)
