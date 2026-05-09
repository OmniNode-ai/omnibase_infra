# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for a single branch in an interactive onboarding transition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType


class ModelTransitionBranch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    condition: str | None = Field(default=None)
    next: str
    set_state: dict[str, JsonType] = Field(default_factory=dict)


__all__ = ["ModelTransitionBranch"]
