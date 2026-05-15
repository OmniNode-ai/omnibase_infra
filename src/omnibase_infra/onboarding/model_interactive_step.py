# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for a single interactive onboarding step."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.onboarding.enum_interactive_step_type import EnumInteractiveStepType


class ModelInteractiveStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Step identifier")
    prompt: str = Field(description="User-facing prompt text")
    type: EnumInteractiveStepType
    options: list[str] = Field(default_factory=list)
    condition: str | None = Field(default=None)
    required: bool = Field(default=True)
    action: str | None = Field(default=None)
    produces_capabilities: list[str] = Field(default_factory=list)


__all__ = ["ModelInteractiveStep"]
