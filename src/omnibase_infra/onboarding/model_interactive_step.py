# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pydantic model for a single interactive onboarding step."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelInteractiveStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Step identifier")
    prompt: str = Field(description="User-facing prompt text")
    type: Literal["choice", "multi_choice", "text", "action"]
    options: list[str] = Field(default_factory=list)
    condition: str | None = Field(default=None)
    required: bool = Field(default=True)
    secret: bool = Field(
        default=False,
        description=(
            "Collect this step through the masked adapter path. The value is "
            "never echoed, never stored in the step-result receipt, and may "
            "only reach disk through the policy's credentials_output block."
        ),
    )
    action: str | None = Field(default=None)
    produces_capabilities: list[str] = Field(default_factory=list)


__all__ = ["ModelInteractiveStep"]
