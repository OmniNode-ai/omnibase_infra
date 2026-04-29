# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner-cost pricing policy loaded from the pricing manifest."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerCostPolicy(BaseModel):
    """Pricing policy for CI runner avoidance estimates."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    github_hosted_per_minute_usd: float = Field(
        ...,
        ge=0.0,
        description="GitHub-hosted runner baseline cost per runner minute in USD.",
    )


__all__: list[str] = ["ModelRunnerCostPolicy"]
