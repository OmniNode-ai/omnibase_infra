# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Budget cap model for autonomous off-peak evaluations.

Related:
    - OMN-6795: Define eval task models and enums

.. versionadded:: 0.29.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEvalBudgetCap(BaseModel):
    """Budget cap for autonomous evaluation tasks.

    Controls spending across evaluation windows to prevent runaway costs.

    Attributes:
        max_tokens_per_window: Maximum total tokens across all tasks in a window.
        max_cost_usd_per_window: Maximum total cost in USD per window.
        window_hours: Duration of the budget window in hours.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens_per_window: int = Field(
        default=500_000,
        ge=1000,
        description="Max tokens per budget window.",
    )
    max_cost_usd_per_window: float = Field(
        default=1.0,
        ge=0.01,
        description="Max cost in USD per budget window.",
    )
    window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Budget window duration in hours.",
    )


__all__: list[str] = ["ModelEvalBudgetCap"]
