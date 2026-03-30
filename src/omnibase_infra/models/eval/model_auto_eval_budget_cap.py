# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-eval budget cap model.

Defines budget constraints for autonomous eval execution. The runner
checks these caps before executing each task and refuses tasks when
the budget is exhausted within the current time window.

Related:
    - OMN-6795: Define eval task models and enums
    - ServiceAutoEvalRunner: Enforces these caps at runtime
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelAutoEvalBudgetCap(BaseModel):
    """Budget constraints for autonomous LLM evaluation windows.

    Attributes:
        max_cost_usd: Maximum total cost in USD within the time window.
        max_calls: Maximum number of LLM calls within the time window.
        time_window_hours: Duration of the budget window in hours.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    max_cost_usd: float = Field(
        ...,
        gt=0.0,
        description="Maximum total cost in USD within the time window.",
    )
    max_calls: int = Field(
        ...,
        gt=0,
        description="Maximum number of LLM calls within the time window.",
    )
    time_window_hours: float = Field(
        default=24.0,
        gt=0.0,
        description="Duration of the budget window in hours.",
    )


__all__: list[str] = ["ModelAutoEvalBudgetCap"]
