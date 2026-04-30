# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost summary response model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase
from omnibase_infra.services.cost_api.model_types import AggregationWindow


class ModelCostSummary(ModelCostApiBase):
    """Top-level LLM cost summary for canonical session aggregate rows."""

    window: AggregationWindow
    total_cost_usd: Decimal = Field(default=Decimal("0.000000"))
    total_tokens: int = 0
    call_count: int = 0
    estimated_coverage_pct: Decimal | None = None
