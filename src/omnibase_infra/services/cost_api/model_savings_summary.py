# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Savings summary response model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase
from omnibase_infra.services.cost_api.model_savings_summary_item import (
    ModelSavingsSummaryItem,
)
from omnibase_infra.services.cost_api.model_types import AggregationWindow


class ModelSavingsSummary(ModelCostApiBase):
    """Top-level savings summary for projected savings estimates."""

    window: AggregationWindow
    total_savings_usd: Decimal = Field(default=Decimal("0.000000"))
    local_cost_usd: Decimal = Field(default=Decimal("0.000000"))
    cloud_cost_usd: Decimal = Field(default=Decimal("0.000000"))
    session_count: int = 0
    items: list[ModelSavingsSummaryItem] = Field(default_factory=list)
