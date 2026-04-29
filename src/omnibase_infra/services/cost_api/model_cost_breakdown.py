# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost breakdown response model."""

from __future__ import annotations

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase
from omnibase_infra.services.cost_api.model_cost_breakdown_item import (
    ModelCostBreakdownItem,
)
from omnibase_infra.services.cost_api.model_types import AggregationWindow


class ModelCostBreakdown(ModelCostApiBase):
    """Collection of cost groups for a single aggregate window."""

    window: AggregationWindow
    items: list[ModelCostBreakdownItem]
