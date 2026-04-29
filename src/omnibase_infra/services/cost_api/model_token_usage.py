# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Token usage response model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase
from omnibase_infra.services.cost_api.model_types import AggregationWindow


class ModelTokenUsage(ModelCostApiBase):
    """Token usage totals from canonical session aggregate rows."""

    window: AggregationWindow
    total_tokens: int = 0
    call_count: int = 0
    average_tokens_per_call: Decimal = Field(default=Decimal("0.000000"))
