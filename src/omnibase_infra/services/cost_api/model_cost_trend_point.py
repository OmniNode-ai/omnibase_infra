# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost trend point response model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase


class ModelCostTrendPoint(ModelCostApiBase):
    """Cost and usage for a real event-time bucket."""

    bucket_start: str
    total_cost_usd: Decimal = Field(default=Decimal("0.000000"))
    total_tokens: int = 0
    call_count: int = 0
