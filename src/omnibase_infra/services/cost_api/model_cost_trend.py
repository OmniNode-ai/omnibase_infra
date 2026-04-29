# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost trend response model."""

from __future__ import annotations

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase
from omnibase_infra.services.cost_api.model_cost_trend_point import ModelCostTrendPoint
from omnibase_infra.services.cost_api.model_types import TrendBucket


class ModelCostTrend(ModelCostApiBase):
    """Time series derived from raw call metric timestamps."""

    bucket: TrendBucket
    days: int
    points: list[ModelCostTrendPoint]
