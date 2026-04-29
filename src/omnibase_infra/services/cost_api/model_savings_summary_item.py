# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Savings summary item grouped by local model."""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase


class ModelSavingsSummaryItem(ModelCostApiBase):
    """Savings totals grouped by local model."""

    model_local: str
    total_savings_usd: Decimal = Field(default=Decimal("0.000000"))
    local_cost_usd: Decimal = Field(default=Decimal("0.000000"))
    cloud_cost_usd: Decimal = Field(default=Decimal("0.000000"))
    session_count: int = 0
