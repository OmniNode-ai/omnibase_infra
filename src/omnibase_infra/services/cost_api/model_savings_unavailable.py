# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Savings unavailable response model."""

from __future__ import annotations

from typing import Literal

from omnibase_infra.services.cost_api.model_cost_api_base import ModelCostApiBase


class ModelSavingsUnavailable(ModelCostApiBase):
    """Stable unavailable response for future savings estimation routes."""

    status: Literal["unavailable"] = "unavailable"
    code: Literal["savings_summary_not_implemented"] = "savings_summary_not_implemented"
    message: str
